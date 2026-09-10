from dataclasses import dataclass
from datetime import time

import polars as pl

from orb.schemas import ELIGIBLE_TICKER_SESSIONS_SCHEMA, validate_schema

LOOKBACK_SESSIONS_FOR_VOLUME_AND_ATR = 14

MIN_OPENING_PRICE = 5
MIN_AVG_TRADING_VOLUME = 1_000_000
MIN_ATR = 0.5

REGULAR_SESSION_OPEN_MINUTES = 9 * 60 + 30

IS_LONG = pl.col("direction") == 1


@dataclass
class ORBStrategy:
    opening_range_len_1_min_bars: int
    # if we buy we set SL = entry - daily_atr * daily_atr_percentage
    # if we sell we set SL = entry + daily_atr * daily_atr_percentage
    daily_atr_percentage: float
    # full universe orb paper waits for a break out of the opening range (True)
    # ORB on TQQ paper doesn't (False)
    wait_for_breakout: bool = True
    # Full ORB uses this criteria (True), TQQQ ORB doesn't (False).
    apply_eligibility_criteria: bool = True
    # TQQQ ORB stops at the opposite extreme of the opening range (True),
    # full ORB stops at `daily_atr_percentage` of the daily ATR (False).
    stop_at_opening_range_extreme: bool = False
    # Leave None to hold positions until EoD
    tp_units_of_risk_r: float | None = None

    def opening_range_breakouts(self, session_bars: pl.LazyFrame) -> pl.LazyFrame:
        is_in_opening_range = pl.col("ts").dt.time() < self._opening_range_end_time()
        is_up = pl.col("opening_range_close") > pl.col("opening_range_open")
        return (
            session_bars.group_by("ticker", "session")
            .agg(
                pl.col("session_close").first(),
                pl.col("atr").first(),
                opening_range_open=pl.col("open").filter(is_in_opening_range).first(),
                opening_range_close=pl.col("close").filter(is_in_opening_range).last(),
                opening_range_high=pl.col("high").filter(is_in_opening_range).max(),
                opening_range_low=pl.col("low").filter(is_in_opening_range).min(),
            )
            # Drops the sessions whose opening range closed where it opened
            .filter(pl.col("opening_range_close") != pl.col("opening_range_open"))
            .with_columns(
                direction=pl.when(is_up).then(1).otherwise(-1),
                trigger_price=pl.when(is_up)
                .then(pl.col("opening_range_high"))
                .otherwise(pl.col("opening_range_low")),
                opening_range_extreme=pl.when(is_up)
                .then(pl.col("opening_range_low"))
                .otherwise(pl.col("opening_range_high")),
            )
        )

    def _opening_range_end_time(self) -> time:
        # ponytail: 09:30 is the US equity/ETF open; another exchange would need
        # its session pulled from the store.
        end = REGULAR_SESSION_OPEN_MINUTES + self.opening_range_len_1_min_bars
        return time(end // 60, end % 60)

    def is_after_opening_range(self) -> pl.Expr:
        return pl.col("ts").dt.time() >= self._opening_range_end_time()

    def has_reached_trigger(self) -> pl.Expr:
        if not self.wait_for_breakout:
            # Entry at the open of the bar right after the opening range, in the
            # direction of the opening range.
            return pl.lit(value=True)
        return (
            pl.when(IS_LONG)
            .then(pl.col("high") >= pl.col("trigger_price"))
            .otherwise(pl.col("low") <= pl.col("trigger_price"))
        )

    def fill_price(self) -> pl.Expr:
        if not self.wait_for_breakout:
            return pl.col("open")
        # Fills a long at the worse of the trigger and the bar's open, so a bar
        # that gapped through the trigger pays the gap instead of the stale
        # trigger price.
        return (
            pl.when(IS_LONG)
            .then(pl.max_horizontal("trigger_price", "open"))
            # The mirror for shorts: the lower of the two.
            .otherwise(pl.min_horizontal("trigger_price", "open"))
        )

    def can_exit_on_bar(self) -> pl.Expr:
        if not self.wait_for_breakout:
            # Entry is the bar's own open, so the whole bar prints while the
            # position is held.
            return pl.col("bar_number") >= pl.col("entry_bar_number")
        # A breakout fills partway through its bar, so that bar's low and high
        # carry prices from before the position existed. Its close does not: it
        # is the last print of the bar, so it comes after the fill, and a close
        # beyond the stop is a breach the position certainly lived through.
        is_after_entry_bar = pl.col("bar_number") > pl.col("entry_bar_number")
        return is_after_entry_bar | (
            (pl.col("bar_number") == pl.col("entry_bar_number"))
            & self._has_closed_beyond_stop_loss()
        )

    def _has_closed_beyond_stop_loss(self) -> pl.Expr:
        return (
            pl.when(IS_LONG)
            .then(pl.col("close") <= pl.col("stop_loss"))
            .otherwise(pl.col("close") >= pl.col("stop_loss"))
        )

    def stop_fill_price(self) -> pl.Expr:
        """Price a stop order gets on the bar that breaches it.

        The mirror of `fill_price`: a bar whose open is already past the stop
        pays that open, not the stale stop price. On the entry bar the open
        printed before the position existed, so the stop itself is the only
        price the breach can be read at.
        """
        open_after_entry_bar = (
            pl.when(pl.col("bar_number") > pl.col("entry_bar_number"))
            .then(pl.col("open"))
            .otherwise(pl.col("stop_loss"))
        )
        return (
            pl.when(IS_LONG)
            .then(pl.min_horizontal(open_after_entry_bar, pl.col("stop_loss")))
            .otherwise(pl.max_horizontal(open_after_entry_bar, pl.col("stop_loss")))
        )

    def has_reached_stop_loss(self) -> pl.Expr:
        return (
            pl.when(IS_LONG)
            .then(pl.col("low") <= pl.col("stop_loss"))
            .otherwise(pl.col("high") >= pl.col("stop_loss"))
        )

    def has_reached_take_profit(self) -> pl.Expr:
        return (
            pl.when(IS_LONG)
            .then(pl.col("high") >= pl.col("target_price"))
            .otherwise(pl.col("low") <= pl.col("target_price"))
        )

    def stop_loss(self) -> pl.Expr:
        if self.stop_at_opening_range_extreme:
            return pl.col("opening_range_extreme")
        return (
            pl.col("entry_price")
            - pl.col("direction") * pl.col("atr") * self.daily_atr_percentage
        )

    def take_profit(self) -> pl.Expr:
        if self.tp_units_of_risk_r is None:
            return pl.lit(None, dtype=pl.Float64)
        return (
            pl.col("entry_price")
            + pl.col("direction") * pl.col("risk_per_share") * self.tp_units_of_risk_r
        )

    @validate_schema(ELIGIBLE_TICKER_SESSIONS_SCHEMA)
    def eligible_ticker_sessions(
        self,
        split_adjusted_daily_bars: pl.DataFrame,
        unadjusted_daily_bars: pl.DataFrame,
    ) -> pl.DataFrame:
        """Return a Dataframe where each row is one specific trading session for a specific ticker that passes the eligibility criteria outlined in section 2.1 of the paper.

        Parameters
        ----------
        split_adjusted_daily_bars : pl.DataFrame
            Daily bars for many tickers, matching `BARS_SCHEMA`, adjusted for
            splits. A split leaves no price jump in them, so the true range
            carries no fake gap and the ATR stays clean across it.
        unadjusted_daily_bars : pl.DataFrame
            The same sessions at the prices and share counts that traded, which
            is the scale the $5 opening price, the 1,000,000 share volume and
            the $0.50 ATR thresholds are quoted in.

        Returns
        -------
        pl.DataFrame
            Each row is one ticker session that passes the filter, typed as
            `ELIGIBLE_TICKER_SESSIONS_SCHEMA`. `atr_pct` is the ATR as a
            fraction of the opening price, which carries no price scale and so
            applies to the unadjusted intraday bars unchanged.

        """
        atr_pct = (
            split_adjusted_daily_bars.lazy()
            .sort("ticker", "ts")
            .select(
                "ticker",
                "ts",
                atr_pct=(_atr_expr() / pl.col("open")).shift(1).over("ticker"),
            )
        )
        return (
            unadjusted_daily_bars.lazy()
            .sort("ticker", "ts")
            .with_columns(
                avg_trading_volume=pl.col("volume")
                .rolling_mean(LOOKBACK_SESSIONS_FOR_VOLUME_AND_ATR)
                .shift(1)
                .over("ticker"),
            )
            .join(atr_pct, on=("ticker", "ts"), how="inner")
            .with_columns(atr=pl.col("atr_pct") * pl.col("open"))
            # these are the criteria
            .filter(self._eligibility_criteria())
            .select("ticker", pl.col("ts").dt.date().alias("session"), "atr_pct")
            .collect()
        )

    def _eligibility_criteria(self) -> pl.Expr:
        if not self.apply_eligibility_criteria:
            return pl.col("atr_pct").is_not_null()
        return (
            (pl.col("open") > MIN_OPENING_PRICE)
            & (pl.col("avg_trading_volume") >= MIN_AVG_TRADING_VOLUME)
            & (pl.col("atr") > MIN_ATR)
        )


def _true_range() -> pl.Expr:
    """One ticker's true range: the widest of today's range and the two gaps to yesterday's close.

    See TR in https://en.wikipedia.org/wiki/Average_true_range.
    """
    previous_close = pl.col("close").shift()
    return pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous_close).abs(),
        (pl.col("low") - previous_close).abs(),
    )


def _atr_expr() -> pl.Expr:
    """Wilder's running average of one ticker's true ranges.

    The first value is the simple mean of the first `period` true ranges, and
    every value after it moves a fourteenth of the way to the newest one. The
    sessions before that seed carry no average, so they are null.
    """
    period = LOOKBACK_SESSIONS_FOR_VOLUME_AND_ATR
    true_range = _true_range()
    session_numbers = pl.int_range(pl.len())
    seeded = (
        pl.when(session_numbers < period - 1)
        .then(pl.lit(None, dtype=pl.Float64))
        .when(session_numbers == period - 1)
        .then(true_range.rolling_mean(period))
        .otherwise(true_range)
    )
    return seeded.ewm_mean(alpha=1 / period, adjust=False, ignore_nulls=True)
