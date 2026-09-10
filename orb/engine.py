from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import pairwise

import polars as pl
from firstrate_data import AssetType, EquitiesAdjustment, Store, Timeframe, TradingHours
from tqdm.auto import tqdm

from orb.portfolio import Portfolio
from orb.results import BacktestResult
from orb.schemas import BARS_SCHEMA, ORDER_SCHEMA, TIMEZONE, validate_schema
from orb.strategy import ORBStrategy

ONE_DAY = timedelta(days=1)
SETUP_STAGES = 4

# Wilder's average carries every true range before it, so the warmup is long
# enough for its seed to weigh under a percent by the first traded session.
N_OF_WARMUP_DAYS = timedelta(days=180)


@dataclass(frozen=True)
class BarsChunk:
    """One period of minute bars to read, and the tickers worth reading in it."""

    # A backtest reads more minute bars than fit in RAM at once, so it reads
    # them one year at a time.
    # 1min bars per session = (30 + 60*6) = 390
    # 390 * 252 = 98.280 bars a year per ticker.
    INTERVAL = "1y"

    start: date
    end: date
    tickers: tuple[str, ...]


@dataclass
class BacktestEngine:
    """Reads bars in chunks, turns each chunk into orders, settles them by session.

    The strategy runs as polars expressions. The stop loss test walks the held
    bars of a session once instead of joining every held bar back against its
    entry.
    """

    store: Store
    strategy: ORBStrategy
    start: date
    end: date
    portfolio: Portfolio = field(default_factory=Portfolio)

    # left unspecified to trade the whole stock universe
    asset_type: AssetType = AssetType.STOCK
    tickers: tuple[str, ...] | None = None

    chunk_interval: str = BarsChunk.INTERVAL
    show_progress: bool = True

    def run(self) -> BacktestResult:
        run_started = datetime.now(tz=TIMEZONE)
        tqdm_stages = self._progress_bar("read daily bars", total=SETUP_STAGES)

        self.portfolio.reset()
        # The paper takes daily bars from CRSP, which are split-adjusted, so a
        # split does not leave a price jump that inflates the ATR.
        split_adjusted_daily_bars = self._daily_bars(EquitiesAdjustment.SPLIT)
        unadjusted_daily_bars = self._daily_bars(EquitiesAdjustment.UNADJUSTED)

        tqdm_stages.set_description("select eligible ticker sessions")
        tqdm_stages.update()
        self._eligible = self.strategy.eligible_ticker_sessions(
            split_adjusted_daily_bars,
            unadjusted_daily_bars,
        )

        tqdm_stages.set_description("count sessions")
        tqdm_stages.update()
        iterations = self._sessions_count(unadjusted_daily_bars)

        tqdm_stages.set_description("plan chunks")
        tqdm_stages.update()
        chunks_to_read = list(self._chunks_to_read())
        tqdm_stages.update()
        tqdm_stages.close()

        chunks = self._progress_bar(
            "read minute bars",
            total=len(chunks_to_read),
            unit="chunk",
        )
        orders_by_chunk = []
        for chunk in chunks_to_read:
            chunks.set_postfix(
                chunk=f"{chunk.start} to {chunk.end}",
                tickers=len(chunk.tickers),
            )
            orders_by_chunk.append(self._orders_from_chunk(chunk))
            chunks.update()
        chunks.close()
        orders = (
            pl.concat(orders_by_chunk)
            if orders_by_chunk
            else pl.DataFrame(schema=ORDER_SCHEMA)
        )

        orders_by_session = orders.partition_by("session")
        settlement = self._progress_bar(
            "settle orders",
            total=len(orders_by_session),
            unit="session",
        )
        total_positions = 0
        for session_orders in orders_by_session:
            total_positions += self.portfolio.settle_session_positions(session_orders)
            settlement.update()
        settlement.close()

        run_finished = datetime.now(tz=TIMEZONE)

        return BacktestResult(
            run_started,
            run_finished,
            self.start,
            self.end,
            run_finished - run_started,
            iterations,
            orders.height,
            total_positions,
        )

    def _sessions_count(self, daily_bars: pl.DataFrame) -> int:
        sessions_count: pl.DataFrame = (
            daily_bars.lazy()
            .filter(pl.col("ts").dt.date() >= self.start)
            .select(pl.col("ts").dt.date().n_unique())
            .collect()
        )
        return int(sessions_count.item())

    def _chunks_to_read(self) -> Iterator[BarsChunk]:
        chunk_starts = pl.date_range(
            self.start,
            self.end,
            interval=self.chunk_interval,
            eager=True,
        ).to_list()
        for start, next_start in pairwise([*chunk_starts, self.end + ONE_DAY]):
            end = next_start - ONE_DAY
            if tickers := self._eligible_tickers_in_chunk(start, end):
                yield BarsChunk(start, end, tickers)

    def _eligible_tickers_in_chunk(self, start: date, end: date) -> tuple[str, ...]:
        in_chunk = self._eligible.filter(pl.col("session").is_between(start, end))
        return tuple(in_chunk["ticker"].unique().to_list())

    @validate_schema(ORDER_SCHEMA)
    def _orders_from_chunk(self, chunk: BarsChunk) -> pl.DataFrame:
        bars = self._eligible_minute_bars_from_chunk(chunk)
        breakouts = self.strategy.opening_range_breakouts(bars)
        bars_after_range = self._bars_after_opening_range(bars, breakouts)
        entries = self._entries(bars_after_range)
        exits = self._exits(bars_after_range, entries)
        return self._orders(entries, exits)

    def _eligible_minute_bars_from_chunk(self, chunk: BarsChunk) -> pl.LazyFrame:
        return (
            self._minute_bars_from_chunk(chunk)
            .lazy()
            .with_columns(session=pl.col("ts").dt.date())
            .join(self._eligible.lazy(), on=("ticker", "session"), how="inner")
            .sort("ticker", "session", "ts")
            .with_columns(
                bar_number=pl.int_range(1, pl.len() + 1).over("ticker", "session"),
                session_close=pl.col("close").last().over("ticker", "session"),
                # The ATR arrives as a fraction of the opening price, so the
                # here atr is converted back into dollars.
                atr=pl.col("atr_pct")
                * pl.col("open").first().over("ticker", "session"),
            )
        )

    def _bars_after_opening_range(
        self,
        bars: pl.LazyFrame,
        breakouts: pl.LazyFrame,
    ) -> pl.LazyFrame:
        return (
            bars.filter(self.strategy.is_after_opening_range())
            .join(
                breakouts.select(
                    "ticker",
                    "session",
                    "direction",
                    "trigger_price",
                    "opening_range_extreme",
                ),
                on=("ticker", "session"),
                how="inner",
            )
            .with_columns(
                has_reached_trigger=self.strategy.has_reached_trigger(),
                fill_price=self.strategy.fill_price(),
            )
        )

    def _entries(self, bars_after_range: pl.LazyFrame) -> pl.LazyFrame:
        return (
            bars_after_range.filter("has_reached_trigger")
            .group_by("ticker", "session")
            .agg(
                pl.col("atr").first(),
                pl.col("direction").first(),
                pl.col("session_close").first(),
                pl.col("opening_range_extreme").first(),
                entry_bar_number=pl.col("bar_number").min(),
                entry_price=pl.col("fill_price").sort_by("bar_number").first(),
            )
            .with_columns(stop_loss=self.strategy.stop_loss())
            .with_columns(
                risk_per_share=(pl.col("entry_price") - pl.col("stop_loss"))
                * pl.col("direction"),
            )
            # An entry sitting on its own stop has no risk per share, so its
            # position size would be unbounded.
            .filter(pl.col("risk_per_share") > 0)
            .with_columns(target_price=self.strategy.take_profit())
        )

    def _exits(
        self,
        bars_after_range: pl.LazyFrame,
        entries: pl.LazyFrame,
    ) -> pl.LazyFrame:
        held_bars = bars_after_range.join(
            entries.select(
                "ticker",
                "session",
                "entry_bar_number",
                "stop_loss",
                "target_price",
            ),
            on=("ticker", "session"),
            how="inner",
        ).filter(self.strategy.can_exit_on_bar())

        return held_bars.group_by("ticker", "session").agg(
            stop_bar_number=pl.col("bar_number")
            .filter(self.strategy.has_reached_stop_loss())
            .min(),
            target_bar_number=pl.col("bar_number")
            .filter(self.strategy.has_reached_take_profit())
            .min(),
            stop_fill_price=self.strategy.stop_fill_price()
            .filter(self.strategy.has_reached_stop_loss())
            .sort_by(pl.col("bar_number").filter(self.strategy.has_reached_stop_loss()))
            .first(),
        )

    def _orders(self, entries: pl.LazyFrame, exits: pl.LazyFrame) -> pl.DataFrame:
        # A bar holding both the stop and the target counts as stopped: the bar
        # does not say which price printed first.
        is_stopped = pl.col("stop_bar_number").is_not_null() & (
            pl.col("target_bar_number").is_null()
            | (pl.col("stop_bar_number") <= pl.col("target_bar_number"))
        )
        exit_price = (
            pl.when(pl.col("is_stopped"))
            .then(pl.col("stop_fill_price"))
            .when(pl.col("target_bar_number").is_not_null())
            .then(pl.col("target_price"))
            .otherwise(pl.col("session_close"))
        )
        return (
            # Left join keeps an entry with no exits row instead of dropping it:
            # its null stop_bar_number reads as not stopped, so it exits at the
            # session close.
            entries.join(exits, on=("ticker", "session"), how="left")
            .with_columns(is_stopped=is_stopped)
            .with_columns(exit_price=exit_price)
            .with_columns(
                units_of_risk_r=(pl.col("exit_price") - pl.col("entry_price"))
                * pl.col("direction")
                / pl.col("risk_per_share"),
            )
            .select(ORDER_SCHEMA.keys())
            .sort("session", "ticker")
            .collect()
        )

    # ------------------------------------------------------------------
    # querying bars
    # ------------------------------------------------------------------

    @validate_schema(BARS_SCHEMA)
    def _daily_bars(self, adjustment: EquitiesAdjustment) -> pl.DataFrame:
        """Return a Dataframe containing daily OHLC bars of `self.tickers`, covering the full backtest period."""
        return self.store.bars(
            asset_type=self.asset_type,
            timeframe=Timeframe.DAY_1,
            adjustment=adjustment,
            ticker=self.tickers,
            start=self.start - N_OF_WARMUP_DAYS,
            end=self.end,
        ).pl()

    @validate_schema(BARS_SCHEMA)
    def _minute_bars_from_chunk(self, chunk: BarsChunk) -> pl.DataFrame:
        return self.store.bars(
            asset_type=self.asset_type,
            timeframe=Timeframe.MIN_1,
            # pg 8: "this intraday data remained unadjusted for stock splits or dividend"
            adjustment=EquitiesAdjustment.UNADJUSTED,
            ticker=chunk.tickers,
            start=chunk.start,
            end=chunk.end,
            hours=TradingHours.REGULAR,
        ).pl()

    # ------------------------------------------------------------------
    # tqdm progress bar
    # ------------------------------------------------------------------

    def _progress_bar(self, description: str, total: int, unit: str = "step") -> tqdm:
        return tqdm(
            total=total,
            desc=description,
            unit=unit,
            leave=True,
            disable=not self.show_progress,
        )
