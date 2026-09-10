from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl
from firstrate_data import AssetType, EquitiesAdjustment, Store, Timeframe


@dataclass
class BacktestResult:
    run_started: datetime
    run_finished: datetime
    backtest_start: date
    backtest_end: date
    elapsed_time: timedelta
    iterations: int
    total_orders: int
    total_positions: int


def buy_and_hold_balance(
    ticker: str,
    start: date = date(2016, 1, 1),
    end: date = date(2023, 12, 31),
    starting_balance: float = 25_000.0,
    store: Store | None = None,
    asset_type: AssetType = AssetType.ETF,
) -> pl.DataFrame:
    daily_bars = (
        (store or Store.from_env())
        .bars(
            asset_type=asset_type,
            timeframe=Timeframe.DAY_1,
            adjustment=EquitiesAdjustment.SPLIT,
            ticker=ticker,
            start=start - timedelta(days=15),
            end=end,
        )
        .pl()
        .select(
            session=pl.col("ts").dt.date(),
            close=pl.col("close").cast(pl.Float64),
        )
        .sort("session")
    )
    session_return = pl.col("close") / pl.col("close").shift()
    return (
        daily_bars.with_columns(session_return=session_return)
        .filter(pl.col("session") >= start)
        .select(
            "session",
            balance=starting_balance
            * pl.col("session_return").fill_null(1.0).cum_prod(),
        )
    )
