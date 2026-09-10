from collections.abc import Callable, Mapping
from functools import wraps
from typing import ParamSpec
from zoneinfo import ZoneInfo

import polars as pl

P = ParamSpec("P")

TIMEZONE = ZoneInfo("America/New_York")

BARS_SCHEMA = {
    "ts": pl.Datetime(time_unit="us", time_zone=TIMEZONE),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "open_interest": pl.Int64,
    "asset_type": pl.String,
    # "dataset": pl.String,
    "adjustment": pl.String,
    "timeframe": pl.String,
    "ticker": pl.String,
}
ORDER_SCHEMA = {
    "ticker": pl.String,
    "session": pl.Date,  # TODO rename to date?
    "direction": pl.Int32,
    "entry_price": pl.Float64,
    "stop_loss": pl.Float64,
    "atr": pl.Float64,
    "risk_per_share": pl.Float64,
    "is_stopped": pl.Boolean,
    "exit_price": pl.Float64,
    "units_of_risk_r": pl.Float64,
}

ELIGIBLE_TICKER_SESSIONS_SCHEMA = {
    "ticker": pl.String,
    "session": pl.Date,
    "atr_pct": pl.Float64,
}


def validate_schema(
    schema: Mapping[str, pl.DataType | type[pl.DataType]],
) -> Callable[[Callable[P, pl.DataFrame]], Callable[P, pl.DataFrame]]:
    """Match the decorated function's Dataframe against `schema`.

    The returned Dataframe carries `schema`'s columns in its order and dtypes,
    and a column the schema does not name raises.
    """

    def decorator(
        build_dataframe: Callable[P, pl.DataFrame],
    ) -> Callable[P, pl.DataFrame]:
        @wraps(build_dataframe)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> pl.DataFrame:
            return build_dataframe(*args, **kwargs).match_to_schema(schema)

        return wrapper

    return decorator
