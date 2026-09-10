# quantifyig PnL
from dataclasses import dataclass, field
from datetime import date

import polars as pl


@dataclass
class Portfolio:
    starting_balance: float = 25_000.0
    risk_per_position: float = 0.01
    max_leverage: float = 4.0
    # 0.0035 ORB full, 0.0005 TQQQ
    commission_per_share: float = 0.0035
    balance: float = field(init=False)
    balance_by_session: list[tuple[date, float]] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.balance = self.starting_balance
        self.balance_by_session = []

    def equity_curve(self) -> pl.DataFrame:
        return pl.DataFrame(
            self.balance_by_session,
            schema={"session": pl.Date, "balance": pl.Float64},
            orient="row",
        )

    def settle_session_positions(self, session_positions: pl.DataFrame) -> int:
        if session_positions.is_empty():
            return 0

        shares = (
            self.balance * self.risk_per_position / session_positions["risk_per_share"]
        )
        exposure = (shares * session_positions["entry_price"]).sum()
        max_exposure = self.balance * self.max_leverage
        if exposure > max_exposure:
            shares = shares * (max_exposure / exposure)

        positions = session_positions.with_columns(shares=shares).filter(
            pl.col("shares") > 0,
        )

        if not positions.is_empty():
            gross_pnl = (
                (positions["exit_price"] - positions["entry_price"])
                * positions["direction"]
                * positions["shares"]
            )
            commission = 2 * positions["shares"] * self.commission_per_share
            self.balance += float((gross_pnl - commission).sum())

        self.balance_by_session.append((session_positions["session"][0], self.balance))
        return positions.height
