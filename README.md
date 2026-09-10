# Opening Range Breakout (ORB) paper reproduction

Backtests the opening range breakout strategy of two papers by Zarattini et al.:

- [Can Day Trading Really Be Profitable?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416622): 5-minute ORB on QQQ and TQQQ. Reproduced in `orb_qqq.ipynb`.
- [A Profitable Day Trading Strategy for the U.S. Equity Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284): the same strategy over the U.S. stock universe. Reproduced in `orb_full.ipynb`.

`ORBStrategy` implements both variants: the equity-market paper waits for a breakout of the opening range and applies the section 2.1 eligibility filters, the QQQ paper enters at the open of the bar after the opening range and stops at its opposite extreme.

> [!NOTE]
> Work in progress. `orb_full.ipynb` currently runs the section 2.1 base strategy only. The section 4 *Stocks in Play* refinement is not implemented yet.

## Data

The backtest reads 1-minute and daily bars from a [FirstRate Data](https://firstratedata.com) store built with [`firstrate_data`](https://github.com/MK27MK/firstrate_data). A FirstRate subscription is required; no market data is included in this repo.

Set `FIRSTRATE_DATA_PATH` to the store directory and `FIRSTRATE_USERID` to the FirstRate user id, in a `.env` file at the repo root:

```
FIRSTRATE_DATA_PATH=path/to/the/store
FIRSTRATE_USERID=your-id-here
```

`FIRSTRATE_USERID` is only read when downloading. `Store.from_env()`, which both notebooks call, needs `FIRSTRATE_DATA_PATH` alone.

For the tickers and dates being backtested, the store must contain: 1-minute unadjusted bars, daily unadjusted bars, and daily split-adjusted bars. The daily bars are read from 180 days before the backtest start, to warm up the ATR.

## Running

```sh
uv sync
uv run jupyter lab
```

Then run `orb_qqq.ipynb` or `orb_full.ipynb`. The full universe run takes several minutes since it runs on thousands of stocks.

## Assumptions I had to make

Neither paper specifies what happens when two of the entry, stop loss and take profit levels are hit within the same candle. Answering it would require tick data. Working with OHLC data only, as [this notebook](https://concretumgroup.com/orb-strategy-backtest-in-python-using-alpaca-10-years-of-free-data/) from the same researchers also does, I resolved the edge case pessimistically.
