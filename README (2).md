# Systematic Options Strategy Backtester — Indian Weekly Expiry Markets

A rules-based backtesting engine for a systematic short-strangle strategy on
Nifty50, built around real Indian derivatives-market mechanics: weekly
Thursday expiries, lot-size-based position sizing, and delta-targeted
strike selection.

## Strategy logic

- **Entry**: every Friday, sell a strangle — short call and short put each
  selected by scanning the chain for the strike closest to a target
  **16-delta** (a standard "sell the wings" systematic rule).
- **Risk management**: mark-to-market daily; exit early if running loss
  exceeds a configurable multiple of premium collected (stop-loss).
- **Exit**: hold to Thursday expiry otherwise, settle against the terminal
  underlying price.
- Tested over 104 simulated weekly cycles (2 years) against a Nifty path
  with 13% realized vol and a 15% vol-risk-premium in the strikes sold.

## Key results (104-week backtest, ₹5L notional capital base)

| Config | Ann. Return | Sharpe | Max Drawdown | Win Rate | Stop-out Rate |
|---|---|---|---|---|---|
| No stop-loss | 11.0% | 6.62 | -6,263 | 90.4% | 0% |
| Stop-loss @ 1.5x premium | 10.6% | 6.89 | -3,895 | 87.5% | 7.7% |
| Stop-loss @ 2.5x premium | 10.5% | 6.13 | -4,230 | 89.4% | 2.9% |

The 1.5x stop-loss rule **cut max drawdown by ~38% for <0.5pp of annualized
return** — the core risk/reward tradeoff a systematic desk has to quantify
before sizing a strategy.

Benchmark: over the same simulated 2-year path the underlying fell -26.4%,
illustrating the strategy's premium-collection edge in a falling/flat market
(with the important caveat that a real strangle carries tail risk a GBM
simulation understates — noted as a limitation below).

## Stack
Python, NumPy, Pandas, SciPy

## Limitations (documented deliberately, not hidden)
- Uses GBM-simulated paths, not historical Nifty option-chain data — no fat
  tails, no vol clustering, no gap risk. Real backtests need historical
  chain data (NSE bhavcopy / vendor feeds).
- No transaction costs, slippage, or margin financing cost modeled — Sharpe
  would compress materially with realistic frictions.
- Ignores overnight/weekend gap risk, which is the dominant tail risk for
  short-strangle sellers in practice.

## Files
- `backtester.py` — full backtest engine, run with `python backtester.py`
- `strangle_trade_log.csv` — week-by-week trade log
- `equity_curve.csv` — cumulative P&L curve
