# Options Pricing Engine & Greeks-Based Delta-Hedging Backtest

A from-scratch options pricing and risk-management engine built around Indian
index options (Nifty50-style), covering the full pipeline a quant desk needs:
price → Greeks → implied vol → vol surface → hedge.

## What it does

1. **Black-Scholes-Merton pricing engine** — European call/put pricing with
   dividend yield, plus full closed-form Greeks (delta, gamma, vega, theta, rho).
2. **Implied volatility solver** — Brent's method root-finder that inverts
   market option prices back to implied vol.
3. **Synthetic option chain + IV surface** — builds a 21-strike x 5-expiry
   Nifty-style chain (7/14/30/60/90 DTE) with a realistic negative skew and
   term structure, then solves and visualizes the full IV surface.
4. **Delta-hedging backtest** — simulates 2,000 GBM price paths, sells an
   ATM straddle on each, and compares unhedged P&L variance against a
   discretely-rebalanced delta-hedged position.

## Key result

Daily delta-rehedging cut the standard deviation of short-straddle P&L from
**15,850 to 3,520 — a 95.1% variance reduction** — versus a 4.4% reduction
from a static (set-and-forget) hedge, quantifying why rehedge frequency
matters for a Greeks-based hedging desk.

## Stack
Python, NumPy, Pandas, SciPy (`scipy.stats.norm`, `scipy.optimize.brentq`)

## Files
- `pricing_engine.py` — full engine + backtest, run directly with `python pricing_engine.py`
- `synthetic_nifty_chain.csv` — generated option chain with solved IVs
- `iv_surface.csv` — pivoted implied volatility surface (strike x expiry)
