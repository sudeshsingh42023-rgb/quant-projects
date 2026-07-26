"""
Moving-Average Crossover Backtest
---------------------------------
Hypothesis being tested: "When a short-term (fast) moving average of price
crosses ABOVE a long-term (slow) moving average, the trend is turning
upward -> go long. When it crosses BELOW, the trend is turning down ->
exit / go flat."

This script:
  1. Loads daily price data (real data via yfinance, OR simulated data
     if you don't have internet / haven't installed yfinance yet).
  2. Computes two moving averages.
  3. Generates a trading signal from the crossover rule.
  4. Simulates the strategy's returns day by day (a "backtest").
  5. Compares the strategy against simply buying and holding.
  6. Reports performance metrics: total return, annualized return,
     volatility, Sharpe ratio, and max drawdown.
  7. Saves an equity-curve chart to disk.

HOW TO GET REAL MARKET DATA (run this on your own laptop, not a locked-down
sandbox):
    pip install yfinance
    then set USE_REAL_DATA = True below and pick a ticker, e.g. "^NSEI"
    (Nifty 50), "RELIANCE.NS", "AAPL", "EURUSD=X", etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIG - the few knobs you're expected to understand and be able to
# explain in an interview.
# ----------------------------------------------------------------------
USE_REAL_DATA = False      # flip to True once running locally with internet
TICKER = "^NSEI"           # only used if USE_REAL_DATA = True
FAST_WINDOW = 20           # "fast" moving average, in days
SLOW_WINDOW = 50           # "slow" moving average, in days
START_CAPITAL = 100_000    # starting portfolio value, in currency units
RISK_FREE_RATE = 0.06      # annual risk-free rate, for Sharpe ratio (India ~6%)


# ----------------------------------------------------------------------
# STEP 1: GET PRICE DATA
# ----------------------------------------------------------------------
def load_price_data():
    if USE_REAL_DATA:
        import yfinance as yf
        data = yf.download(TICKER, period="3y", interval="1d")
        prices = data["Close"].rename("price").to_frame()
        return prices

    # ---- Simulated data (used here because this sandbox has no internet
    # access to Yahoo Finance). We simulate a Geometric Brownian Motion,
    # the standard textbook model for a stock price: it has the same
    # statistical DNA as a real price series (random daily % moves,
    # compounding, occasional volatility), so the strategy logic below
    # is genuinely being tested, just not on real historical data.
    rng = np.random.default_rng(seed=42)
    n_days = 750  # ~3 years of trading days
    daily_drift = 0.0003        # small average daily upward drift
    daily_vol = 0.012           # daily volatility (~1.2%, realistic for equities)

    daily_returns = rng.normal(loc=daily_drift, scale=daily_vol, size=n_days)
    price_path = 100 * np.exp(np.cumsum(daily_returns))  # start price = 100

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    prices = pd.DataFrame({"price": price_path[: len(dates)]}, index=dates)
    return prices


# ----------------------------------------------------------------------
# STEP 2 & 3: MOVING AVERAGES + SIGNAL
# ----------------------------------------------------------------------
def build_signal(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["fast_ma"] = df["price"].rolling(FAST_WINDOW).mean()
    df["slow_ma"] = df["price"].rolling(SLOW_WINDOW).mean()

    # Signal: +1 (be long) when fast MA is above slow MA, else 0 (flat).
    # We shift by 1 day so we only trade on tomorrow's open using
    # information known at today's close -- avoids "looking into the future".
    df["signal"] = np.where(df["fast_ma"] > df["slow_ma"], 1, 0)
    df["position"] = df["signal"].shift(1).fillna(0)
    return df


# ----------------------------------------------------------------------
# STEP 4: SIMULATE STRATEGY RETURNS (THE ACTUAL "BACKTEST")
# ----------------------------------------------------------------------
def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["market_return"] = df["price"].pct_change()
    df["strategy_return"] = df["market_return"] * df["position"]

    df["buy_hold_equity"] = START_CAPITAL * (1 + df["market_return"]).cumprod()
    df["strategy_equity"] = START_CAPITAL * (1 + df["strategy_return"]).cumprod()
    return df


# ----------------------------------------------------------------------
# STEP 5: PERFORMANCE METRICS
# ----------------------------------------------------------------------
def performance_summary(df: pd.DataFrame, return_col: str, label: str) -> dict:
    returns = df[return_col].dropna()
    n_days = len(returns)
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe = (annualized_return - RISK_FREE_RATE) / annualized_vol if annualized_vol > 0 else np.nan

    equity_col = "strategy_equity" if return_col == "strategy_return" else "buy_hold_equity"
    running_max = df[equity_col].cummax()
    drawdown = (df[equity_col] - running_max) / running_max
    max_drawdown = drawdown.min()

    print(f"\n--- {label} ---")
    print(f"Total return:        {total_return:.2%}")
    print(f"Annualized return:   {annualized_return:.2%}")
    print(f"Annualized vol:      {annualized_vol:.2%}")
    print(f"Sharpe ratio:        {sharpe:.2f}")
    print(f"Max drawdown:        {max_drawdown:.2%}")

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


# ----------------------------------------------------------------------
# STEP 6: PLOT
# ----------------------------------------------------------------------
def plot_equity_curve(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["buy_hold_equity"], label="Buy & Hold", linewidth=1.5)
    plt.plot(df.index, df["strategy_equity"], label="MA Crossover Strategy", linewidth=1.5)
    plt.title(f"MA Crossover ({FAST_WINDOW}/{SLOW_WINDOW}) vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved to: {out_path}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    prices = load_price_data()
    df = build_signal(prices)
    df = run_backtest(df)

    performance_summary(df, "market_return", "Buy & Hold")
    performance_summary(df, "strategy_return", "MA Crossover Strategy")

    plot_equity_curve(df, "equity_curve.png")
