"""
Systematic Options Strategy Backtester — Indian Derivatives (Nifty Weekly Expiry)
-----------------------------------------------------------------------------------
Backtests a rules-based short-strangle strategy sold every Thursday expiry cycle
on a simulated Nifty50 path, with realistic Indian-market mechanics:
  - Weekly Thursday expiry cycle
  - Lot size 25 (Nifty), margin-based sizing
  - Entry: sell OTM strangle (call + put) at fixed delta band on Friday (day after prior expiry)
  - Exit: square off at expiry, OR stop-loss if strategy MTM loss exceeds threshold
Computes strategy equity curve, Sharpe, max drawdown, win rate vs a simple
buy-and-hold-the-index benchmark.

Libraries: NumPy, Pandas, SciPy
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


def bsm_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    if T <= 1e-8:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bsm_delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    if T <= 1e-8:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)


def find_strike_for_delta(S, T, r, sigma, q, target_delta, option_type, strike_step=50):
    """Scan strikes in strike_step increments to find the one closest to target |delta|."""
    strikes = np.arange(S * 0.85, S * 1.15, strike_step)
    deltas = np.array([abs(bsm_delta(S, K, T, r, sigma, q, option_type)) for K in strikes])
    idx = np.argmin(np.abs(deltas - target_delta))
    return strikes[idx]


def simulate_nifty_path(S0, mu, sigma, n_days, seed=7):
    rng = np.random.default_rng(seed)
    dt = 1 / 365
    z = rng.normal(size=n_days)
    log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    return S0 * np.exp(np.cumsum(np.insert(log_ret, 0, 0)))


def run_backtest(n_weeks=104, S0=24500, r=0.065, q=0.012, mu=0.10,
                  realized_vol=0.13, iv_sell=0.15, target_delta=0.16,
                  lot_size=25, n_lots=1, sl_pct=1.5, seed=7):
    """
    n_weeks weekly cycles of: sell 16-delta strangle Friday, hold to Thursday expiry,
    stop out if running loss > sl_pct * premium collected.
    """
    total_days = n_weeks * 7
    path = simulate_nifty_path(S0, mu, realized_vol, total_days, seed=seed)

    trade_log = []
    equity = 0.0
    equity_curve = [0.0]

    for week in range(n_weeks):
        start_idx = week * 7
        S_entry = path[start_idx]
        T_entry = 5 / 365  # Friday to Thursday ~ 5 trading days

        call_K = find_strike_for_delta(S_entry, T_entry, r, iv_sell, q, target_delta, "call")
        put_K = find_strike_for_delta(S_entry, T_entry, r, iv_sell, q, target_delta, "put")

        call_premium = bsm_price(S_entry, call_K, T_entry, r, iv_sell, q, "call")
        put_premium = bsm_price(S_entry, put_K, T_entry, r, iv_sell, q, "put")
        premium_collected = (call_premium + put_premium) * lot_size * n_lots
        stop_loss_level = premium_collected * sl_pct

        stopped_out = False
        running_mtm_loss = 0.0
        for d in range(1, 6):
            if start_idx + d >= len(path):
                break
            S_t = path[start_idx + d]
            T_t = max((5 - d) / 365, 1e-6)
            call_val = bsm_price(S_t, call_K, T_t, r, iv_sell, q, "call")
            put_val = bsm_price(S_t, put_K, T_t, r, iv_sell, q, "put")
            current_liability = (call_val + put_val) * lot_size * n_lots
            running_mtm_loss = current_liability - premium_collected
            if running_mtm_loss > stop_loss_level:
                stopped_out = True
                exit_day = d
                exit_pnl = -stop_loss_level
                break

        if not stopped_out:
            S_exit = path[min(start_idx + 5, len(path) - 1)]
            call_payoff = max(S_exit - call_K, 0) * lot_size * n_lots
            put_payoff = max(put_K - S_exit, 0) * lot_size * n_lots
            exit_pnl = premium_collected - (call_payoff + put_payoff)
            exit_day = 5

        equity += exit_pnl
        equity_curve.append(equity)
        trade_log.append({
            "week": week, "S_entry": round(S_entry), "call_K": call_K, "put_K": put_K,
            "premium_collected": round(premium_collected, 1),
            "pnl": round(exit_pnl, 1), "stopped_out": stopped_out, "exit_day": exit_day
        })

    log_df = pd.DataFrame(trade_log)
    equity_curve = np.array(equity_curve)
    return log_df, equity_curve, path


def performance_stats(log_df, equity_curve, capital_base=500000):
    weekly_returns = log_df["pnl"].values / capital_base
    sharpe = (weekly_returns.mean() / weekly_returns.std()) * np.sqrt(52) if weekly_returns.std() > 0 else np.nan

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max
    max_dd = drawdown.min()

    win_rate = (log_df["pnl"] > 0).mean()
    stopout_rate = log_df["stopped_out"].mean()
    total_pnl = log_df["pnl"].sum()
    avg_win = log_df.loc[log_df["pnl"] > 0, "pnl"].mean()
    avg_loss = log_df.loc[log_df["pnl"] <= 0, "pnl"].mean()

    return {
        "total_pnl": total_pnl,
        "annualized_return_pct": (total_pnl / capital_base) * (52 / len(log_df)) * 100,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate_pct": win_rate * 100,
        "stopout_rate_pct": stopout_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Backtest: Weekly 16-delta short strangle on simulated Nifty, 104 weeks (2y)")
    print("=" * 70)

    for sl in [None, 1.5, 2.5]:
        sl_pct = sl if sl else 1e9
        log_df, equity_curve, path = run_backtest(sl_pct=sl_pct)
        stats = performance_stats(log_df, equity_curve)
        label = "No stop-loss" if sl is None else f"Stop-loss @ {sl}x premium"
        print(f"\n[{label}]")
        for k, v in stats.items():
            print(f"  {k:24s}: {v:10.2f}")

    print("\n" + "=" * 70)
    print("Benchmark: Buy-and-hold Nifty over same 2-year path")
    print("=" * 70)
    bh_return = (path[-1] / path[0] - 1) * 100
    print(f"  Buy-and-hold return: {bh_return:.2f}%  (Nifty {path[0]:.0f} -> {path[-1]:.0f})")

    log_df.to_csv("strangle_trade_log.csv", index=False)
    pd.Series(equity_curve).to_csv("equity_curve.csv")
