"""
Options Pricing & Greeks-Based Delta-Hedging Engine
----------------------------------------------------
Simulates a Nifty50-style underlying, prices European options via
Black-Scholes-Merton, backs out implied volatility from "market" quotes,
builds an IV smile/surface, and runs a discrete delta-hedging backtest
to show variance reduction from a Greeks-based hedge vs an unhedged
short-option position.

Libraries: NumPy, Pandas, SciPy
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# 1. Black-Scholes-Merton pricing + full Greeks
# ----------------------------------------------------------------------

def bsm_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    if T <= 0:
        intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price


def bsm_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

    return {"delta": delta, "gamma": gamma, "vega": vega / 100,
            "theta": theta / 365, "rho": rho / 100}


# ----------------------------------------------------------------------
# 2. Implied volatility solver (Brent's method)
# ----------------------------------------------------------------------

def implied_vol(market_price, S, K, T, r, q=0.0, option_type="call"):
    def obj(sigma):
        return bsm_price(S, K, T, r, sigma, q, option_type) - market_price
    try:
        return brentq(obj, 1e-4, 5.0, xtol=1e-6, maxiter=200)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------
# 3. Synthetic Nifty50 option chain -> volatility smile/surface
# ----------------------------------------------------------------------

def build_synthetic_chain(spot=24500, r=0.065, q=0.012, seed=42):
    rng = np.random.default_rng(seed)
    expiries = np.array([7, 14, 30, 60, 90]) / 365.0          # weekly/monthly Nifty expiries
    strikes = np.arange(spot * 0.90, spot * 1.10, spot * 0.01)  # +/-10% around spot, 1% steps

    rows = []
    for T in expiries:
        atm_vol = 0.13 + 0.02 * np.sqrt(T)          # term structure: longer T -> slightly higher vol
        for K in strikes:
            moneyness = np.log(K / spot)
            skew = -0.35 * moneyness + 1.1 * moneyness ** 2   # negative skew, typical of index options
            true_sigma = max(0.06, atm_vol + skew)
            noise = rng.normal(0, 0.003)
            quoted_sigma = max(0.05, true_sigma + noise)

            opt_type = "put" if K < spot else "call"
            mkt_price = bsm_price(spot, K, T, r, quoted_sigma, q, opt_type)
            mkt_price = round(max(mkt_price, 0.05), 2)  # tick-size floor

            iv = implied_vol(mkt_price, spot, K, T, r, q, opt_type)
            rows.append({"T_days": round(T * 365), "K": round(K), "type": opt_type,
                         "mkt_price": mkt_price, "implied_vol": iv})
    return pd.DataFrame(rows), spot, r, q


def summarize_vol_surface(df):
    surface = df.pivot_table(index="K", columns="T_days", values="implied_vol")
    print("Implied Volatility Surface (rows=Strike, cols=Days to Expiry):")
    print(surface.round(4).to_string())
    return surface


# ----------------------------------------------------------------------
# 4. Discrete delta-hedging backtest: hedged vs unhedged short straddle
# ----------------------------------------------------------------------

def simulate_gbm_path(S0, mu, sigma, T, steps, seed=1):
    rng = np.random.default_rng(seed)
    dt = T / steps
    z = rng.normal(size=steps)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    path = S0 * np.exp(np.cumsum(log_returns))
    return np.insert(path, 0, S0)


def delta_hedge_backtest(S0=24500, K=24500, T=30/365, r=0.065, q=0.012,
                          true_sigma=0.15, hedge_sigma=0.15, steps=30,
                          n_paths=2000, rehedge_every=1, lot_size=25):
    """Short 1 ATM straddle (short call + short put), lot_size units each.
    Compare P&L variance: unhedged vs delta-hedged (re-hedged every `rehedge_every` steps)."""
    dt = T / steps
    unhedged_pnls, hedged_pnls = [], []

    for p in range(n_paths):
        path = simulate_gbm_path(S0, mu=r - q, sigma=true_sigma, T=T, steps=steps, seed=1000 + p)
        t_remain = np.linspace(T, 0, steps + 1)

        call0 = bsm_price(S0, K, T, r, hedge_sigma, q, "call")
        put0 = bsm_price(S0, K, T, r, hedge_sigma, q, "put")
        premium_received = (call0 + put0) * lot_size

        # Unhedged: just settle at expiry
        ST = path[-1]
        payoff = (max(ST - K, 0) + max(K - ST, 0)) * lot_size
        unhedged_pnls.append(premium_received - payoff)

        # Delta-hedged: rebalance underlying position each `rehedge_every` steps
        cash = premium_received
        shares_held = 0.0
        for i in range(steps + 1):
            S_t = path[i]
            T_t = max(t_remain[i], 1e-6)
            if i < steps:
                g_call = bsm_greeks(S_t, K, T_t, r, hedge_sigma, q, "call")
                g_put = bsm_greeks(S_t, K, T_t, r, hedge_sigma, q, "put")
                position_delta = -(g_call["delta"] + g_put["delta"]) * lot_size  # delta of short straddle
                required_hedge_shares = -position_delta  # shares needed to zero out net delta
                if i % rehedge_every == 0:
                    trade = required_hedge_shares - shares_held
                    cash -= trade * S_t
                    shares_held = required_hedge_shares
            else:
                cash += shares_held * S_t
                shares_held = 0.0
        hedged_pnls.append(cash - payoff)

    unhedged_pnls = np.array(unhedged_pnls)
    hedged_pnls = np.array(hedged_pnls)
    return unhedged_pnls, hedged_pnls


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: Sample BSM price + Greeks (ATM Nifty call, 30D, vol=15%)")
    print("=" * 70)
    S, K, T, r, q, sigma = 24500, 24500, 30/365, 0.065, 0.012, 0.15
    price = bsm_price(S, K, T, r, sigma, q, "call")
    greeks = bsm_greeks(S, K, T, r, sigma, q, "call")
    print(f"Price: {price:.2f} | Greeks: { {k: round(v,4) for k,v in greeks.items()} }")

    print("\n" + "=" * 70)
    print("STEP 2: Build synthetic option chain, solve implied vols, build surface")
    print("=" * 70)
    chain_df, spot, r, q = build_synthetic_chain()
    chain_df.to_csv("synthetic_nifty_chain.csv", index=False)
    surface = summarize_vol_surface(chain_df)
    surface.to_csv("iv_surface.csv")

    print("\n" + "=" * 70)
    print("STEP 3: Delta-hedging backtest — short ATM straddle, 2000 simulated paths")
    print("=" * 70)
    for rehedge in [1, 5, 30]:  # daily, weekly-ish, never (buy&hold hedge only at t0)
        unhedged, hedged = delta_hedge_backtest(rehedge_every=rehedge)
        label = {1: "Daily re-hedge", 5: "Every-5-day re-hedge", 30: "Static (t=0 only) hedge"}[rehedge]
        print(f"\n[{label}]")
        print(f"  Unhedged  -> mean P&L: {unhedged.mean():8.1f}  std: {unhedged.std():8.1f}")
        print(f"  Hedged    -> mean P&L: {hedged.mean():8.1f}  std: {hedged.std():8.1f}")
        var_reduction = 1 - (hedged.std() ** 2) / (unhedged.std() ** 2)
        print(f"  Variance reduction from hedging: {var_reduction*100:.1f}%")
