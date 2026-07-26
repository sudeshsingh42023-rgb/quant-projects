"""
Black-Scholes Option Pricer & Greeks Calculator
------------------------------------------------
What this answers: "Given a stock/underlying price, a strike price, time
to expiry, volatility, and interest rate -- what is a European call/put
option theoretically worth, and how sensitive is that value to each input?"

The five sensitivities computed here ("the Greeks") are the standard
vocabulary of any options desk:
  Delta - how much the option price moves per $1 move in the underlying
  Gamma - how much Delta itself moves per $1 move in the underlying
  Vega  - how much the option price moves per 1% change in volatility
  Theta - how much value the option loses per day, all else equal
          ("time decay")
  Rho   - how much the option price moves per 1% change in interest rates

This script:
  1. Prices a European call and put using the Black-Scholes formula.
  2. Computes all five Greeks analytically.
  3. Sweeps the underlying price across a range to show how option value
     and Delta change as the market moves -- and plots both.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIG - a sample option contract. Change these and re-run to build
# intuition for how each input drives price.
# ----------------------------------------------------------------------
S = 100      # current price of the underlying (e.g. a stock or index)
K = 100      # strike price
T = 0.5      # time to expiry, in years (0.5 = 6 months)
r = 0.06     # annual risk-free interest rate (6%, typical for India)
sigma = 0.20 # annualized volatility of the underlying (20%)


# ----------------------------------------------------------------------
# CORE FORMULA
# ----------------------------------------------------------------------
def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call"):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put, via put-call parity relationship built into the formula
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365  # convert to per-day
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # per 1% rate move
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100  # per 1% vol move

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ----------------------------------------------------------------------
# SENSITIVITY SWEEP - how does price/delta change as the underlying moves?
# ----------------------------------------------------------------------
def sensitivity_sweep(K, T, r, sigma, option_type="call", spot_range=(70, 130)):
    spots = np.linspace(spot_range[0], spot_range[1], 200)
    prices = [bs_price(s, K, T, r, sigma, option_type) for s in spots]
    deltas = [bs_greeks(s, K, T, r, sigma, option_type)["delta"] for s in spots]
    return spots, prices, deltas


def plot_sensitivity(spots, prices, deltas, option_type):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(spots, prices, color="tab:blue")
    ax1.axvline(K, color="grey", linestyle="--", linewidth=1, label=f"Strike = {K}")
    ax1.set_title(f"{option_type.capitalize()} Price vs Underlying Price")
    ax1.set_xlabel("Underlying Price")
    ax1.set_ylabel("Option Price")
    ax1.legend()

    ax2.plot(spots, deltas, color="tab:orange")
    ax2.axvline(K, color="grey", linestyle="--", linewidth=1, label=f"Strike = {K}")
    ax2.set_title(f"{option_type.capitalize()} Delta vs Underlying Price")
    ax2.set_xlabel("Underlying Price")
    ax2.set_ylabel("Delta")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("option_sensitivity.png", dpi=150)
    print("Chart saved to: option_sensitivity.png")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    for option_type in ["call", "put"]:
        price = bs_price(S, K, T, r, sigma, option_type)
        greeks = bs_greeks(S, K, T, r, sigma, option_type)

        print(f"\n--- {option_type.upper()} (S={S}, K={K}, T={T}y, r={r:.0%}, sigma={sigma:.0%}) ---")
        print(f"Price: {price:.2f}")
        for name, value in greeks.items():
            print(f"{name.capitalize():>6}: {value:.4f}")

    spots, prices, deltas = sensitivity_sweep(K, T, r, sigma, option_type="call")
    plot_sensitivity(spots, prices, deltas, "call")
