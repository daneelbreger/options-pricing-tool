"""
Options Pricing Tool — Black-Scholes Model
Features: Call/Put pricing, Greeks (risk metrics), and charts
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# ─────────────────────────────────────────────
#  Core Black-Scholes Functions
# ─────────────────────────────────────────────

def d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 for Black-Scholes."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Price a European option using Black-Scholes.
    S     : Current stock price
    K     : Strike price
    T     : Time to expiry (in years)
    r     : Risk-free rate (e.g. 0.05 = 5%)
    sigma : Volatility (e.g. 0.20 = 20%)
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute the five main Greeks.
    Returns a dict with Delta, Gamma, Theta, Vega, Rho.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    sign = 1 if option_type == "call" else -1

    delta = sign * norm.cdf(sign * d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100          # per 1% vol move
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - sign * r * K * np.exp(-r * T) * norm.cdf(sign * d2)) / 365  # per day
    rho   = sign * K * T * np.exp(-r * T) * norm.cdf(sign * d2) / 100      # per 1% rate move

    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────

def plot_dashboard(S, K, T, r, sigma):
    """
    Render a 6-panel dashboard:
      1. Option Price vs Stock Price (Call & Put)
      2. Payoff at Expiry
      3. Delta vs Stock Price
      4. Gamma vs Stock Price
      5. Theta vs Time to Expiry
      6. Vega vs Volatility
    """
    stock_range  = np.linspace(S * 0.5, S * 1.5, 300)
    time_range   = np.linspace(0.01, 1.0, 300)
    sigma_range  = np.linspace(0.05, 0.80, 300)

    call_prices = [black_scholes_price(s, K, T, r, sigma, "call") for s in stock_range]
    put_prices  = [black_scholes_price(s, K, T, r, sigma, "put")  for s in stock_range]

    call_payoff = np.maximum(stock_range - K, 0)
    put_payoff  = np.maximum(K - stock_range, 0)

    call_delta  = [greeks(s, K, T, r, sigma, "call")["Delta"] for s in stock_range]
    put_delta   = [greeks(s, K, T, r, sigma, "put") ["Delta"] for s in stock_range]
    call_gamma  = [greeks(s, K, T, r, sigma, "call")["Gamma"] for s in stock_range]

    call_theta  = [greeks(S, K, t, r, sigma, "call")["Theta"] for t in time_range]
    put_theta   = [greeks(S, K, t, r, sigma, "put") ["Theta"] for t in time_range]

    call_vega   = [greeks(S, K, T, r, s, "call")["Vega"] for s in sigma_range]

    # ── Current values ───────────────────────
    cp = black_scholes_price(S, K, T, r, sigma, "call")
    pp = black_scholes_price(S, K, T, r, sigma, "put")
    cg = greeks(S, K, T, r, sigma, "call")
    pg = greeks(S, K, T, r, sigma, "put")

    # ── Layout ───────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor="#0f1117")
    fig.suptitle(
        f"Options Pricing Dashboard  |  S=${S}  K=${K}  T={T:.2f}yr  σ={sigma*100:.0f}%  r={r*100:.1f}%",
        fontsize=14, color="white", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                           left=0.07, right=0.97, top=0.92, bottom=0.06)

    DARK   = "#0f1117"
    PANEL  = "#1a1d27"
    CALL   = "#00d4aa"
    PUT    = "#ff6b6b"
    GAMMA  = "#f7c59f"
    ACCENT = "#7b9cff"
    TEXT   = "#ccccdd"

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=9, pad=6)
        ax.tick_params(colors=TEXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333348")
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.grid(color="#22253a", linestyle="--", linewidth=0.5)

    # 1 — Price vs Stock (top-left, spans 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(stock_range, call_prices, color=CALL, lw=2, label=f"Call  ${cp:.2f}")
    ax1.plot(stock_range, put_prices,  color=PUT,  lw=2, label=f"Put   ${pp:.2f}")
    ax1.axvline(S, color="white", lw=1, ls=":", alpha=0.6, label=f"Spot ${S}")
    ax1.axvline(K, color=GAMMA,   lw=1, ls=":", alpha=0.6, label=f"Strike ${K}")
    ax1.set_xlabel("Stock Price ($)")
    ax1.set_ylabel("Option Price ($)")
    ax1.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, loc="upper left")
    style_ax(ax1, "Option Price vs Stock Price")

    # 2 — Payoff at Expiry (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(stock_range, call_payoff, color=CALL, lw=2, label="Call Payoff")
    ax2.plot(stock_range, put_payoff,  color=PUT,  lw=2, label="Put Payoff")
    ax2.fill_between(stock_range, call_payoff, alpha=0.12, color=CALL)
    ax2.fill_between(stock_range, put_payoff,  alpha=0.12, color=PUT)
    ax2.axvline(K, color=GAMMA, lw=1, ls=":", alpha=0.7)
    ax2.set_xlabel("Stock Price at Expiry ($)")
    ax2.set_ylabel("Payoff ($)")
    ax2.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
    style_ax(ax2, "Payoff Diagram at Expiry")

    # 3 — Delta (mid-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(stock_range, call_delta, color=CALL, lw=2, label="Call Δ")
    ax3.plot(stock_range, put_delta,  color=PUT,  lw=2, label="Put Δ")
    ax3.axvline(S, color="white", lw=1, ls=":", alpha=0.5)
    ax3.axhline(0, color="#444", lw=0.8)
    ax3.set_xlabel("Stock Price ($)")
    ax3.set_ylabel("Delta")
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
    style_ax(ax3, "Delta  (Price Sensitivity)")

    # 4 — Gamma (mid-centre)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(stock_range, call_gamma, color=GAMMA, lw=2)
    ax4.axvline(S, color="white", lw=1, ls=":", alpha=0.5)
    ax4.set_xlabel("Stock Price ($)")
    ax4.set_ylabel("Gamma")
    style_ax(ax4, "Gamma  (Delta Sensitivity)")

    # 5 — Theta (mid-right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(time_range, call_theta, color=CALL, lw=2, label="Call Θ")
    ax5.plot(time_range, put_theta,  color=PUT,  lw=2, label="Put Θ")
    ax5.axvline(T, color="white", lw=1, ls=":", alpha=0.5)
    ax5.set_xlabel("Time to Expiry (years)")
    ax5.set_ylabel("Theta ($/day)")
    ax5.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
    style_ax(ax5, "Theta  (Time Decay per Day)")

    # 6 — Vega (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(sigma_range * 100, call_vega, color=ACCENT, lw=2)
    ax6.axvline(sigma * 100, color="white", lw=1, ls=":", alpha=0.5)
    ax6.set_xlabel("Volatility (%)")
    ax6.set_ylabel("Vega ($ per 1% vol)")
    style_ax(ax6, "Vega  (Volatility Sensitivity)")

    # 7 — Greeks summary table (bottom, spans 2 cols)
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis("off")
    ax7.set_facecolor(PANEL)

    headers = ["Greek", "Call", "Put", "Interpretation"]
    rows = [
        ["Delta  (Δ)", f"{cg['Delta']:+.4f}", f"{pg['Delta']:+.4f}", "$ change per $1 stock move"],
        ["Gamma (Γ)", f"{cg['Gamma']:.5f}",  f"{pg['Gamma']:.5f}",  "Delta change per $1 move"],
        ["Theta  (Θ)", f"{cg['Theta']:+.4f}", f"{pg['Theta']:+.4f}", "$ lost per calendar day"],
        ["Vega   (ν)", f"{cg['Vega']:+.4f}",  f"{pg['Vega']:+.4f}",  "$ change per 1% vol move"],
        ["Rho    (ρ)", f"{cg['Rho']:+.4f}",   f"{pg['Rho']:+.4f}",   "$ change per 1% rate move"],
    ]

    col_widths = [0.14, 0.10, 0.10, 0.44]
    col_x      = [0.01, 0.20, 0.32, 0.44]
    row_h      = 0.155
    header_y   = 0.88

    for i, (hdr, cx) in enumerate(zip(headers, col_x)):
        ax7.text(cx, header_y, hdr, transform=ax7.transAxes,
                 fontsize=8, color=ACCENT, fontweight="bold")

    for r_idx, row in enumerate(rows):
        y = header_y - (r_idx + 1) * row_h
        bg_color = "#1e2133" if r_idx % 2 == 0 else PANEL
        ax7.add_patch(plt.Rectangle((0, y - 0.04), 1.0, row_h,
                                    transform=ax7.transAxes,
                                    color=bg_color, zorder=0))
        for c_idx, (val, cx) in enumerate(zip(row, col_x)):
            color = CALL if c_idx == 1 else (PUT if c_idx == 2 else TEXT)
            ax7.text(cx, y, val, transform=ax7.transAxes,
                     fontsize=7.5, color=color)

    ax7.set_title("Greeks Summary Table", color=TEXT, fontsize=9, pad=6)
    for spine in ax7.spines.values():
        spine.set_edgecolor("#333348")

    fig.patch.set_facecolor(DARK)
    plt.savefig("/mnt/user-data/outputs/options_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor=DARK)
    print("Dashboard saved → options_dashboard.png")
    plt.show()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Parameters (edit these) ──────────────
    S     = 150.0   # Current stock price ($)
    K     = 155.0   # Strike price ($)
    T     = 0.25    # Time to expiry (0.25 = 3 months)
    r     = 0.05    # Risk-free rate  (0.05 = 5%)
    sigma = 0.25    # Volatility      (0.25 = 25%)

    # ── Print summary ────────────────────────
    print("=" * 54)
    print(f"  BLACK-SCHOLES OPTIONS PRICING TOOL")
    print("=" * 54)
    print(f"  Stock Price : ${S}")
    print(f"  Strike      : ${K}")
    print(f"  Expiry      : {T*12:.1f} months  ({T:.4f} yr)")
    print(f"  Risk-Free r : {r*100:.1f}%")
    print(f"  Volatility  : {sigma*100:.0f}%")
    print("-" * 54)

    for opt in ("call", "put"):
        price = black_scholes_price(S, K, T, r, sigma, opt)
        g     = greeks(S, K, T, r, sigma, opt)
        print(f"\n  {opt.upper()}  →  ${price:.4f}")
        for name, val in g.items():
            print(f"    {name:<8} {val:+.5f}")

    print("=" * 54)
    print("\n  Generating dashboard…")
    plot_dashboard(S, K, T, r, sigma)
