from __future__ import annotations

import os
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

# Global visual constants

DEFAULT_STYLE = "dark_background"
DEFAULT_DPI = 150
OUTPUT_DIR = "outputs"

# Shared palette for dark_background
_PALETTE = {
    "ACCENT": "#00C8FF",  # Bright blue — S_t path, BSM delta
    "GOLD": "#FFD166",    # Golden yellow — strike, held delta
    "RED": "#FF6B6B",     # Red — OTM, negative P&L, option value
    "GREEN": "#06D6A0",   # Green — ITM, positive P&L, portfolio value
    "GRAY": "#8888AA",    # Gray — secondary axes
    "BG": "#0E0E1A",      # Figure background
}


def _ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _savefig(filename: str) -> None:
    """Save the current figure to OUTPUT_DIR."""
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(
        path,
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
        facecolor=plt.gcf().get_facecolor(),
    )
    print(f"Figure saved: {path}")


def _style_ax(ax: plt.Axes, bg: str, gray: str) -> None:
    """Apply the shared dark style to an axis."""
    ax.set_facecolor(bg)
    ax.tick_params(colors=gray)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")


# 1. P&L distribution vs gamma error


def plot_pnl_distribution(pnls: np.ndarray, gamma_errors: np.ndarray, style: str = DEFAULT_STYLE) -> None:
    """
    Overlay normalized histograms of hedge P&L and gamma error,
    with vertical lines at each mean and at zero.
    """
    plt.style.use(style)
    palette = _PALETTE

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=palette["BG"])
    _style_ax(ax, palette["BG"], palette["GRAY"])

    ax.hist(
        pnls,
        bins=60,
        alpha=0.5,
        label="P&L",
        density=True,
        color=palette["ACCENT"],
    )
    ax.hist(
        gamma_errors,
        bins=60,
        alpha=0.5,
        label="Gamma Error",
        density=True,
        color=palette["GOLD"],
    )

    ax.axvline(
        np.mean(pnls),
        color=palette["ACCENT"],
        linestyle="--",
        label="P&L mean",
    )
    ax.axvline(
        np.mean(gamma_errors),
        color=palette["GOLD"],
        linestyle="--",
        label="Gamma Error mean",
    )
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)

    ax.set_title(
        "P&L Distribution Comparison",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("P&L ($)", color="white", fontsize=10)
    ax.set_ylabel("Density", color="white", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.2)

    plt.tight_layout()
    _savefig("pnl_distribution.png")
    plt.show()


# 2. GBM paths


def plot_sample_paths(paths: np.ndarray, K: float, n_display: int = 20, style: str = DEFAULT_STYLE) -> None:
    """
    Display a sample of simulated GBM paths with the strike level.
    """
    plt.style.use(style)
    palette = _PALETTE

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=palette["BG"])
    _style_ax(ax, palette["BG"], palette["GRAY"])

    n = min(n_display, len(paths))
    for i in range(n):
        ax.plot(paths[i], alpha=0.3, linewidth=0.8, color=palette["ACCENT"])

    ax.axhline(
        K,
        color=palette["GOLD"],
        linestyle="--",
        linewidth=1.5,
        label=f"Strike K={K}",
    )

    ax.set_title("Simulated GBM Paths", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time step", color="white", fontsize=10)
    ax.set_ylabel("Price ($)", color="white", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.2)

    plt.tight_layout()
    _savefig("sample_paths.png")
    plt.show()


# 3. Rebalancing frequency comparison


def plot_rebal_comparison(pnls_dict: dict[int, np.ndarray], style: str = DEFAULT_STYLE) -> None:
    """
    Plot two side-by-side charts:
      - std(hedging error) as a function of rebalancing frequency
      - E[hedging error] as a function of rebalancing frequency
    """
    plt.style.use(style)
    palette = _PALETTE

    labels = [f"every {k} step{'s' if k > 1 else ''}" for k in pnls_dict.keys()]
    stds = [np.std(v) for v in pnls_dict.values()]
    means = [np.mean(v) for v in pnls_dict.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=palette["BG"])

    # Standard deviations
    ax = axes[0]
    _style_ax(ax, palette["BG"], palette["GRAY"])
    bars = ax.bar(labels, stds, color=palette["ACCENT"], alpha=0.85, zorder=3)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9, color="white")
    ax.set_title(
        "σ(hedging error) vs rebal. frequency",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Rebalancing frequency", color="white", fontsize=10)
    ax.set_ylabel("σ(hedging error)", color="white", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Means
    ax2 = axes[1]
    _style_ax(ax2, palette["BG"], palette["GRAY"])
    bar_colors = [palette["GREEN"] if mean >= 0 else palette["RED"] for mean in means]
    bars2 = ax2.bar(labels, means, color=bar_colors, alpha=0.85, zorder=3)
    ax2.bar_label(bars2, fmt="%.4f", padding=3, fontsize=9, color="white")
    ax2.axhline(0, color="white", linewidth=0.8, alpha=0.5)
    ax2.set_title(
        "E[hedging error] vs rebal. frequency",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Rebalancing frequency", color="white", fontsize=10)
    ax2.set_ylabel("Mean P&L ($)", color="white", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _savefig("rebal_comparison.png")
    plt.show()


# 4. Console statistics


def print_stats(pnls: np.ndarray, gamma_errors: np.ndarray, label: str = "") -> None:
    """
    Print descriptive statistics for P&L and gamma error.

    Metrics: mean, standard deviation, min/max, VaR 95%/99%, CVaR 95%,
    skewness, kurtosis, % of errors > 0.5.
    """
    var95 = np.percentile(pnls, 5)
    var99 = np.percentile(pnls, 1)

    print(f"\n{'=' * 40}")
    print(f"Results — {label}")
    print(f"  Mean P&L              : {np.mean(pnls):.4f}")
    print(f"  Mean gamma error      : {np.mean(gamma_errors):.4f}")
    print(f"  Std P&L               : {np.std(pnls):.4f}")
    print(f"  Std gamma error       : {np.std(gamma_errors):.4f}")
    print(f"  P&L min / max         : {np.min(pnls):.3f} / {np.max(pnls):.3f}")
    print(f"  VaR 95%               : {var95:.4f}")
    print(f"  VaR 99%               : {var99:.4f}")
    print(f"  CVaR 95%              : {np.mean(pnls[pnls <= var95]):.4f}")
    print(f"  Skewness              : {scipy_stats.skew(pnls):.4f}")
    print(f"  Kurtosis              : {scipy_stats.kurtosis(pnls):.4f}")
    print(f"  |P&L| > 0.5  (%)      : {100 * np.mean(np.abs(pnls) > 0.5):.1f}%")


# 5. Detailed trace for one path


def _plot_price_path(ax: plt.Axes, t: np.ndarray, S: np.ndarray, K: float, rebal_idx: list[int],
                     kind: str, palette: dict[str, str]) -> None:
    """Subplot 1: underlying path with ITM/OTM zones."""
    ax.plot(t, S, color=palette["ACCENT"], lw=1.6, label="$S_t$")
    ax.axhline(K, color=palette["GOLD"], lw=1.2, ls="--", label=f"Strike $K={K}$")

    if rebal_idx:
        ax.scatter(
            t[rebal_idx],
            S[rebal_idx],
            color=palette["GOLD"],
            s=22,
            zorder=5,
            alpha=0.7,
            label="Rebalancing",
        )

    itm_cond = (S > K) if kind == "call" else (S < K)
    otm_cond = ~itm_cond
    ax.fill_between(t, S, K, where=itm_cond, alpha=0.10, color=palette["GREEN"], label="ITM")
    ax.fill_between(t, S, K, where=otm_cond, alpha=0.08, color=palette["RED"], label="OTM")

    ax.set_ylabel("Price $S_t$", color="white", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.2)


def _plot_delta_and_values(ax: plt.Axes, ax_right: plt.Axes, t: np.ndarray, delta_th: np.ndarray,
                           delta_held: np.ndarray, opt_val: np.ndarray, port_val: np.ndarray,
                           rebal_idx: list[int], palette: dict[str, str]) -> None:
    """Subplot 2: theoretical/held delta on the left, values on the right."""
    ax.step(
        t,
        delta_held,
        where="post",
        color=palette["GOLD"],
        lw=1.5,
        label="Delta held (step)",
    )
    ax.plot(
        t,
        delta_th,
        color=palette["ACCENT"],
        lw=1.0,
        ls="--",
        alpha=0.8,
        label="BSM delta (theoretical)",
    )

    if rebal_idx:
        ax.scatter(
            t[rebal_idx],
            delta_held[rebal_idx],
            color=palette["GOLD"],
            s=22,
            zorder=5,
            alpha=0.7,
        )

    ax.set_ylabel("Delta", color="white", fontsize=10)

    ax_right.plot(
        t,
        port_val,
        color=palette["GREEN"],
        lw=1.4,
        alpha=0.9,
        label="Portfolio value",
    )
    ax_right.plot(
        t,
        opt_val,
        color=palette["RED"],
        lw=1.4,
        alpha=0.9,
        ls="-.",
        label="BSM option value",
    )
    ax_right.set_ylabel("Value ($)", color="white", fontsize=10)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=8,
        framealpha=0.2,
    )


def _plot_cumulative_pnl(ax: plt.Axes, t: np.ndarray, pnl: np.ndarray, palette: dict[str, str]) -> None:
    """Subplot 3: cumulative P&L with colored zones and final annotation."""
    ax.plot(t, pnl, color=palette["GREEN"], lw=1.5, label="Cumulative hedge P&L")
    ax.axhline(0, color=palette["GRAY"], lw=0.8, ls="--")
    ax.fill_between(t, pnl, 0, where=(pnl >= 0), alpha=0.15, color=palette["GREEN"])
    ax.fill_between(t, pnl, 0, where=(pnl < 0), alpha=0.15, color=palette["RED"])

    pnl_final = pnl[-1]
    color_final = palette["GREEN"] if pnl_final >= 0 else palette["RED"]
    ax.annotate(
        f"Final P&L: {pnl_final:+.4f}",
        xy=(t[-1], pnl_final),
        xytext=(-60, 15),
        textcoords="offset points",
        color=color_final,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=color_final, lw=1.2),
    )

    ax.set_xlabel("Time $t$ (years)", color="white", fontsize=10)
    ax.set_ylabel("Cumulative P&L ($)", color="white", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.2)


def plot_hedge_trace(trace: dict[str, Any], K: float, kind: str = "call",
                     figsize: tuple[int, int] = (13, 10), style: str = DEFAULT_STYLE) -> None:
    """
    Plot a 3-panel figure for one hedge path:
      1. S_t with strike, ITM/OTM zones, rebalancing markers
      2. Theoretical vs held delta + portfolio vs option value (right axis)
      3. Cumulative hedge P&L
    """
    plt.style.use(style)
    palette = _PALETTE

    fig = plt.figure(figsize=figsize, facecolor=palette["BG"])
    gs = gridspec.GridSpec(
        3,
        1,
        hspace=0.55,
        left=0.09,
        right=0.97,
        top=0.93,
        bottom=0.07,
        height_ratios=[1.2, 1.2, 1.0],
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax2b = ax2.twinx()
    ax3 = fig.add_subplot(gs[2])

    for ax in (ax1, ax2, ax3):
        _style_ax(ax, palette["BG"], palette["GRAY"])
    _style_ax(ax2b, palette["BG"], palette["GRAY"])

    t = trace["times"]
    S = trace["S"]
    rebal_idx = trace["rebal_idx"]

    _plot_price_path(ax1, t, S, K, rebal_idx, kind, palette)
    ax1.set_title(
        f"Delta hedge trace — {kind.capitalize()}",
        color="white",
        fontsize=13,
        pad=10,
        fontweight="bold",
    )

    _plot_delta_and_values(
        ax2,
        ax2b,
        t,
        trace["delta_th"],
        trace["delta_held"],
        trace["option_val"],
        trace["portfolio_val"],
        rebal_idx,
        palette,
    )
    ax2.tick_params(colors=palette["GRAY"])
    ax2b.tick_params(colors=palette["GRAY"])

    _plot_cumulative_pnl(ax3, t, trace["pnl"], palette)

    _savefig("hedge_trace.png")
    plt.show()


# 6. Gamma error vs P&L scatter plot


def plot_gamma_vs_pnl(pnls: np.ndarray, gamma_errors: np.ndarray, figsize: tuple[int, int] = (8, 8),
                      style: str = DEFAULT_STYLE) -> None:
    """
    Plot gamma_error (x) vs hedge P&L (y), colored by KDE density.

    If the model is correct, points should align with the diagonal y = x.
    Also display the OLS regression line and associated statistics
    (R², correlation, slope).
    """
    plt.style.use(style)
    palette = _PALETTE

    # Linear regression
    slope, intercept, r_value, _, _ = scipy_stats.linregress(gamma_errors, pnls)
    r2 = r_value**2
    corr = float(np.corrcoef(gamma_errors, pnls)[0, 1])

    # KDE-based density coloring
    xy = np.vstack([gamma_errors, pnls])
    kde = scipy_stats.gaussian_kde(xy)
    density = kde(xy)
    order = density.argsort()

    fig, ax = plt.subplots(figsize=figsize, facecolor=palette["BG"])
    _style_ax(ax, palette["BG"], palette["GRAY"])

    sc = ax.scatter(
        gamma_errors[order],
        pnls[order],
        c=density[order],
        cmap="plasma",
        s=14,
        alpha=0.7,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Density", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=palette["GRAY"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=palette["GRAY"], fontsize=8)

    # Shared limits for square axes
    all_vals = np.concatenate([gamma_errors, pnls])
    lo, hi = all_vals.min(), all_vals.max()
    pad = (hi - lo) * 0.08
    lim = (lo - pad, hi + pad)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    x_line = np.array(lim)

    # Theoretical line y = x
    ax.plot(
        x_line,
        x_line,
        color=palette["GOLD"],
        lw=1.6,
        ls="--",
        label="$y = x$  (theoretical)",
    )

    # OLS line
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color=palette["GREEN"],
        lw=1.4,
        ls="-",
        label=f"OLS  $y = {slope:.3f}x {intercept:+.4f}$",
    )

    # Zero axes
    ax.axhline(0, color=palette["GRAY"], lw=0.6, alpha=0.4)
    ax.axvline(0, color=palette["GRAY"], lw=0.6, alpha=0.4)

    # Statistical annotation
    stats_text = (
        f"$R^2$         = {r2:.4f}\n"
        f"Correlation = {corr:.4f}\n"
        f"OLS slope   = {slope:.4f}\n"
        f"$n$           = {len(pnls):,}"
    )
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#1A1A2E",
            edgecolor="#333355",
            alpha=0.85,
        ),
        family="monospace",
    )

    ax.set_xlabel("Gamma error (analytical approximation)", color="white", fontsize=11)
    ax.set_ylabel("Hedge P&L (simulation)", color="white", fontsize=11)
    ax.set_title(
        "Gamma error vs Hedge P&L\n"
        r"Alignment on $y=x$ $\Rightarrow$ $\Gamma$ approximation is accurate",
        color="white",
        fontsize=12,
        pad=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    _savefig("scatter_gamma_pnl.png")
    plt.show()