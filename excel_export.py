import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


#  Color palette
BG_DARK  = "#0A0F1E"
BG_MID   = "#0D1B2A"
BG_PANEL = "#112233"
ORANGE   = "#FF6B00"
TEAL     = "#00C8C8"
GOLD     = "#FFD700"
WHITE    = "#FFFFFF"
GRAY_L   = "#C8CDD4"
GRAY_M   = "#8A9BB0"
GREEN    = "#00D084"
RED      = "#FF4D4D"
PURPLE   = "#A855F7"
CYAN     = "#22D3EE"

SERIES_COLORS = [ORANGE, TEAL, GOLD, PURPLE, CYAN]


#  Axis styling

def _style_ax(ax, title="", ylabel="", xlabel="", grid_axis="y"):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=GRAY_L, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BG_MID)
    ax.yaxis.label.set_color(GRAY_M)
    ax.xaxis.label.set_color(GRAY_M)
    if title:
        ax.set_title(title, color=ORANGE, fontsize=10, fontweight="bold", pad=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=GRAY_M)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=GRAY_M)
    if grid_axis:
        ax.grid(axis=grid_axis, color=BG_MID, linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, fmt="{:.1f}%", scale=100):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        offset = abs(h) * 0.03 + 0.002
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + (offset if h >= 0 else -offset * 4),
            fmt.format(h * scale),
            ha="center", va="bottom",
            color=WHITE, fontsize=7, fontweight="bold")


#  Solver precision

_SLSQP_OPTS          = {"ftol": 1e-12, "eps": 1e-8, "maxiter": 1_000}
_N_FRONTIER_POINTS   = 600
_N_RANDOM_PORTFOLIOS = 4_000


#  Efficient Frontier computation

def _compute_efficient_frontier(bl_returns, cov_matrix, risk_free,
                                 max_weights=None,
                                 n_points=_N_FRONTIER_POINTS,
                                 n_random=_N_RANDOM_PORTFOLIOS):
    """
    Builds Markowitz efficient frontier.

    """
    n  = len(bl_returns)
    cm = np.array(cov_matrix)
    br = np.array(bl_returns)

    if max_weights is not None:
        bounds = [(0.0, float(mw)) for mw in max_weights]
    else:
        bounds = [(0.0, 1.0)] * n

    upper = np.array([b[1] for b in bounds])

    def port_vol(w):
        return float(np.sqrt(w @ cm @ w))

    def port_ret(w):
        return float(w @ br)

    base_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0 = np.full(n, 1.0 / n)

    #  1. Random portfolio cloud 
    rng = np.random.default_rng(seed=0)
    rand_vols, rand_rets, rand_sharpes = [], [], []

    for _ in range(n_random):
        raw = rng.dirichlet(np.ones(n))
        raw = np.clip(raw, 0.0, upper)
        s   = raw.sum()
        if s < 1e-9:
            continue
        w  = raw / s
        rv = port_vol(w)
        rr = port_ret(w)
        rand_vols.append(rv)
        rand_rets.append(rr)
        rand_sharpes.append((rr - risk_free) / rv if rv > 1e-12 else 0.0)

    rand_vols    = np.array(rand_vols)
    rand_rets    = np.array(rand_rets)
    rand_sharpes = np.array(rand_sharpes)

    #  2. Global Minimum Variance 
    res_gmv = minimize(port_vol, w0, method="SLSQP",
                       bounds=bounds, constraints=base_constraints,
                       options=_SLSQP_OPTS)
    gmv_w   = res_gmv.x if res_gmv.success else w0
    gmv_vol = port_vol(gmv_w)
    gmv_ret = port_ret(gmv_w)

    #  3. Max Sharpe Ratio (tangent portfolio) 
    def neg_sharpe(w):
        v = port_vol(w)
        return -(port_ret(w) - risk_free) / v if v > 1e-12 else 0.0

    res_msr = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=base_constraints,
                       options=_SLSQP_OPTS)
    msr_w   = res_msr.x if res_msr.success else w0
    msr_vol = port_vol(msr_w)
    msr_ret = port_ret(msr_w)

    #  4. Frontier curve, min-variance per target return, warm start
   
    def neg_ret(w):
        return -port_ret(w)

    res_maxret = minimize(neg_ret, w0, method="SLSQP",
                          bounds=bounds, constraints=base_constraints,
                          options=_SLSQP_OPTS)
    ret_max = -res_maxret.fun if res_maxret.success else float(np.max(br))

    ret_min     = gmv_ret
    # Tiny safety margin pulled inward (not outward) to stay just inside the
    # feasible region avoids edge-of-feasibility numerical failures.
    ret_max     = ret_min + (ret_max - ret_min) * 0.999
    target_rets = np.linspace(ret_min, ret_max, n_points)

    frontier_vols = []
    frontier_rets = []
    w_prev = gmv_w.copy()

    for target in target_rets:
        constraints = base_constraints + [
            {"type": "eq", "fun": lambda w, t=target: port_ret(w) - t}]
        res = minimize(port_vol, w_prev, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options=_SLSQP_OPTS)
        if res.success:
            frontier_vols.append(port_vol(res.x))
            frontier_rets.append(target)
            w_prev = res.x.copy()
        # On failure, retry once from the original GMV start (escape a bad
        # warm-start chain) before moving on 
        else:
            res_retry = minimize(port_vol, gmv_w, method="SLSQP",
                                 bounds=bounds, constraints=constraints,
                                 options=_SLSQP_OPTS)
            if res_retry.success:
                frontier_vols.append(port_vol(res_retry.x))
                frontier_rets.append(target)
                w_prev = res_retry.x.copy()

    frontier_vols    = np.array(frontier_vols)
    frontier_rets    = np.array(frontier_rets)
    frontier_sharpes = (frontier_rets - risk_free) / frontier_vols

    return (frontier_vols, frontier_rets, frontier_sharpes,
            gmv_vol, gmv_ret, msr_vol, msr_ret,
            rand_vols, rand_rets, rand_sharpes)


#  Chart builder 

def build_and_save_charts(metrics, output_path):
    """
    Generates a 3-chart dashboard and saves it as PNG.
    """
    pa          = metrics["per_asset"]
    port        = metrics["portfolio"]
    tickers     = metrics["tickers"]
    cov         = metrics["covariance"]
    bl_ret      = metrics["bl_returns"]
    rf          = metrics["risk_free"]
    max_weights = metrics.get("max_weights")   
    n           = len(tickers)
    colors = (SERIES_COLORS * (n // len(SERIES_COLORS) + 1))[:n]

    ret_v = port["Rendement Portefeuille"]
    vol_v = port["Volatilite Portefeuille"]
    sh_v  = port["Sharpe Portefeuille"]
    mdd_v = port["Max Drawdown"]

    #  Figure setup 
    fig = plt.figure(figsize=(24, 11), facecolor=BG_DARK)
    fig.suptitle(
        "BLACK-LITTERMAN  |  PORTFOLIO ANALYTICS",
        color=ORANGE, fontsize=15, fontweight="bold", y=0.98)
    
    subtitle = (
        f"Expected Return: {ret_v*100:.1f}%   |   "
        f"Volatility: {vol_v*100:.1f}%   |   "
        f"Sharpe: {sh_v:.2f}   |   "
        f"Max Drawdown: {mdd_v*100:.1f}%")

    fig.text(0.5, 0.943, subtitle, ha="center", color=GRAY_L, fontsize=9)
    fig.add_artist(Line2D(
        [0.03, 0.97], [0.932, 0.932],
        transform=fig.transFigure, color=ORANGE, linewidth=1.2))

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1, 1, 1.6],
        hspace=0.50, wspace=0.38,
        left=0.05, right=0.97, top=0.915, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, :2])  # Allocation + Risk Contribution
    ax2 = fig.add_subplot(gs[1, :2])  # BL Return vs Prior             
    ax3 = fig.add_subplot(gs[:, 2])   # Efficient Frontier

    ax3.set_box_aspect(1.0)

    #  1. GROUPED BAR: Allocation vs Risk Contribution
    x     = np.arange(n)
    width = 0.32
    alloc = pa["Poids Optimal"].values
    rc    = pa["Risk Contribution (%)"].values / 100

    bars_a = ax1.bar(x - width / 2, alloc, width,
                     label="Weight", color=ORANGE,
                     edgecolor=BG_DARK, linewidth=0.8, alpha=0.90)
    bars_r = ax1.bar(x + width / 2, rc, width,
                     label="Risk Contrib", color=TEAL,
                     edgecolor=BG_DARK, linewidth=0.8, alpha=0.90)

    _style_ax(ax1, "Allocation vs Risk Contribution", ylabel="Weight / Contribution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers, fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.legend(facecolor=BG_MID, edgecolor=ORANGE, labelcolor=GRAY_L, fontsize=8)
    _bar_labels(ax1, bars_a)
    _bar_labels(ax1, bars_r)

    # Diversification note: if weight ≈ risk contrib → well diversified
    ax1.text(0.98, 0.97,
             "Orange = Weight   |   Teal = Risk Contrib",
             transform=ax1.transAxes, ha="right", va="top",
             color=GRAY_M, fontsize=6.5, style="italic")

    #  2. GROUPED BAR: BL Return vs Prior
    prior = pa["Rendement Prior (CAPM)"].values
    bl_r  = pa["Rendement BL Ajuste"].values

    bars_p = ax2.bar(x - width / 2, prior, width, label="Prior (CAPM)",
                     color=TEAL, edgecolor=BG_DARK, linewidth=0.8, alpha=0.85)
    bars_b = ax2.bar(x + width / 2, bl_r, width, label="BL Return",
                     color=ORANGE, edgecolor=BG_DARK, linewidth=0.8, alpha=0.85)

    _style_ax(ax2, "BL Return vs Market Prior", ylabel="Annual Return")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.legend(facecolor=BG_MID, edgecolor=ORANGE, labelcolor=GRAY_L, fontsize=8)
    _bar_labels(ax2, bars_p)
    _bar_labels(ax2, bars_b)

    #  3. EFFICIENT FRONTIER 
    _style_ax(ax3, "Efficient Frontier", ylabel="Expected Return", xlabel="Volatility",
              grid_axis=None)
    ax3.grid(color=BG_MID, linewidth=0.5, linestyle="--")

    try:
        (fvols, frets, fsharpes,
         gmv_vol, gmv_ret,
         msr_vol, msr_ret,
         rand_vols, rand_rets, rand_sharpes) = _compute_efficient_frontier(
             bl_ret, cov, rf, max_weights=max_weights)

        #  Random portfolio cloud (background)
        ax3.scatter(
            rand_vols, rand_rets,
            c=rand_sharpes, cmap="plasma",
            s=6, alpha=0.35, zorder=1, linewidths=0)

        # Colour the frontier curve by Sharpe ratio
        from matplotlib.collections import LineCollection
        points  = np.array([fvols, frets]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm_sh  = plt.Normalize(fsharpes.min(), fsharpes.max())
        lc = LineCollection(segments, cmap="YlOrRd", norm=norm_sh, linewidth=3.0, zorder=4)
        lc.set_array(fsharpes[:-1])
        ax3.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax3, pad=0.02, fraction=0.035)
        cbar.set_label("Sharpe Ratio", color=GRAY_M, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=GRAY_M, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=GRAY_L)
        cbar.outline.set_edgecolor(BG_MID)

        # Capital Market Line
        cml_x = np.array([0, fvols.max() * 1.1])
        cml_y = rf + (msr_ret - rf) / msr_vol * cml_x
        ax3.plot(cml_x, cml_y, color=GRAY_M, linewidth=1.0,
                 linestyle="--", zorder=3, label="Capital Market Line")

        # GMV marker
        ax3.scatter(gmv_vol, gmv_ret, color=GREEN, s=100, zorder=6,
                    edgecolors=WHITE, linewidths=0.8, label="Min Variance")
        ax3.annotate(
            f"GMV\n{gmv_ret*100:.1f}% / {gmv_vol*100:.1f}%",
            xy=(gmv_vol, gmv_ret), xytext=(8, 6), textcoords="offset points",
            color=GREEN, fontsize=7.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8))

        # MSR / Tangency marker
        ax3.scatter(msr_vol, msr_ret, color=GOLD, s=150, zorder=7,
                    marker="*", edgecolors=WHITE, linewidths=0.7, label="Max Sharpe (Tangent)")
        ax3.annotate(
            f"MSR\n{msr_ret*100:.1f}% / {msr_vol*100:.1f}%",
            xy=(msr_vol, msr_ret), xytext=(8, -14), textcoords="offset points",
            color=GOLD, fontsize=7.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.8))

        # Current optimised portfolio
        ax3.scatter(vol_v, ret_v, color=ORANGE, s=130, zorder=8,
                    marker="D", edgecolors=WHITE, linewidths=0.8, label="Optimal Portfolio")
        ax3.annotate(
            f"Optimal\n{ret_v*100:.1f}% / {vol_v*100:.1f}%",
            xy=(vol_v, ret_v), xytext=(-72, 10), textcoords="offset points",
            color=ORANGE, fontsize=7.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))

        # Individual assets
        asset_vols = pa["Volatilite"].values
        asset_rets = pa["Rendement BL Ajuste"].values
        for i, t in enumerate(tickers):
            ax3.scatter(asset_vols[i], asset_rets[i],
                        color=colors[i], s=80, zorder=5,
                        marker="o", edgecolors=WHITE, linewidths=0.7)
            ax3.annotate(
                t, xy=(asset_vols[i], asset_rets[i]),
                xytext=(5, 4), textcoords="offset points",
                color=colors[i], fontsize=8, fontweight="bold")

        ax3.set_xlim(left=0)
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax3.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax3.legend(facecolor=BG_MID, edgecolor=ORANGE, labelcolor=GRAY_L,
                   fontsize=7.5, loc="lower right")

    except Exception as e:
        ax3.text(0.5, 0.5, f"Efficient Frontier\nunavailable\n({e})",
                 ha="center", va="center", color=RED, fontsize=9,
                 transform=ax3.transAxes)

    #  Footer
    fig.text(
        0.5, 0.01,
        "Black-Litterman Model  |  Generated by Python / matplotlib",
        ha="center", color=GRAY_M, fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)


#  Excel tables

def export_results(wb_xlwings, assets_df, weights, metrics):
    """
    1. Writes result tables into the 'Results' sheet via xlwings.
    2. Generates a PNG dashboard in the same folder as the .xlsm.
    3. Opens the PNG with the native Windows image viewer.
    """
    port = metrics["portfolio"]
    pa   = metrics["per_asset"]

    ws = wb_xlwings.sheets["Results"]
    ws.clear_contents()

    #  Table A: Portfolio Metrics 
    ws.range("A1").value = "BLACK-LITTERMAN  |  PORTFOLIO ANALYTICS"
    ws.range("A3").value = "PORTFOLIO METRICS"
    ws.range("A4").value = [["METRIC", "VALUE"]]

    mdd      = port["Max Drawdown"]
    mdd_lh   = port.get("Max Drawdown LookAhead", float('nan'))
    alpha_v  = port.get("Alpha vs Benchmark", float('nan'))
    beta_v   = port.get("Beta vs Benchmark", float('nan'))
    port_data = [
        ["Expected Return (annual)",          f"{port['Rendement Portefeuille']*100:.2f}%"],
        ["Volatility (annual)",               f"{port['Volatilite Portefeuille']*100:.2f}%"],
        ["Sharpe Ratio",                      f"{port['Sharpe Portefeuille']:.4f}"],
        ["VaR 95% (annual)",                  f"{port['VaR 95%']*100:.2f}%"],
        ["Expected Shortfall 95%",            f"{port['Expected Shortfall 95%']*100:.2f}%"],
        ["Max Drawdown (walk-forward)",       f"{mdd*100:.2f}%" if not np.isnan(mdd) else "—"],
        ["Max Drawdown (look-ahead, ref.)",   f"{mdd_lh*100:.2f}%" if not np.isnan(mdd_lh) else "—"],
        ["Diversification Index",             f"{port['Indice Diversification']:.4f}"],
        ["Alpha vs Benchmark (annual)",       f"{alpha_v*100:+.2f}%" if not np.isnan(alpha_v) else "—"],
        ["Beta vs Benchmark",                 f"{beta_v:.3f}" if not np.isnan(beta_v) else "—"],]
    ws.range("A5").value = port_data

    #  Table B: Asset-Level Analysis
    row_b = 5 + len(port_data) + 2
    ws.range(f"A{row_b}").value = "ASSET-LEVEL ANALYSIS"
    ws.range(f"A{row_b + 1}").value = [[
        "TICKER", "WEIGHT",
        "PRIOR RETURN (CAPM)", "BL RETURN (adjusted)", "SPREAD (BL - Prior)",
        "VOLATILITY", "SHARPE RATIO", "RISK CONTRIBUTION %"]]
    asset_data = []
    for _, row in pa.iterrows():
        asset_data.append([
            row["Ticker"],
            f"{row['Poids Optimal']*100:.1f}%",
            f"{row['Rendement Prior (CAPM)']*100:.2f}%",
            f"{row['Rendement BL Ajuste']*100:.2f}%",
            f"{row['Spread BL vs Prior']*100:+.2f}%",
            f"{row['Volatilite']*100:.2f}%",
            f"{row['Sharpe']:.4f}",
            f"{row['Risk Contribution (%)']:.1f}%",])
    ws.range(f"A{row_b + 2}").value = asset_data

    #  Table C: Portfolio Summary 
    row_c = row_b + 2 + len(asset_data) + 2
    ws.range(f"A{row_c}").value = "PORTFOLIO SUMMARY"
    ws.range(f"A{row_c + 1}").value = [[
        "Return", "Volatility", "Sharpe", "VaR 95%", "ES 95%",
        "Max DD (walk-fwd)", "Max DD (look-ahead)", "Alpha vs Bench", "Beta vs Bench"]]
    ws.range(f"A{row_c + 2}").value = [[
        f"{port['Rendement Portefeuille']*100:.2f}%",
        f"{port['Volatilite Portefeuille']*100:.2f}%",
        f"{port['Sharpe Portefeuille']:.2f}",
        f"{port['VaR 95%']*100:.2f}%",
        f"{port['Expected Shortfall 95%']*100:.2f}%",
        f"{mdd*100:.2f}%"    if not np.isnan(mdd)    else "—",
        f"{mdd_lh*100:.2f}%" if not np.isnan(mdd_lh) else "—",
        f"{alpha_v*100:+.2f}%" if not np.isnan(alpha_v) else "—",
        f"{beta_v:.3f}"        if not np.isnan(beta_v)  else "—",]]

    print("[excel_export] Tables written to 'Results' sheet.")

    #  PNG dashboard
    original_dir = os.path.dirname(wb_xlwings.fullname)
    chart_path   = os.path.join(original_dir, "BL_Analytics_Charts.png")

    build_and_save_charts(metrics, chart_path)
    print(f"[excel_export] Charts saved → {chart_path}")

    try:
        os.startfile(chart_path)
        print("[excel_export] Charts opened in Windows image viewer.")
    except AttributeError:
        print(f"[excel_export] Open manually: {chart_path}")
