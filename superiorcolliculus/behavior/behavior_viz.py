"""behavior_viz.py

Plotting helpers for the behavioral guidance-law analysis.  Two families:

  1. Discrete delay figures — replicate the SfN_2025_poster.ipynb boxplots
     (VAF by speed bin × tau × mode).

  2. Kernel figures — visualize causal/acausal/symmetric temporal kernels
     and compare them to the discrete-delay results.

All functions return a matplotlib Figure so the caller controls saving.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

DT_MS = 1000.0 / 30.0   # 33.33 ms per frame


# ---------------------------------------------------------------------------
# Discrete delay figures
# ---------------------------------------------------------------------------

def plot_vaf_boxplot(
    reg_df: pd.DataFrame,
    x: str = "tau (ms)",
    y: str = "VAF",
    col: str = "mode",
    hue: str = "GL, speed (cm/s)",
    model_filter: str = "II",
    ylim: tuple = (0, 1),
    figsize: tuple = None,
) -> plt.Figure:
    """Faceted boxplot of VAF by delay × mode, coloured by GL × speed bin.

    Replicates the main SfN poster figure.

    Parameters
    ----------
    reg_df       : output of guidance_law_fitting.run_all_guidance_laws
    x            : x-axis column (discrete delay in ms)
    y            : metric column ('VAF' or 'r2')
    col          : facet column
    hue          : colour grouping, typically 'GL, speed (cm/s)'
    model_filter : keep only rows where reg_df['model'] == model_filter;
                   pass None to skip filtering
    ylim         : y-axis limits
    figsize      : passed to catplot; None lets seaborn choose

    Returns
    -------
    matplotlib Figure
    """
    plot_df = reg_df.copy()
    if model_filter is not None and "model" in plot_df.columns:
        plot_df = plot_df.loc[plot_df["model"] == model_filter]

    n_hue = plot_df[hue].nunique() if hue in plot_df.columns else 1
    palette = sns.color_palette("tab20", n_colors=min(n_hue, 20))

    kw = {}
    if figsize is not None:
        kw["height"] = figsize[1]
        kw["aspect"] = figsize[0] / figsize[1]

    g = sns.catplot(
        data=plot_df,
        x=x, y=y, col=col, hue=hue,
        kind="box",
        palette=palette,
        linewidth=0.8,
        fliersize=2,
        **kw,
    )
    g.set(ylim=ylim)
    g.set_axis_labels(x, y)
    g.set_titles("{col_name}")
    plt.tight_layout()
    return g.figure


def plot_vaf_by_speed(
    reg_df: pd.DataFrame,
    tau_ms: float,
    mode: str,
    gl: str = "PN",
    y: str = "VAF",
    ax: plt.Axes = None,
) -> plt.Figure:
    """Boxplot of VAF vs speed bin for a single (mode, GL, tau) condition.

    Parameters
    ----------
    reg_df : full results DataFrame
    tau_ms : which delay to show (e.g. 0 or 99)
    mode   : behavioral mode to filter
    gl     : 'PN' or 'PP'
    y      : metric column

    Returns
    -------
    matplotlib Figure
    """
    sub = reg_df.loc[
        (reg_df["mode"] == mode) &
        (reg_df["GL"] == gl) &
        (reg_df["tau (ms)"] == tau_ms)
    ]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.figure

    sns.boxplot(data=sub, x="i", y=y, ax=ax, color="steelblue", linewidth=0.8,
                fliersize=2)
    ax.set(xlabel="speed (cm/s)", ylabel=y, ylim=(0, 1),
           title=f"{mode}  {gl}  τ={tau_ms:.0f} ms")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Kernel figures
# ---------------------------------------------------------------------------

def _lag_axis(n_lags: int, mode: str, dt_ms: float = DT_MS) -> np.ndarray:
    """Return time axis in ms for a kernel of length n_lags."""
    if (mode == "symmetric") or (mode=="weighted"):
        L = (n_lags - 1) // 2
        return np.arange(-L, L + 1) * dt_ms
    else:
        return np.arange(n_lags) * dt_ms


def plot_kernel(
    kernel: np.ndarray,
    mode: str = "causal",
    dt_ms: float = DT_MS,
    title: str = "",
    ax: plt.Axes = None,
) -> plt.Figure:
    """Plot a single temporal integration kernel.

    Causal/acausal: x-axis is lag in ms (0 = current frame, positive = past
    for causal, positive = future for acausal).
    Symmetric: x-axis spans −L·dt to +L·dt, negative = future, positive = past.

    Parameters
    ----------
    kernel : 1-D weight array, shape (n_lags,)
    mode   : 'causal' | 'acausal' | 'symmetric'
    dt_ms  : ms per frame
    title  : axis title
    ax     : existing Axes to draw on; creates new figure if None

    Returns
    -------
    matplotlib Figure
    """
    kernel = np.asarray(kernel)
    t = _lag_axis(len(kernel), mode, dt_ms)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.5))
    else:
        fig = ax.figure

    ax.bar(t, kernel, width=dt_ms * 0.85, color="steelblue", edgecolor="none")
    ax.axhline(0, color="k", lw=0.6)
    
    if mode == "symmetric":
        ax.axvline(0, color="k", lw=0.6, ls="--", label="current frame")

    xlabel = "lag (ms)" if mode in ("causal", "acausal") else "time offset (ms)  [−=future, +=past]"
    ax.set(xlabel=xlabel, ylabel="weight", title=title)
    fig.tight_layout()
    return fig


def plot_kernel_comparison(
    causal_kernel: np.ndarray,
    acausal_kernel: np.ndarray,
    dt_ms: float = DT_MS,
    title: str = "",
) -> plt.Figure:
    """Side-by-side causal and acausal kernels with mirrored x-axes.

    Both kernels have length L (not 2L+1). The causal panel shows past
    integration (positive lag = further in the past); the acausal panel shows
    future anticipation (positive lag = further in the future). Plotting them
    as mirrors makes temporal asymmetry visually apparent.

    Returns
    -------
    matplotlib Figure with two panels
    """
    causal_kernel  = np.asarray(causal_kernel)
    acausal_kernel = np.asarray(acausal_kernel)
    L  = len(causal_kernel)
    t  = np.arange(L) * dt_ms

    fig, (ax_c, ax_a) = plt.subplots(1, 2, figsize=(6, 2.5), sharey=True)

    ax_c.bar(t, causal_kernel,  width=dt_ms * 0.85, color="steelblue", edgecolor="none")
    ax_a.bar(t, acausal_kernel, width=dt_ms * 0.85, color="tomato",    edgecolor="none")
    for ax in (ax_c, ax_a):
        ax.axhline(0, color="k", lw=0.6)

    ax_c.set(xlabel="lag into past (ms)",   ylabel="weight", title="causal")
    ax_a.set(xlabel="lag into future (ms)", title="acausal")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_kernel_overlay(
    kernel_results_df: pd.DataFrame,
    predictor: str,
    mode: str,
    kernel_col: str = "mean_symmetric_kernel",
    dt_ms: float = DT_MS,
    ax: plt.Axes = None,
    legend: bool = True,
) -> plt.Figure:
    """One panel showing each subject's kernel as a separate line.

    The kernel is fit once per subject across all bouts; speed-bin rows are
    deduplicated so each subject appears exactly once.  Lines are coloured by
    subject (tab10).  A vertical dashed line marks t=0 for symmetric kernels.

    Parameters
    ----------
    kernel_results_df : output of kernel_regression.run_kernel_analysis_by_mode_speed
    predictor         : 'alphadot' or 'Phi' (used for axis/title labels)
    mode              : behavioral mode string (used for title)
    kernel_col        : which kernel column to plot ('mean_symmetric_kernel',
                        'mean_causal_kernel', 'mean_acausal_kernel',
                        or 'mean_weighted_kernel')
    dt_ms             : ms per frame

    Returns
    -------
    matplotlib Figure
    """
    df = kernel_results_df.dropna(subset=[kernel_col])


    if "symmetric" in kernel_col:
        kernel_mode = "symmetric"
    elif "acausal" in kernel_col:
        kernel_mode = "acausal"
    elif "causal" in kernel_col:
        kernel_mode = "causal"
    elif "weighted" in kernel_col:
        kernel_mode = "weighted"
    else:
        raise ValueError(f"unrecognized kernel_col: {kernel_col}")

    # One row per subject — kernel is identical across speed-bin rows
    has_subjects = "subject" in df.columns
    if has_subjects:
        df_plot = df.drop_duplicates(subset="subject")
        subjects = list(df_plot["subject"])
    else:
        df_plot = df.iloc[[0]]
        subjects = ["all"]

    n_subj = len(subjects)
    palette = sns.color_palette("tab10", n_colors=min(n_subj, 10))
    subj_color = {s: palette[i % 10] for i, s in enumerate(subjects)}

    first_k = np.asarray(df_plot[kernel_col].iloc[0])
    t = _lag_axis(len(first_k), kernel_mode, dt_ms)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.figure

    for _, row in df_plot.iterrows():
        subj = row["subject"] if has_subjects else "all"
        k = np.asarray(row[kernel_col])
        ax.plot(t, k, color=subj_color[subj], lw=1.4, label=subj)
        # ax.axhline(np.mean(k), zorder=-np.inf, alpha=0.6)

    ax.axhline(0, color="k", lw=0.5)
    
    if (kernel_mode == "symmetric") or (kernel_mode == "weighted"):
        ax.axvline(0, color="k", lw=0.8, ls="--")

    xlabel = "time offset (ms)  [−=future, +=past]" if kernel_mode == "symmetric" else "lag (ms)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("weight")
    ax.set_title(f"{kernel_mode} kernel")

    if n_subj > 1 and legend:
        ax.legend(title="subject", fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.35, 0))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparison figures
# ---------------------------------------------------------------------------

def plot_causal_vs_acausal(
    kernel_results_df: pd.DataFrame,
    mode: str,
    predictor: str,
    metric: str = "r2",
    ax: plt.Axes = None,
    legend: bool = True,
) -> plt.Figure:
    """Scatter of causal r² vs acausal r² per speed bin.

    Points on the diagonal indicate no temporal asymmetry.
    Points above the diagonal mean the acausal model outperforms the causal
    one (thetadot leads the predictor, or both are driven by a shared signal).

    Returns
    -------
    matplotlib Figure
    """
    df = kernel_results_df.dropna(subset=["causal_r2", "acausal_r2", 'causal_VAF', 'acausal_VAF'])
    if np.any(ax == None):
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    has_subjects = "subject" in df.columns and df["subject"].nunique() > 1

    # Colour by speed bin (coolwarm); subjects use different markers
    bin_order = sorted(df["speed_bin"].unique(), key=lambda s: int(s.split("-")[0]))
    n_bins = len(bin_order)
    bin_idx = {b: i for i, b in enumerate(bin_order)}
    bin_colors = {b: plt.cm.coolwarm(i / max(n_bins - 1, 1)) for i, b in enumerate(bin_order)}

    all_vals = pd.concat([df[f"causal_{metric}"], df[f"acausal_{metric}"]])
    lim = max(float(all_vals.max()), 0.1)

    if has_subjects:
        subjects = sorted(df["subject"].unique())
        subj_markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
        for subj, marker in zip(subjects, subj_markers):
            sdf = df[df["subject"] == subj]
            colors = [bin_colors[b] for b in sdf["speed_bin"]]
            ax.scatter(
                sdf[f"causal_{metric}"], sdf[f"acausal_{metric}"],
                c=colors, marker=marker, s=55, zorder=3,
                edgecolors="k", linewidths=0.4,
            )
    else:
        colors = [bin_colors[b] for b in df["speed_bin"]]
        ax.scatter(
            df[f"causal_{metric}"], df[f"acausal_{metric}"],
            c=colors, s=60, zorder=3,
        )

    # Legend: speed bin → colour
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=bin_colors[b],
                   markeredgecolor="k", markeredgewidth=0.4, markersize=6, label=b)
        for b in bin_order
    ]
    if legend:
        ax.legend(handles=handles, title="speed bin", fontsize=6, loc="lower right", bbox_to_anchor=(1.45, 0),
            frameon=False)

    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set(
        xlabel=f"causal {metric}",
        ylabel=f"acausal {metric}",
        
    )
    return fig


def plot_approach_comparison(
    discrete_df: pd.DataFrame,
    kernel_df: pd.DataFrame,
    mode: str,
    predictor: str,
    discrete_metric: str = "VAF",
    kernel_metric: str = "median_r2_test",
) -> plt.Figure:
    """Two-panel comparison: kernel r² vs best discrete-tau performance.

    Left panel: scatter of kernel r² vs best-tau discrete VAF per speed bin.
                Points on the diagonal indicate equal performance.
                Points above the diagonal indicate the kernel approach wins.

    Right panel: kernel median peak lag (ms) vs best discrete tau (ms) per
                 speed bin.  Points on the diagonal confirm the kernel
                 recovers the same dominant delay as the discrete search.

    Parameters
    ----------
    discrete_df    : reg_df slice for this mode + predictor (all tau values)
    kernel_df      : output of kernel_regression.run_kernel_analysis_by_mode_speed
    mode, predictor: used for plot title
    discrete_metric: column in discrete_df to compare ('VAF' or 'r2')
    kernel_metric  : column in kernel_df for the kernel performance

    Returns
    -------
    matplotlib Figure with two panels
    """
    _PRED_TO_GL = {"alphadot": "PN", "Phi": "PP", "Phidot": "dPP", "alpha": "iPN"}
    gl = _PRED_TO_GL.get(predictor, predictor)

    # Best discrete tau per speed bin (maximum VAF across tau values)
    best_discrete = (
        discrete_df.loc[discrete_df["GL"] == gl]
        .groupby("i")[discrete_metric]
        .max()
        .reset_index()
        .rename(columns={"i": "speed_bin", discrete_metric: "best_discrete"})
    )

    best_tau_ms = (
        discrete_df.loc[discrete_df["GL"] == gl]
        .sort_values(discrete_metric, ascending=False)
        .drop_duplicates("i")[["i", "tau (ms)"]]
        .rename(columns={"i": "speed_bin", "tau (ms)": "best_tau_ms"})
    )

    # Merge with kernel results
    merged = (
        kernel_df[["speed_bin", kernel_metric, "median_peak_lag_ms"]]
        .merge(best_discrete, on="speed_bin", how="inner")
        .merge(best_tau_ms,   on="speed_bin", how="inner")
    )

    if merged.empty:
        fig, ax = plt.subplots()
        ax.set_title("no overlapping speed bins")
        return fig

    n = len(merged)
    colors = sns.color_palette("coolwarm", n_colors=n)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7, 3.2))

    # Left: performance comparison
    lim = max(merged[[kernel_metric, "best_discrete"]].max().max(), 0.1)
    for i, (_, row) in enumerate(merged.iterrows()):
        ax_l.scatter(row["best_discrete"], row[kernel_metric],
                     color=colors[i], s=60, zorder=3)
        ax_l.annotate(row["speed_bin"], (row["best_discrete"], row[kernel_metric]),
                      fontsize=6, ha="left", va="bottom")
    ax_l.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax_l.set(
        xlabel=f"best discrete τ {discrete_metric}",
        ylabel=f"kernel {kernel_metric}",
        xlim=(-1, lim * 1.05), ylim=(-1, lim * 1.05),
        title="performance",
    )

    # Right: peak lag comparison
    all_lags = pd.concat([merged["best_tau_ms"], merged["median_peak_lag_ms"]])
    lag_lim = max(all_lags.max(), 10)
    for i, (_, row) in enumerate(merged.iterrows()):
        ax_r.scatter(row["best_tau_ms"], row["median_peak_lag_ms"],
                     color=colors[i], s=60, zorder=3)
        ax_r.annotate(row["speed_bin"], (row["best_tau_ms"], row["median_peak_lag_ms"]),
                      fontsize=6, ha="left", va="bottom")
    ax_r.plot([0, lag_lim], [0, lag_lim], "k--", lw=0.8)
    ax_r.set(
        xlabel="best discrete τ (ms)",
        ylabel="kernel peak lag (ms)",
        xlim=(0, lag_lim * 1.1), ylim=(0, lag_lim * 1.1),
        title="delay agreement",
    )

    fig.suptitle(f"{mode}  {predictor}", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Single-bout model comparison
# ---------------------------------------------------------------------------

def plot_bout_model_comparison(
    bout_data: pd.DataFrame,
    predictor: str,
    response: str = "thetadot",
    speed_col: str = "speed",
    L: int = 20,
    alpha: float = 100.0,
    tau_frames: int = 3,
    xlim: tuple = None,
    title: str = "",
    sampling_rate: float = 30.0,
    DISCRETE_TAU: bool = True,
    KERNEL: bool = True,
    cv_params: dict = None,
    pd_params: dict = None,
    loo_kernel: dict = None,
    loo_bilinear_kernel: dict = None,
    kernel_mode: str = "causal",
) -> plt.Figure:
    """Two-axis figure comparing plain and bilinear models on one hunting bout.

    Top axis  : raw predictor time series.
    Bottom axis: ground-truth response + model predictions:
        - Guidance law (plain Type II)  at τ=0 and τ=tau_frames * dt
        - Bilinear guidance law          at τ=tau_frames
        - Plain symmetric kernel (Ridge)
        - Bilinear speed-modulated kernel
        - PD controller  (optional, via pd_params)
    Inset (bottom-right of bottom axis): fitted symmetric kernel shape.

    Parameters
    ----------
    bout_data     : DataFrame for one bout; must contain predictor, response,
                    speed_col, and (for PD) 'Phi'/'Phidot' columns.
    predictor     : 'alphadot' or 'Phi'
    response      : column name, default 'thetadot'
    speed_col     : column name for speed, default 'speed'
    L             : kernel half-width in frames
    alpha         : Ridge regularisation strength
    tau_frames    : discrete delay for the shifted GL fit
    xlim          : (t_start, t_end) in seconds; None shows full bout
    title         : suptitle; auto-generated from bout metadata if empty
    sampling_rate : frames per second
    DISCRETE_TAU  : whether to include the shifted GL fit (requires tau_frames > 0)
    KERNEL        : whether to include kernel
    cv_params     : optional dict of cross-validated GL parameters to use instead
                    of fitting on this bout.  Expected keys:
                      'slope_tau0'  — float, Type II slope at τ=0
                      'slope_tau'   — float, Type II slope at τ=tau_frames
                    Any missing key falls back to in-bout fitting.
    pd_params     : optional dict for PD controller overlay.  Expected keys:
                      'A'    — float, Phi gain
                      'B'    — float, Phidot gain
                      'tau1' — int, Phi delay in frames
                      'tau2' — int, Phidot delay in frames
    loo_kernel    : optional dict returned by kr.fit_behavioral_kernel_loo.
                    When provided, skips in-bout plain kernel fitting and uses
                    these pre-fit LOO weights instead.  r2_test reflects
                    held-out bout performance.  No train/test split line is drawn.
    loo_bilinear_kernel : optional dict returned by
                    kr.fit_speed_modulated_kernel_loo.  Same semantics as
                    loo_kernel but for the bilinear model.

    Returns
    -------
    matplotlib Figure
    """
    from superiorcolliculus.behavior import kernel_regression as kr
    from superiorcolliculus.behavior import guidance_law_fitting as glf

    dt_ms = 1000.0 / sampling_rate

    x_raw = bout_data[predictor].values.copy()
    y     = bout_data[response].values
    speed = bout_data[speed_col].values
    T = len(x_raw)
    t = np.arange(T) / sampling_rate

    cv = cv_params or {}

    if DISCRETE_TAU:
    # ── GL τ=0 (plain Type II) ────────────────────────────────────────────────
        if "slope_tau0" in cv:
            slope0 = cv["slope_tau0"]
            r2_gl0 = np.corrcoef(x_raw, y)[0, 1] ** 2
        else:
            slope0, r2_gl0 = glf.type2_regression(x_raw, y)
        y_hat_gl0 = float(slope0) * x_raw

    # ── GL τ=tau_frames (plain, shifted fit) ──────────────────────────────────
        if "slope_tau" in cv:
            slope3 = cv["slope_tau"]
            r2_gl3 = np.corrcoef(x_raw[:-tau_frames if tau_frames > 0 else None], y[tau_frames or None:])[0, 1] ** 2
        else:
            slope3, r2_gl3 = glf.type2_regression(x_raw[:-tau_frames if tau_frames > 0 else None], y[tau_frames or None:])
        y_hat_gl3 = np.full(T, np.nan)
        y_hat_gl3[tau_frames or None:] = float(slope3) * x_raw[:-tau_frames if tau_frames > 0 else None]

    # ── Plain kernel ──────────────────────────────────────────────────────────
    if KERNEL:
        k_fit = loo_kernel if loo_kernel is not None else \
                kr.fit_behavioral_kernel(x_raw, y, L=L, alpha=alpha, mode=kernel_mode)
        if k_fit is not None:
            _k_mode = k_fit["config"].get("mode", kernel_mode)
            _k_L    = k_fit["config"].get("L", L)
            X_k, _  = kr.build_behavioral_lag_matrix(x_raw, _k_L, mode=_k_mode)
            y_hat_k = X_k @ k_fit["coef"] + k_fit["intercept"]
            train_fraction = k_fit["config"].get("train_fraction")
            split_t = (int(T * train_fraction) / sampling_rate) if train_fraction is not None else None
        else:
            y_hat_k, split_t = np.full(T, np.nan), None

        # ── Bilinear kernel ───────────────────────────────────────────────────────
        bl_k = loo_bilinear_kernel if loo_bilinear_kernel is not None else None
        if bl_k is not None:
            _bl_mode = bl_k["config"].get("mode", kernel_mode)
            _bl_L    = bl_k["config"].get("L", L)
            X_bl, _  = kr.build_behavioral_lag_matrix(x_raw, _bl_L, mode=_bl_mode)
            spd_z_k  = (speed - bl_k["config"]["speed_mean"]) / bl_k["config"]["speed_std"]
            X_aug    = np.hstack([X_bl, spd_z_k[:, None] * X_bl])
            y_hat_bl_k = X_aug @ np.concatenate([bl_k["K0"], bl_k["K1"]]) + bl_k["intercept"]
        else:
            y_hat_bl_k = np.full(T, np.nan)

    # ── figure ────────────────────────────────────────────────────────────────
    tau_ms     = int(tau_frames * dt_ms)
    top_units  = {"alphadot": "deg/s", "Phi": "deg", "Phidot": "deg/s", "alpha": "deg"}.get(predictor, "")
    top_label  = f"{predictor} ({top_units})" if top_units else predictor

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax_top.plot(t, x_raw, lw=0.9, color="C2")
    ax_top.set_ylabel(top_label)
    ax_top.set_title(title or predictor)
    ax_top.tick_params(labelbottom=False)

    ax_bot.plot(t, y,           lw=0.9, color="C0",  label="thetadot")
    
    if DISCRETE_TAU:
        # ax_bot.plot(t, y_hat_gl0,   lw=0.9, color="C3",  ls="--",
        #             label=f"GL \u03c4=0 (r\u00b2={float(r2_gl0):.3f})")
        ax_bot.plot(t, y_hat_gl3,   lw=0.9, color="C4",  ls="--",
                    label=f"GL \u03c4={tau_ms}ms (r\u00b2={float(r2_gl3):.3f})")
    if KERNEL:
        if k_fit is not None:
            k_label = "kernel" if k_fit["config"].get("loo") else "kernel"
            ax_bot.plot(t, y_hat_k, lw=1.1, color="C1",
                        label=f"{k_label} (r\u00b2={k_fit['r2_test']:.3f})")
            if split_t is not None:
                ax_bot.axvline(split_t, color="gray", lw=0.8, ls=":", label="train/test")
        if bl_k is not None:
            bl_label = "bilinear kernel" if bl_k["config"].get("loo") else "bilinear kernel"
            ax_bot.plot(t, y_hat_bl_k, lw=1.1, color="C5",
                        label=f"{bl_label} (r\u00b2={bl_k['r2_test']:.3f})")

    # ── PD controller overlay ─────────────────────────────────────────────────
    if pd_params is not None:
        A_pd = float(pd_params["A"])
        B_pd = float(pd_params["B"])
        t1   = int(pd_params["tau1"])
        t2   = int(pd_params["tau2"])

        phi_s    = np.full(T, np.nan)
        phidot_s = np.full(T, np.nan)
        phi_raw    = bout_data["Phi"].values
        phidot_raw = bout_data["Phidot"].values
        phi_s[t1:]    = phi_raw[:-t1]    if t1 > 0 else phi_raw
        phidot_s[t2:] = phidot_raw[:-t2] if t2 > 0 else phidot_raw
        y_hat_pd = A_pd * phi_s + B_pd * phidot_s

        valid = ~np.isnan(y_hat_pd)
        r2_pd = float(np.corrcoef(y_hat_pd[valid], y[valid])[0, 1] ** 2) if valid.sum() > 1 else float("nan")
        t1_ms = int(round(t1 * dt_ms))
        t2_ms = int(round(t2 * dt_ms))
        ax_bot.plot(t, y_hat_pd, lw=1.1, color="C9", ls="-",
                    label=f"PD (A={A_pd:.2f}, B={B_pd:.2f}, \u03c4\u2081={t1_ms}ms, \u03c4\u2082={t2_ms}ms, r\u00b2={r2_pd:.3f})")

    ax_bot.set_ylabel(f"{response} (deg/s)")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.legend(frameon=False, fontsize=8, ncol=3)

    if xlim is not None:
        ax_top.set_xlim(xlim)

    # ── Kernel inset (bottom-right of bottom axis) ────────────────────────────
    if KERNEL and k_fit is not None:
        ax_inset = ax_bot.inset_axes([0.79, 0.03, 0.20, 0.48])
        ax_inset.patch.set_alpha(0.6)
        _inset_mode = k_fit["config"].get("mode", kernel_mode)
        _inset_L    = k_fit["config"].get("L", L)
        _n_k = len(k_fit["coef"])
        if _inset_mode == "symmetric":
            lags_ms = (np.arange(_n_k) - _inset_L) * dt_ms
        else:
            lags_ms = np.arange(_n_k) * dt_ms
        ax_inset.plot(lags_ms, k_fit["coef"], lw=1.0, color="C1")
        ax_inset.axhline(0, color="k", lw=0.5)
        ax_inset.axvline(0, color="k", lw=0.5, ls=":")
        ax_inset.set_xlabel("lag (ms)", fontsize=6)
        ax_inset.tick_params(labelsize=5)
        ax_inset.set_xlim(lags_ms[0], lags_ms[-1])
        ax_inset.spines["top"].set_visible(False)
        ax_inset.spines["right"].set_visible(False)
        ax_inset.text(0.04, 0.95, "kernel", transform=ax_inset.transAxes,
                      fontsize=6, va="top", ha="left")

    fig.tight_layout()
    return fig


def plot_bout_mixed_comparison(
    bout_data: pd.DataFrame,
    coef_A_alone: np.ndarray,
    coef_B_alone: np.ndarray,
    coef_A_mixed: np.ndarray,
    coef_B_mixed: np.ndarray,
    ic_A: float,
    ic_B: float,
    ic_mixed: float,
    L: int,
    predictor_A: str = "alphadot",
    predictor_B: str = "Phi",
    response: str = "thetadot",
    sampling_rate: float = 30.0,
    xlim: tuple = None,
    title: str = "",
) -> plt.Figure:
    """Three-panel figure comparing PN, PP, and mixed kernel predictions on one bout.

    Uses pre-fit LOO kernels (from kr.collect_all_predictions_loo bout dicts)
    so no refitting occurs here.

    Layout
    ------
    Row 0 (small) : predictor_A time series  (e.g. alphadot)
    Row 1 (small) : predictor_B time series  (e.g. Phi)
    Row 2 (large) : thetadot + PN prediction + PP prediction + mixed prediction
                    Two kernel insets (bottom-right): PN alone/mixed, PP alone/mixed

    Parameters
    ----------
    bout_data    : DataFrame for one bout
    coef_A_alone : (n_lags,) kernel for predictor_A from the A-only LOO fit
    coef_B_alone : (n_lags,) kernel for predictor_B from the B-only LOO fit
    coef_A_mixed : (n_lags,) predictor_A marginal kernel from the mixed LOO fit
    coef_B_mixed : (n_lags,) predictor_B marginal kernel from the mixed LOO fit
    ic_A, ic_B, ic_mixed : intercepts from the respective LOO fits
    L            : kernel half-width in frames (used for lag axis only)
    predictor_A  : column name, default 'alphadot'
    predictor_B  : column name, default 'Phi'
    response     : column name, default 'thetadot'
    sampling_rate: Hz
    xlim         : (t_start, t_end) in seconds; None = full bout
    title        : figure suptitle

    Returns
    -------
    matplotlib Figure
    """
    from superiorcolliculus.behavior import kernel_regression as kr

    dt_ms = 1000.0 / sampling_rate

    x_A  = bout_data[predictor_A].values.astype(float)
    x_B  = bout_data[predictor_B].values.astype(float)
    y    = bout_data[response].values.astype(float)
    T    = len(y)
    t    = np.arange(T) / sampling_rate

    # Reconstruct predictions from pre-fit kernels
    XA, _ = kr.build_behavioral_lag_matrix(x_A, L, mode="symmetric")
    XB, _ = kr.build_behavioral_lag_matrix(x_B, L, mode="symmetric")
    X_mx  = np.hstack([XA, XB])

    y_hat_A  = XA  @ coef_A_alone                                   + ic_A
    y_hat_B  = XB  @ coef_B_alone                                   + ic_B
    y_hat_mx = X_mx @ np.concatenate([coef_A_mixed, coef_B_mixed]) + ic_mixed

    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_A   = 1.0 - np.sum((y - y_hat_A)  ** 2) / ss_tot if ss_tot > 0 else float("nan")
    r2_B   = 1.0 - np.sum((y - y_hat_B)  ** 2) / ss_tot if ss_tot > 0 else float("nan")
    r2_mx  = 1.0 - np.sum((y - y_hat_mx) ** 2) / ss_tot if ss_tot > 0 else float("nan")

    lags_ms = (np.arange(2 * L + 1) - L) * dt_ms
    units   = {"alphadot": "deg/s", "Phi": "deg", "Phidot": "deg/s"}

    # ── Layout: 3 rows, height ratios [1, 1, 3] ──────────────────────────────
    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(3, 1, height_ratios=[1, 1, 3], hspace=0.06)
    ax_a   = fig.add_subplot(gs[0])
    ax_b   = fig.add_subplot(gs[1], sharex=ax_a)
    ax_bot = fig.add_subplot(gs[2], sharex=ax_a)

    ax_a.plot(t, x_A, lw=0.9, color="C0")
    ax_a.set_ylabel(f"{predictor_A} ({units.get(predictor_A,'')})", fontsize=8)
    ax_a.tick_params(labelbottom=False)
    ax_a.spines["bottom"].set_visible(False)

    ax_b.plot(t, x_B, lw=0.9, color="C1")
    ax_b.set_ylabel(f"{predictor_B} ({units.get(predictor_B,'')})", fontsize=8)
    ax_b.tick_params(labelbottom=False)
    ax_b.spines["bottom"].set_visible(False)

    ax_bot.plot(t, y,        lw=1.0, color="k",  label=response)
    ax_bot.plot(t, y_hat_A,  lw=1.1, color="C0",
                label=f"PN ({predictor_A})  r\u00b2={r2_A:.3f}")
    ax_bot.plot(t, y_hat_B,  lw=1.1, color="C1",
                label=f"PP ({predictor_B})  r\u00b2={r2_B:.3f}")
    ax_bot.plot(t, y_hat_mx, lw=1.5, color="C2", ls="--",
                label=f"Mixed  r\u00b2={r2_mx:.3f}")
    ax_bot.set_ylabel(f"{response} (deg/s)")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.legend(frameon=False, fontsize=8, ncol=2)

    if xlim is not None:
        ax_a.set_xlim(xlim)

    # ── Kernel insets: stacked vertically on the right edge ──────────────────
    # [x0, y0, width, height] in axes fraction; both share the same x column.
    INSET_X = 1.051
    INSET_W = 0.16

    def _kernel_inset(parent_ax, y0, inset_h, coef_alone, coef_mixed, color, label,
                      show_xlabel=False):
        ax_in = parent_ax.inset_axes([INSET_X, y0, INSET_W, inset_h])
        ax_in.patch.set_alpha(0.75)
        ax_in.plot(lags_ms, coef_alone, color=color, lw=1.2, label="alone")
        ax_in.plot(lags_ms, coef_mixed, color=color, lw=1.2, ls="--", label="mixed")
        ax_in.axhline(0, color="k", lw=0.5)
        ax_in.axvline(0, color="k", lw=0.5, ls=":")
        ax_in.tick_params(labelsize=5)
        ax_in.set_xlim(lags_ms[0], lags_ms[-1])
        if show_xlabel:
            ax_in.set_xlabel("lag (ms)", fontsize=6)
        else:
            ax_in.tick_params(labelbottom=False)
        for sp in ("top", "right"):
            ax_in.spines[sp].set_visible(False)
        ax_in.text(0.05, 0.96, label, transform=ax_in.transAxes,
                   fontsize=6, va="top", color=color)
        ax_in.legend(fontsize=5, frameon=False, loc="lower right")

    # Top inset: predictor A (PN / alphadot)
    _kernel_inset(ax_bot, 0.53, 0.44, coef_A_alone, coef_A_mixed, "C0", predictor_A,
                  show_xlabel=False)
    # Bottom inset: predictor B (PP / Phi)
    _kernel_inset(ax_bot, 0.04, 0.44, coef_B_alone, coef_B_mixed, "C1", predictor_B,
                  show_xlabel=True)

    fig.suptitle(title, y=1.01, fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# Paired-bootstrap VAF comparison figures (e.g. PP vs. PN) -- moved from
# dissemination/CCN_2026_poster/CCN_2026_behavior.ipynb's notebook-local
# plotting cell. Pair with analysis/behavior/vaf_bootstrap.py's
# bootstrap_vaf_df/pair_by_replicate/compute_bootstrap_significance, which
# produce the paired_df/sig_df these expect.
# ---------------------------------------------------------------------------

def add_cross_summary(ax, paired_df, bin_col, order, palette):
    """Per-bin cross marker: center at (mean x, mean y), arms independently
    spanning each axis's own IQR (25th-75th percentile) -- x-arm from the
    first paired column's quartiles, y-arm from the second's. `paired_df`'s
    two value columns are plotted in the same order scatter_paired uses
    (its 2nd and 3rd columns)."""
    value_cols = [c for c in paired_df.columns if c != bin_col]
    xcol, ycol = value_cols[0], value_cols[1]
    for b in order:
        sub = paired_df[paired_df[bin_col] == b]
        if len(sub) == 0:
            continue
        x, y = sub[xcol].values, sub[ycol].values
        mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
        q1_x, q3_x = np.percentile(x, [25, 75])
        q1_y, q3_y = np.percentile(y, [25, 75])
        color = palette[b]
        ax.errorbar(mean_x, mean_y,
                     xerr=[[mean_x - q1_x], [q3_x - mean_x]],
                     yerr=[[mean_y - q1_y], [q3_y - mean_y]],
                     fmt="o", color=color, markersize=5, markeredgecolor="black", markeredgewidth=0.8,
                     ecolor="black", elinewidth=1.6, capsize=4, capthick=1.4, zorder=6)


def scatter_paired(ax, paired_df, bin_col, order, palette, x="vaf_pp", y="vaf_pn"):
    """Scattered paired-bootstrap replicates (x vs. y, colored by bin) plus
    a unity line and add_cross_summary's per-bin mean+IQR cross."""
    sns.scatterplot(data=paired_df, x=x, y=y, hue=bin_col, hue_order=order,
                     palette=palette, s=8, alpha=0.5, linewidth=0, ax=ax, legend=False)
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", zorder=0)
    add_cross_summary(ax, paired_df, bin_col, order, palette)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)


def annotate_bin_significance(ax, sig_df, column_filter):
    """Stack each bin's Holm-corrected p-value in a corner of the panel.
    `sig_df` is compute_bootstrap_significance's output, Holm-corrected into
    a "p_holm" column by the caller (see superiorcolliculus.analysis.
    holm_correction)."""
    sub = sig_df[sig_df["column"] == column_filter]
    lines = []
    for _, row in sub.iterrows():
        star = "*" if row["p_holm"] < 0.05 else ""
        lines.append(f"{str(row['bin']).replace(chr(10), ' ')}: p={row['p_holm']:.3f}{star}")
    if lines:
        ax.text(0.3, 0.83, "\n".join(lines), transform=ax.transAxes, fontsize=17,
                ha="left", va="bottom", color="0.1")
