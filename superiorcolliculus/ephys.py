"""Electrophysiology / LN-encoding-model tooling.

Folded in from the standalone neuro_analysis package (neuro_analysis/ephys/
analysis.py in the superior_colliculus repo) — LN/GLM model fitting for
spike-train encoding, CSD windowing, and the ablation/plotting helpers that
go with them. Domain-specific to ephys, unlike analysis.py/processing.py's
general kinematics and signal-processing utilities.

build_lag_matrix and raised_cosine_basis are NOT duplicated here — use
superiorcolliculus.processing.build_lag_matrix / raised_cosine_basis
directly; this module used to carry thin float32/log_warp=False wrappers
around them, which became pure redundancy once both lived in the same
package.
"""
import time
import tqdm.auto as tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, Lasso

import numpy as np


def compute_CSD(traces, events, sr, skips=[], CAR=True):
    """
    Docstring for compute_CSD

    :param traces: LFP data
    :param events: Times converted to lfp timebase
    :param sr: sampling rate of lfp
    :param skips: indices of events that should be ignored
    :param CAR: run common average reference

    returns: 3D array of shape (events, time, depth)
    """
    trace_l = []
    win = .200 * sr

    #
    for i, event in enumerate(events):

        ton, toff = event[0], event[1]
        # Get traces for this event
        win_traces = traces[int(ton - win):int(toff + win),:]

        # Mean-subtract each trace
        MS_traces = win_traces - np.mean(win_traces, axis=0, keepdims=True)

        # Common average reference
        if CAR:
            CAR_values = np.repeat(np.mean(MS_traces, axis=1, keepdims=True), 384, axis=1) # [:,np.newaxis]
            CAR_traces = MS_traces - CAR_values
            trace_l.append(CAR_traces)
        else:
            trace_l.append(MS_traces)


    # window on either side plus event duration
    box_width = int(win * 2 + np.median(np.diff(events)))

    trace_box = np.zeros(
        (
            len(trace_l),
            box_width,
            trace_l[0].shape[1])) * np.nan

    for i, trace in tqdm.tqdm(enumerate(trace_l), total=len(trace_l)):
        if len(trace) < box_width * .75:
            print(f"skipping event {i} with duration {len(trace)/sr:.3f}s")
            continue

        frames, channels = trace.shape
        trace_box[i,:len(trace),:] = trace[:box_width,:]

    return trace_box, win, win + np.median(np.diff(events))


# ── LN model: nonlinearities ─────────────────────────────────────────────────

def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def nl_sigmoid(eta, p):
    """Saturating rate model: R_max * sigmoid(k*(eta-c))  [Hz].

    p = [R_max, k, c].  R_max > 0, k > 0 (larger drive → higher rate).
    """
    R_max, k, c = p
    return np.clip(R_max * _sigmoid(k * (eta - c)), 1e-8, None)


def dnl_sigmoid(eta, p):
    R_max, k, c = p
    s = _sigmoid(k * (eta - c))
    return R_max * k * s * (1.0 - s)


def nl_poisson(eta, p):
    """Exponential rate model: exp(a*eta + b0)  [Hz].  p = [a, b0]."""
    a, b0 = p
    return np.exp(np.clip(a * eta + b0, -20, 15))


def dnl_poisson(eta, p):
    a, _ = p
    return a * nl_poisson(eta, p)


def loss_poisson_nll(y, lam):
    """Poisson negative log-likelihood: -mean(y*log(λ) - λ)."""
    lam = np.clip(lam, 1e-8, None)
    return -np.mean(y * np.log(lam) - lam)


# ── LN model: fitting ─────────────────────────────────────────────────────────

def irls_beta_update(X, y, beta, nl_type, nl_params, reg_type, lambda_reg):
    """One IRLS step: fix NL params, update beta via weighted Ridge or Lasso.

    Uses Poisson variance V = μ for both sigmoid and Poisson nonlinearities.
    """
    n = X.shape[0]
    eta = X @ beta

    if nl_type == 'sigmoid':
        mu  = nl_sigmoid(eta, nl_params)
        dmu = dnl_sigmoid(eta, nl_params)
    else:
        mu  = nl_poisson(eta, nl_params)
        dmu = dnl_poisson(eta, nl_params)
    V = mu + 1e-8

    dmu = np.where(np.abs(dmu) < 1e-12, np.sign(dmu + 1e-15) * 1e-12, dmu)
    W   = np.clip(dmu**2 / V, 0.0, 1e6)
    z_w = eta + (y - mu) / dmu

    sq = np.sqrt(W)
    Xw = X   * sq[:, None]
    yw = z_w * sq

    # Column-scale Xw so coordinate descent sees unit-variance features.
    # This dramatically improves Lasso convergence without changing the model:
    # beta_original = beta_scaled / col_scale.
    col_scale = np.sqrt((Xw ** 2).mean(0))
    col_scale[col_scale < 1e-10] = 1.0   # avoid divide-by-zero for dead columns
    Xw_s = Xw / col_scale

    if reg_type == 'ridge':
        clf = Ridge(alpha=lambda_reg * n, fit_intercept=False, solver='lsqr')
    else:
        clf = Lasso(alpha=lambda_reg, fit_intercept=False, max_iter=10000, tol=1e-4)

    clf.fit(Xw_s, yw)
    return clf.coef_ / col_scale


def fit_nl_params(eta, y, nl_type, nl_params_init):
    """Fix beta, optimise NL params via Nelder-Mead on projections s = X @ beta."""
    if nl_type == 'sigmoid':
        def obj(p):
            if p[0] <= 0 or p[1] <= 0:   # R_max > 0, k > 0
                return 1e10
            return loss_poisson_nll(y, nl_sigmoid(eta, p))
    else:
        def obj(p):
            return loss_poisson_nll(y, nl_poisson(eta, p))

    res = minimize(obj, nl_params_init.copy(), method='Nelder-Mead',
                   options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6})
    return res.x


def fit_ln_alternating(X_tr, y_tr, X_val, y_val,
                       reg_type='ridge', nl_type='sigmoid',
                       lambda_reg=0.05, max_iter=50, tol=1e-4,
                       verbose=True):
    """Alternating IRLS + Nelder-Mead optimisation of a 1-D LN model.

    Both sigmoid and Poisson nonlinearities target firing rate (Hz) under
    Poisson NLL.

    Beta is warm-started from a RIDGE fit (not unregularized OLS — see NOTE
    below), and for the Poisson NL, the initial (a, b0) are scaled to the
    warm-start eta's own range rather than fixed constants (see second NOTE).
    Both changes target a specific, confirmed divergence mode (see
    analysis/encoding/LN_ENCODING_METHODS.md, "Poisson NL divergence"): for
    higher-firing units, an unregularized OLS warm start on a ~200-column,
    collinear (lagged, circular sin/cos) design can produce a beta whose eta
    already saturates the Poisson NL's exponent clip on iteration 1. Once
    saturated, the NLL surface is completely flat in (a, b0) — Nelder-Mead
    has no gradient to follow and returns at its own initial guess — and the
    next IRLS iteration reproduces the identical (still-saturated) loss,
    which trivially satisfies the `abs(Δloss) < tol` stopping rule. The net
    effect: the fit "converges" after ~2 iterations with a constant
    (fully-saturated) prediction, silently producing a degenerate model.

    Returns
    -------
    dict with keys: name, reg, nl, beta, nl_params, pred_fn, loss_fn,
                    train_hist, val_hist, n_iter, runtime, n_nonzero.
    """
    n, p = X_tr.shape
    name = f'{reg_type.capitalize()}+{nl_type.capitalize()}'
    # NOTE (warm start): plain `np.linalg.lstsq` (unregularized OLS) is what
    # used to be here. On a collinear design it can return an oversized beta,
    # which is exactly what feeds the divergence described above. Ridge with
    # the run's own lambda_reg keeps the warm start in the same regularized
    # regime the IRLS loop will use anyway (irls_beta_update's first Ridge
    # call uses the identical `alpha=lambda_reg * n`), so this isn't a new
    # hyperparameter — just applying the existing one one step earlier.
    ridge_warm = Ridge(alpha=lambda_reg * n, fit_intercept=False, solver='lsqr')
    ridge_warm.fit(X_tr, y_tr)
    beta = ridge_warm.coef_

    eta_init  = X_tr @ beta
    eta_scale = float(np.std(eta_init))
    if eta_scale < 1e-8:
        eta_scale = 1.0

    if nl_type == 'sigmoid':
        R_max_init = float(np.percentile(y_tr, 95))
        nl_params  = np.array([R_max_init, 1.0, 0.0])
        pred_fn    = nl_sigmoid
    else:
        # NOTE (NL init): a fixed a_init=0.5 assumes eta already lives on an
        # O(1) scale. It doesn't in general — eta's scale depends on the
        # warm-start beta and the (already-standardized, but not unit-
        # variance-guaranteed) design matrix. Scaling a_init inversely to
        # eta's own std keeps a_init * eta at a modest O(1-3) spread
        # regardless of eta's raw scale, and b0_init is centered on the
        # observed mean rate rather than 0, so the very first Poisson NL
        # evaluation starts near the data's actual scale instead of an
        # arbitrary flat guess — the condition that let the NLL surface go
        # completely flat (see divergence description above) in the first
        # place.
        a_init  = 3.0 / eta_scale
        b0_init = float(np.log(max(float(np.mean(y_tr)), 1e-3)))
        nl_params = np.array([a_init, b0_init])
        pred_fn   = nl_poisson
    loss_fn = loss_poisson_nll

    train_hist, val_hist = [], []
    t0 = time.time()

    for i in range(max_iter):
        beta      = irls_beta_update(X_tr, y_tr, beta, nl_type, nl_params, reg_type, lambda_reg)
        eta_tr    = X_tr @ beta
        nl_params = fit_nl_params(eta_tr, y_tr, nl_type, nl_params)

        tr_loss  = loss_fn(y_tr,  pred_fn(eta_tr,       nl_params))
        val_loss = loss_fn(y_val, pred_fn(X_val @ beta, nl_params))
        train_hist.append(tr_loss)
        val_hist.append(val_loss)

        if verbose and (i % 10 == 0 or i == max_iter - 1):
            nnz = int(np.sum(np.abs(beta) > 1e-6))
            print(f'  [{name}] iter {i+1:3d}: '
                  f'train={tr_loss:.4f}  val={val_loss:.4f}  nnz={nnz}')

        if i > 0 and abs(train_hist[-2] - train_hist[-1]) < tol:
            if verbose:
                print(f'  [{name}] converged at iter {i+1}')
            break

    return {
        'name':       name,
        'reg':        reg_type,
        'nl':         nl_type,
        'beta':       beta,
        'nl_params':  nl_params,
        'pred_fn':    pred_fn,
        'loss_fn':    loss_fn,
        'train_hist': train_hist,
        'val_hist':   val_hist,
        'n_iter':     i + 1,
        'runtime':    round(time.time() - t0, 2),
        'n_nonzero':  int(np.sum(np.abs(beta) > 1e-6)),
    }


# ── LN model: visualisation ───────────────────────────────────────────────────

_DEFAULT_PALETTE = {
    'ridge+sigmoid': 'C0', 'lasso+sigmoid': 'C1',
    'ridge+poisson': 'C2', 'lasso+poisson': 'C3',
}


# Degenerate-fit detection (see fit_ln_alternating's docstring for the
# confirmed divergence mechanism this guards against: the Poisson NL
# saturates its exponent clip and the fit "converges" on a garbage,
# arbitrarily-scaled prediction). Two criteria, either one enough:
#   - DEGENERATE_PRED_STD: full saturation collapses pred_te to a literal
#     constant (std ~0) -- the case first found on SC83 unit 218.
#   - DEGENERATE_RATE_MULT: PARTIAL saturation (most, not all, test-fold
#     predictions pinned near the exp(15)~=3.27e6 Hz clip ceiling) still
#     varies sample-to-sample -- std alone misses it (confirmed on SC83 unit
#     478, fold 2: pred_te ranged 1.66e6-3.27e6 Hz, std=5.6e4, easily above
#     the std threshold, yet every value is ~15,000x the unit's own observed
#     max rate of ~214 Hz). Comparing the prediction's mean against a
#     multiple of the unit's own observed max rate scales with whatever's
#     physiologically plausible for that specific unit/dataset, rather than
#     hardcoding a fixed Hz ceiling that would need retuning per dataset.
# Shared by ablation_r2 (below) and
# ln_encoding_pipeline._ablation_refit/_fit_one_unit so every consumer (CSV
# export, per-unit PDF, poster figures) agrees on what counts as degenerate
# and NaNs it out the same way.
DEGENERATE_PRED_STD  = 1e-9
DEGENERATE_RATE_MULT = 20


def is_degenerate_fit(pred, y_ref):
    """True if `pred` looks like a saturated/divergent LN fit rather than a
    genuine prediction for `y_ref`'s scale (see DEGENERATE_PRED_STD /
    DEGENERATE_RATE_MULT above for the two criteria and why both are needed)."""
    if np.std(pred) < DEGENERATE_PRED_STD:
        return True
    ref_max = max(float(np.max(y_ref)), 1.0)
    return bool(np.mean(pred) > DEGENERATE_RATE_MULT * ref_max)


def _boxcar_smooth(y, w):
    """Causal uniform moving average; output same length as input."""
    return np.convolve(y, np.ones(w) / w, mode='full')[:len(y)]


def sincos_groups(predictors):
    """Detect sin_X / cos_X pairs in the predictor list.

    Returns
    -------
    display_names : list[str]  one entry per display predictor;
                    pairs collapse to their base name (e.g. 'Phi')
    groups        : list[list[int]]  indices into `predictors` per display predictor;
                    singletons are [[i]], pairs are [[i_sin, i_cos]]
    """
    pred_list = list(predictors)
    display_names, groups, used = [], [], set()
    for i, p in enumerate(pred_list):
        if i in used:
            continue
        if p.startswith("sin_"):
            base = p[4:]
            cos_name = "cos_" + base
            if cos_name in pred_list:
                j = pred_list.index(cos_name)
                display_names.append(base)
                groups.append([i, j])
                used.update({i, j})
                continue
        display_names.append(p)
        groups.append([i])
        used.add(i)
    return display_names, groups


def ablation_r2(fold_results, predictors, d_past=0, groups=None):
    """Per-predictor ablation R² across CV folds.

    ΔR² = R²_full − R²_ablated, where R²_ablated comes from a genuine refit of
    the whole LN model (beta + NL params) with the group's lag columns removed
    from the design matrix entirely — NOT from zeroing the full model's beta
    and reusing its NL params. The old zero-beta approach could drive
    R²_ablated arbitrarily negative (a fixed nonlinearity fit for a different
    design matrix, evaluated on a mismatched linear predictor), which makes
    ΔR² = R²_full − R²_ablated unbounded above and hard to interpret (values
    >1 are common and don't mean "more than 100% of variance"). Refitting lets
    the remaining predictors compensate, so ΔR² stays a meaningful measure of
    that group's marginal contribution.

    The actual refits happen in ln_encoding_pipeline._fit_one_unit (which has
    the raw train/val arrays and fit hyperparameters); this function only
    aggregates the per-fold results that call already attached to each fold
    dict as 'abl_r2_refit' (ndarray, one value per group, in `groups` order).

    Parameters
    ----------
    fold_results : dict  model_key → list of fold dicts
                   Each fold dict must have 'y_te', 'pred_te', and
                   'abl_r2_refit' (ndarray, shape (n_groups,)).
    predictors   : list[str]
    d_past       : int  if > 0, also compute R² on causal-smoothed traces
    groups       : list[list[int]] or None
                   Which predictor indices were ablated together per output
                   column. None → each predictor ablated individually
                   ([[0],[1],...]). Must match what produced 'abl_r2_refit'.

    Returns
    -------
    contrib    : dict  model_key → ndarray (n_folds, n_groups)  ΔR² per fold/group
    r2_full    : dict  model_key → ndarray (n_folds,)            full-model R² per fold
    r2_smooth  : dict  model_key → ndarray (n_folds,)            R² on smoothed traces
    """
    n_pred  = len(predictors)
    if groups is None:
        groups = [[i] for i in range(n_pred)]
    n_groups = len(groups)

    contrib   = {}
    r2_full   = {}
    r2_smooth = {}

    for key, folds in fold_results.items():
        c_arr  = np.zeros((len(folds), n_groups))
        r_arr  = np.zeros(len(folds))
        rs_arr = np.full(len(folds), np.nan)

        for fi, fold in enumerate(folds):
            pred_full = fold["pred_te"]
            y         = fold["y_te"]
            ss_tot    = np.sum((y - y.mean()) ** 2)
            if ss_tot == 0 or is_degenerate_fit(pred_full, y):
                # Degenerate full-model fit (NL saturated its clip, fully or
                # partially) -- r2_full and every ΔR² derived from it are
                # meaningless; NaN rather than an arbitrarily large
                # SS_res/SS_tot ratio (see is_degenerate_fit above).
                r_arr[fi] = np.nan
                c_arr[fi] = np.nan
                continue
            r2_f      = 1.0 - np.sum((y - pred_full) ** 2) / ss_tot
            r_arr[fi] = r2_f

            if d_past > 0:
                ys   = _boxcar_smooth(y, d_past)
                ps   = _boxcar_smooth(pred_full, d_past)
                ss_s = np.sum((ys - ys.mean()) ** 2)
                rs_arr[fi] = (1.0 - np.sum((ys - ps) ** 2) / ss_s) if ss_s > 0 else np.nan

            c_arr[fi] = fold["abl_r2_refit"]

        contrib[key]   = c_arr
        r2_full[key]   = r_arr
        r2_smooth[key] = rs_arr

    return contrib, r2_full, r2_smooth


def plot_ln_model_summary(results, X, rates_all, unit_idx, predictor_name,
                          d_past, d_future, fps=30, palette=None, figsize=(20, 7)):
    """2×4 summary: temporal filter (left) + nonlinearity curve+scatter (right) per model.

    Parameters
    ----------
    results       : dict  model_key → fit-result dict from fit_ln_alternating
    X             : (T, L) design matrix (full dataset, for η computation)
    rates_all     : (n_units, T) firing-rate array aligned to X
    unit_idx      : int   which row of rates_all to use for scatter
    predictor_name: str   used in the suptitle
    d_past        : int   causal lags
    d_future      : int   acausal lags
    fps           : int
    palette       : dict  model_key → colour (defaults to C0–C3)
    figsize       : tuple
    """
    if palette is None:
        palette = _DEFAULT_PALETTE

    t_lags = np.arange(-d_past + 1, d_future + 1) / fps
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    for idx, (key, res) in enumerate(results.items()):
        row, pair = divmod(idx, 2)
        ax_filt = axes[row, pair * 2]
        ax_nl   = axes[row, pair * 2 + 1]
        col     = palette.get(key, f'C{idx}')

        # Temporal filter
        ax_filt.plot(t_lags, res['beta'], color=col, lw=2)
        ax_filt.axhline(0, color='k', lw=0.5)
        ax_filt.axvline(0, color='gray', lw=0.8, ls='--')
        ax_filt.set_title(f'{res["name"]}  (nnz={res["n_nonzero"]})')
        ax_filt.set_xlabel('Lag (s)')
        ax_filt.set_ylabel('Coefficient')

        # Nonlinearity curve + observed scatter
        eta_all  = X @ res['beta']
        eta_grid = np.linspace(eta_all.min(), eta_all.max(), 300)
        ax_nl.plot(eta_grid, res['pred_fn'](eta_grid, res['nl_params']), color=col, lw=2)
        idx_s = np.random.choice(len(eta_all), size=min(500, len(eta_all)), replace=False)
        ax_nl.scatter(eta_all[idx_s], rates_all[unit_idx][idx_s], color=col, s=4, alpha=0.2)
        ax_nl.set_xlabel('η = Xβ')
        ax_nl.set_ylabel('Rate (Hz)')
        ax_nl.set_title(f'{res["name"]}  NL')

    plt.suptitle(
        f'Temporal filters & nonlinearities  [{predictor_name} → spike, unit {unit_idx}]',
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


def plot_unit_predictions(uid, fold_results, predictors, d_past, d_future,
                          fps=30, palette=None, null_abl_r2=None, pvals=None,
                          rate_ceiling=None):
    """Per-unit summary figure for the encoding pipeline PDF (3-fold CV).

    Three rows:
      Top    : observed firing rate vs stitched predictions across all CV folds
               (fold boundaries marked with vertical dashed lines).
      Middle : per-predictor temporal filters — thin lines per fold, thick mean.
      Bottom : fitted nonlinearity per model — thin curves per fold, thick mean.

    Parameters
    ----------
    uid          : unit ID (for title)
    fold_results : dict  model_key → list of per-fold dicts.
                   Each fold dict is the fit_ln_alternating result extended with
                   'y_te', 'X_te', 'pred_te', 'fold'.
    predictors   : list[str]  predictor names, length n_pred
    d_past       : int   causal lags
    d_future     : int   acausal lags (0 for causal-only)
    fps          : int
    palette      : dict  model_key → colour

    Returns
    -------
    fig : matplotlib Figure  (caller handles pdf.savefig + plt.close)
    """
    if palette is None:
        palette = _DEFAULT_PALETTE

    n_pred   = len(predictors)
    n_models = len(fold_results)
    # beta_full is the full-lag-resolution reconstruction (B @ coef_basis per
    # predictor) that ln_encoding_pipeline._fit_one_unit attaches whenever the
    # run used n_basis (raised-cosine kernel fitting) -- fall back to plain
    # "beta" for ordinary (non-basis) runs, where beta already IS full
    # resolution. Only affects display (this function); the nonlinearity/
    # ablation sections below correctly keep using "beta" paired with its own
    # "X_te", since those two must stay in whatever space (raw or basis) the
    # fit actually happened in.
    first_fold0 = next(iter(fold_results.values()))[0]
    first_beta  = first_fold0.get("beta_full", first_fold0["beta"])
    L        = first_beta.shape[-1] // n_pred
    if d_future > 0:
        t_lags = (d_past - np.arange(L)) / fps
    else:
        t_lags = np.arange(-d_past + 1, 1) / fps

    display_names, groups = sincos_groups(predictors)
    n_display = len(display_names)
    ncols     = max(n_display, 2 * n_models)

    contrib, r2_full, r2_smooth = ablation_r2(
        fold_results, predictors, d_past=d_past, groups=groups
    )

    first_folds = next(iter(fold_results.values()))
    n_folds     = len(first_folds)

    has_polar = (L > 1) and any(len(g) == 2 for g in groups)
    row_filter = n_folds
    if has_polar:
        row_polar  = n_folds + 1
        row_nl     = n_folds + 2
        row_ab     = n_folds + 3
        n_rows     = n_folds + 4
        height_ratios = [1.2] * n_folds + [1, 1, 1, 1.5]
    else:
        row_nl     = n_folds + 1
        row_ab     = n_folds + 2
        n_rows     = n_folds + 3
        height_ratios = [1.2] * n_folds + [1, 1, 1.5]

    fig_h = (n_folds * 2.5) + (10 if not has_polar else 12)
    fig = plt.figure(figsize=(max(14, 4 * ncols), fig_h))
    gs  = fig.add_gridspec(n_rows, ncols, height_ratios=height_ratios, hspace=0.65, wspace=0.45)

    # ── timeseries: one full-width row per CV fold ────────────────────────────
    from scipy.ndimage import gaussian_filter1d
    _sigma_ts = max(d_past, 1) / 2.355   # FWHM → σ
    for fi, fold_data in enumerate(first_folds):
        ax_ts = fig.add_subplot(gs[fi, :])
        y_f   = fold_data["y_te"]
        T_f   = len(y_f)
        t_f   = np.arange(T_f) / fps
        y_sm  = gaussian_filter1d(y_f.astype(np.float64), sigma=_sigma_ts)

        ax_ts.plot(t_f, y_f,  color='lightgray', lw=0.4, alpha=0.6)
        ax_ts.plot(t_f, y_sm, color='k',         lw=0.8,
                   label='observed (smoothed)' if fi == 0 else None)

        for key, folds in fold_results.items():
            col  = palette.get(key, 'C4')
            pred = folds[fi]["pred_te"]
            ax_ts.plot(t_f, pred, color=col, lw=0.7, alpha=0.8,
                       label=folds[0]["name"] if fi == 0 else None)

        # Cap y-axis at smoothed-signal 99th-pct to prevent sparse instantaneous
        # sub-bin spikes from collapsing the display (each fold sized independently).
        _preds_fi = [folds[fi]["pred_te"] for folds in fold_results.values()]
        _y_ceil = max(
            float(np.percentile(y_sm, 99)),
            *[float(np.percentile(p, 99)) for p in _preds_fi],
        ) * 1.3
        ax_ts.set_ylim(0, max(_y_ceil, 1.0))

        ax_ts.set_ylabel('Rate (Hz)', fontsize=8)
        ax_ts.set_title(f'Unit {uid}  fold {fi}', fontsize=9)
        if fi == n_folds - 1:
            ax_ts.set_xlabel('Time (s)', fontsize=8)
        else:
            ax_ts.set_xticklabels([])
        if fi == 0:
            ax_ts.legend(fontsize=8, loc='upper right')

    # ── temporal filters ──────────────────────────────────────────────────────
    filter_axes = [fig.add_subplot(gs[row_filter, j]) for j in range(n_display)]

    for key, folds in fold_results.items():
        col   = palette.get(key, 'C4')
        betas = np.stack([f.get("beta_full", f["beta"]) for f in folds])   # (n_folds, n_pred*L)
        b3d   = betas.reshape(n_folds, n_pred, L)       # (n_folds, n_pred, L)

        for j, (ax_f, dname, grp) in enumerate(zip(filter_axes, display_names, groups)):
            if L == 1:
                # smoothed-input mode: scalar weights — bar + per-fold dots
                if len(grp) == 2:
                    i_sin, i_cos = grp
                    w_sin = b3d[:, i_sin, 0]
                    w_cos = b3d[:, i_cos, 0]
                    ax_f.bar([-0.2], [w_sin.mean()], width=0.35, color=col, alpha=0.75, label='sin')
                    ax_f.bar([ 0.2], [w_cos.mean()], width=0.35, color=col, alpha=0.45, label='cos')
                    ax_f.scatter(np.full(len(w_sin), -0.2), w_sin, color=col, s=12, zorder=3)
                    ax_f.scatter(np.full(len(w_cos),  0.2), w_cos, color=col, s=12, zorder=3, alpha=0.6)
                    ax_f.legend(fontsize=6, loc='upper right', framealpha=0.5)
                    ax_f.set_xticks([])
                else:
                    i = grp[0]
                    weights = b3d[:, i, 0]
                    ax_f.bar([0], [weights.mean()], width=0.4, color=col, alpha=0.75)
                    ax_f.scatter(np.zeros(len(weights)), weights, color=col, s=12, zorder=3)
                    ax_f.set_xticks([])
                ax_f.axhline(0, color='k', lw=0.4)
                ax_f.set_xlabel('(smoothed)', fontsize=7)
            elif len(grp) == 2:
                # sin/cos pair → overlay both components (sin solid, cos dashed)
                i_sin, i_cos = grp
                for fold_beta in b3d[:, i_sin, :]:
                    ax_f.plot(t_lags, fold_beta, color=col, lw=0.5, alpha=0.35, ls='-')
                ax_f.plot(t_lags, b3d[:, i_sin, :].mean(0), color=col, lw=1.2, ls='-',  label='sin')
                for fold_beta in b3d[:, i_cos, :]:
                    ax_f.plot(t_lags, fold_beta, color=col, lw=0.5, alpha=0.35, ls='--')
                ax_f.plot(t_lags, b3d[:, i_cos, :].mean(0), color=col, lw=1.2, ls='--', label='cos')
                ax_f.legend(fontsize=6, loc='upper left', framealpha=0.5)
                ax_f.axhline(0, color='k', lw=0.4)
                ax_f.axvline(0, color='gray', lw=0.6, ls='--')
                ax_f.set_xlabel('Lag (s)', fontsize=8)
            else:
                # single predictor — temporal filter
                i = grp[0]
                for fold_beta in b3d[:, i, :]:
                    ax_f.plot(t_lags, fold_beta, color=col, lw=0.5, alpha=0.35)
                ax_f.plot(t_lags, b3d[:, i, :].mean(0), color=col, lw=1.2)
                ax_f.axhline(0, color='k', lw=0.4)
                ax_f.axvline(0, color='gray', lw=0.6, ls='--')
                ax_f.set_xlabel('Lag (s)', fontsize=8)
            if j == 0:
                ax_f.set_ylabel('Coef', fontsize=8)
            ax_f.set_title(dname, fontsize=9)

    # ── polar kernels (row 2, one subplot per sin/cos pair) ──────────────────
    # Each lag τ defines a vector (β_cos[τ], β_sin[τ]) in the plane.
    # Plot the trajectory from past (light) to present (dark) on polar axes.
    for j, (dname, grp) in enumerate(zip(display_names, groups)):
        if len(grp) != 2 or L == 1:
            continue   # skip scalar predictors and smoothed-input mode
        i_sin, i_cos = grp
        ax_p = fig.add_subplot(gs[row_polar, j], projection='polar')
        ax_p.set_title(dname, fontsize=8, pad=4)
        # lag color: past → light, present → dark
        lag_norm = np.linspace(0.2, 1.0, L)
        for key, folds in fold_results.items():
            col  = palette.get(key, 'C4')
            betas = np.stack([f.get("beta_full", f["beta"]) for f in folds])   # (n_folds, n_pred*L)
            b3d   = betas.reshape(n_folds, n_pred, L)
            b_sin_mean = b3d[:, i_sin, :].mean(0)          # (L,)
            b_cos_mean = b3d[:, i_cos, :].mean(0)
            angles_mean = np.arctan2(b_sin_mean, b_cos_mean)
            radii_mean  = np.sqrt(b_sin_mean ** 2 + b_cos_mean ** 2)
            # thin per-fold lines
            for fold_betas in b3d:
                ang = np.arctan2(fold_betas[i_sin], fold_betas[i_cos])
                rad = np.sqrt(fold_betas[i_sin] ** 2 + fold_betas[i_cos] ** 2)
                for li in range(L):
                    c = list(plt.cm.get_cmap('Greys')(lag_norm[li])[:3]) + [0.25]
                    ax_p.plot(ang[li], rad[li], 'o', color=col, ms=2, alpha=0.25)
            # mean trajectory: color-coded dots from past to present
            for li in range(L):
                alpha = lag_norm[li]
                ax_p.plot(angles_mean[li], radii_mean[li], 'o',
                          color=col, ms=4, alpha=alpha)
            # connect mean trajectory with thin line
            ax_p.plot(angles_mean, radii_mean, '-', color=col, lw=0.8, alpha=0.6)
        ax_p.set_yticklabels([])
        ax_p.tick_params(labelsize=6)

    # ── nonlinearity curves (one subplot per model spanning 2 cols) ─────────
    for m, (key, folds) in enumerate(fold_results.items()):
        ax_nl = fig.add_subplot(gs[row_nl, m * 2 : m * 2 + 2])
        col   = palette.get(key, 'C4')

        eta_all  = np.concatenate([f["X_te"] @ f["beta"] for f in folds])
        eta_lo   = np.percentile(eta_all, 1)
        eta_hi   = np.percentile(eta_all, 99)
        eta_grid = np.linspace(eta_lo, eta_hi, 200)

        all_curves = []
        for fold_res in folds:
            curve = fold_res["pred_fn"](eta_grid, fold_res["nl_params"])
            ax_nl.plot(eta_grid, curve, color=col, lw=0.5, alpha=0.35)
            all_curves.append(curve)

        mean_params = np.mean([f["nl_params"] for f in folds], axis=0)
        mean_curve  = folds[0]["pred_fn"](eta_grid, mean_params)
        ax_nl.plot(eta_grid, mean_curve, color=col, lw=1.2)

        y_ceil = np.percentile(np.concatenate(all_curves), 99) * 1.15
        ax_nl.set_xlim(eta_lo, eta_hi)
        ax_nl.set_ylim(0, y_ceil)
        ax_nl.set_xlabel('η = Xβ', fontsize=8)
        ax_nl.set_ylabel('Rate (Hz)', fontsize=8)
        nlp_str = "  ".join(f"{v:.2f}" for v in mean_params)
        ax_nl.set_title(f'{folds[0]["name"]} NL  [{nlp_str}]', fontsize=9)

    # ── ablation ΔR² bars (left) + GT vs predicted scatter (right) ──────────
    fold_colors = ['#4878CF', '#6ACC65', '#D65F5F']   # blue / green / red per fold
    bar_width   = 0.25
    for m, (key, folds) in enumerate(fold_results.items()):
        ax_ab = fig.add_subplot(gs[row_ab, m * 2])

        c_arr  = contrib[key]   # (n_folds, n_groups)
        r_arr  = r2_full[key]   # (n_folds,)

        n_folds_k = c_arr.shape[0]
        offsets   = (np.arange(n_folds_k) - (n_folds_k - 1) / 2) * bar_width
        x_pos     = np.arange(n_display)

        for fi in range(n_folds_k):
            fc = fold_colors[fi % len(fold_colors)]
            ax_ab.bar(x_pos + offsets[fi], c_arr[fi], width=bar_width,
                      color=fc, alpha=0.8, label=f'fold {fi}')
            # horizontal tick at this fold's total R²_full
            r2_fi = float(r_arr[fi])
            if not np.isnan(r2_fi):
                ax_ab.hlines(r2_fi,
                             x_pos[0] + offsets[fi] - bar_width * 0.4,
                             x_pos[-1] + offsets[fi] + bar_width * 0.4,
                             colors=fc, lw=1.2, ls='--', alpha=0.8)

        # null distribution stripplot + significance markers (first model only)
        if m == 0 and null_abl_r2 is not None:
            rng_jit = np.random.default_rng(42)
            for g_idx in range(c_arr.shape[1]):
                null_vals = null_abl_r2[:, g_idx]
                jitter    = rng_jit.uniform(-0.18, 0.18, size=len(null_vals))
                ax_ab.scatter(g_idx + jitter, null_vals,
                              s=4, color='gray', alpha=0.4, linewidths=0,
                              rasterized=True, zorder=0)
                if pvals is not None:
                    p      = float(pvals[g_idx])
                    marker = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                    if marker:
                        bar_top  = float(np.nanmax(c_arr[:, g_idx]))
                        null_top = float(null_vals.max())
                        y_top    = max(bar_top, null_top) * 1.1
                        ax_ab.text(g_idx, y_top, marker,
                                   ha='center', va='bottom', fontsize=9, color='k')

        ax_ab.axhline(0, color='k', lw=0.5)
        ax_ab.set_xticks(x_pos)
        ax_ab.set_xticklabels(display_names, fontsize=7, rotation=30, ha='right')
        ax_ab.set_ylabel('ΔR²', fontsize=8)
        ax_ab.legend(fontsize=6, loc='upper right', framealpha=0.5)
        ax_ab.set_title(folds[0]["name"], fontsize=9)

    # ── GT vs predicted scatter (right of each bar chart) ───────────────────
    for m, (key, folds) in enumerate(fold_results.items()):
        ax_sc = fig.add_subplot(gs[row_ab, m * 2 + 1])

        all_pred = np.concatenate([f["pred_te"] for f in folds])
        all_obs  = np.concatenate([f["y_te"]    for f in folds])
        lim_lo   = min(float(all_obs.min()), float(all_pred.min()))
        if rate_ceiling is not None:
            lim_hi = float(rate_ceiling)
        else:
            lim_hi = max(float(all_obs.max()), float(all_pred.max()))

        for fi, fold in enumerate(folds):
            fc   = fold_colors[fi % len(fold_colors)]
            y_te = fold["y_te"]
            p_te = fold["pred_te"]

            m_fit, b_fit = np.polyfit(p_te, y_te, 1)
            ss_res = np.sum((y_te - (m_fit * p_te + b_fit)) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2_fit = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

            ax_sc.scatter(p_te, y_te, color=fc, s=3, alpha=0.3, linewidths=0, rasterized=True)
            x_line = np.array([p_te.min(), p_te.max()])
            ax_sc.plot(x_line, m_fit * x_line + b_fit, color=fc, lw=1.2,
                       label=f'f{fi}: m={m_fit:.2f} b={b_fit:.2f} R²={r2_fit:.3f}')

        ax_sc.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color='k', lw=0.6, ls='--', alpha=0.4)
        ax_sc.set_xlim(lim_lo, lim_hi)
        ax_sc.set_ylim(lim_lo, lim_hi)
        ax_sc.set_xlabel('Predicted (Hz)', fontsize=8)
        ax_sc.set_ylabel('Observed (Hz)', fontsize=8)
        ax_sc.legend(fontsize=6, loc='upper left', framealpha=0.6)
        ax_sc.set_title(folds[0]["name"], fontsize=9)

    fig.suptitle(f'Unit {uid}  |  predictors: {", ".join(predictors)}', fontsize=11)
    return fig


# ── VAR(1) population model ───────────────────────────────────────────────────

def fit_var1(X_pop, U_beh, folds, lambda_reg=0.1, sequential=False):
    """Fit VAR(1): X[t+1] = A X[t] + B u(t) via multivariate Ridge.

    Parameters
    ----------
    X_pop      : (T, n_units)  smoothed population firing rates (Hz)
    U_beh      : (T, n_beh)   behavioral predictors, aligned to X_pop
    folds      : list of (idx_tr, idx_val, idx_te) index arrays
    lambda_reg : Ridge regularisation strength
    sequential : bool
        If True, fit B first (u(t) → x(t+1)), then fit A on the behavioral
        residuals r(t) = x(t+1) − B u(t).  B captures maximal behavioral
        variance before A is introduced.

    Returns
    -------
    A     : (n_units, n_units)  mean connectivity matrix across folds
    B     : (n_units, n_beh)   mean input matrix across folds
    r2_te : (n_units, n_folds) test R² per unit per fold
    """
    Xp  = X_pop[:-1].astype(np.float32)   # x(t)
    Up  = U_beh[:-1].astype(np.float32)   # u(t)
    Y   = X_pop[1:].astype(np.float32)    # x(t+1)
    n_units = X_pop.shape[1]

    AB_folds, r2_folds = [], []
    for idx_tr, idx_val, idx_te in folds:
        idx_tv = np.concatenate([idx_tr, idx_val])
        if sequential:
            clf_b = Ridge(alpha=lambda_reg, fit_intercept=False, solver='svd')
            clf_b.fit(Up[idx_tv], Y[idx_tv])
            B_f = clf_b.coef_                                    # (n_units, n_beh)
            R   = (Y - Up @ B_f.T).astype(np.float32)           # behavioral residuals
            clf_a = Ridge(alpha=lambda_reg, fit_intercept=False, solver='svd')
            clf_a.fit(Xp[idx_tv], R[idx_tv])
            A_f = clf_a.coef_                                    # (n_units, n_units)
            AB_folds.append(np.hstack([A_f, B_f]))
            y_hat = clf_a.predict(Xp[idx_te]) + clf_b.predict(Up[idx_te])
        else:
            Z   = np.hstack([Xp, Up])
            clf = Ridge(alpha=lambda_reg, fit_intercept=False, solver='svd')
            clf.fit(Z[idx_tv], Y[idx_tv])
            AB_folds.append(clf.coef_)
            y_hat = clf.predict(Z[idx_te])
        ss_res = ((Y[idx_te] - y_hat) ** 2).sum(0)
        ss_tot = ((Y[idx_te] - Y[idx_te].mean(0)) ** 2).sum(0) + 1e-12
        r2_folds.append(1.0 - ss_res / ss_tot)

    AB = np.stack(AB_folds).mean(0)
    return AB[:, :n_units], AB[:, n_units:], np.stack(r2_folds).T


def fit_static_var(X_pop, U_beh, folds, lambda_reg=0.1, sequential=False,
                   nl_type=None, nl_lambda=0.05, nl_max_iter=50, nl_tol=1e-4,
                   return_per_fold=False):
    """Fit static model: x_i(t) = Σ_{j≠i} A_ij x_j(t) + B_i u(t) via Ridge.

    Diagonal of A is zero by construction (leave-one-out per unit).
    Each unit is fit independently so there is no simultaneity bias.

    Parameters
    ----------
    X_pop           : (T, n_units)  smoothed population firing rates (Hz)
    U_beh           : (T, n_beh)   behavioral predictors, aligned to X_pop
    folds           : list of (idx_tr, idx_val, idx_te) index arrays
    lambda_reg      : Ridge regularisation strength for Stage 2 (peer residuals)
    sequential      : bool
        If True, fit B first on behavior alone, compute residuals, then fit A.
    nl_type         : {None, 'sigmoid', 'poisson'}
        When sequential=True, nonlinearity for Stage 1 (per-unit LN fit via
        fit_ln_alternating). None falls back to linear Ridge for Stage 1.
    nl_lambda       : regularisation strength for Stage 1 NL fit
    nl_max_iter     : max alternating iterations for Stage 1 NL fit
    nl_tol          : convergence tolerance for Stage 1 NL fit
    return_per_fold : bool
        If True, also return a list of per-fold dicts for per-unit PDF generation.
        Keys: 'stage1_fits' (list of fit_ln_alternating results, or None),
              'B_f' (linear B, or None when nl_type set),
              'A_fold', 'R', 'idx_te', 'nl_type'.

    Returns
    -------
    A        : (n_units, n_units)  fold-mean connectivity matrix, zero diagonal
    B        : (n_units, n_beh)   fold-mean linear Stage-1 betas
    r2_te    : (n_units, n_folds) full-model test R² per unit per fold
    fold_data: list of dicts (only when return_per_fold=True)
    """
    T, n_units = X_pop.shape
    n_beh      = U_beh.shape[1]
    all_cols   = np.arange(n_units)
    use_nl     = sequential and (nl_type is not None)

    AB_folds, r2_folds, fold_data = [], [], []
    for idx_tr, idx_val, idx_te in folds:
        idx_tv      = np.concatenate([idx_tr, idx_val])
        AB_fold     = np.zeros((n_units, n_units + n_beh), dtype=np.float32)
        r2_fold     = np.zeros(n_units, dtype=np.float32)
        R           = np.zeros_like(X_pop)
        B_f         = None
        stage1_fits = None

        if sequential:
            if use_nl:
                # Stage 1: per-unit LN fit on behavioral predictors
                stage1_fits = []
                U_tr  = U_beh[idx_tr].astype(np.float32)
                U_val = U_beh[idx_val].astype(np.float32)
                for i in range(n_units):
                    y_i = X_pop[:, i].astype(np.float32)
                    res = fit_ln_alternating(
                        U_tr, y_i[idx_tr], U_val, y_i[idx_val],
                        reg_type='ridge', nl_type=nl_type,
                        lambda_reg=nl_lambda, max_iter=nl_max_iter,
                        tol=nl_tol, verbose=False,
                    )
                    b_pred_all = res['pred_fn'](
                        U_beh.astype(np.float32) @ res['beta'], res['nl_params']
                    )
                    R[:, i]            = y_i - b_pred_all
                    AB_fold[i, n_units:] = res['beta']
                    stage1_fits.append(res)
            else:
                # Stage 1: linear Ridge on behavior (all units at once)
                clf_b = Ridge(alpha=lambda_reg, fit_intercept=False, solver='svd')
                clf_b.fit(U_beh[idx_tv], X_pop[idx_tv])
                B_f = clf_b.coef_.astype(np.float32)
                R   = (X_pop - U_beh @ B_f.T).astype(np.float32)

        for i in range(n_units):
            pop_cols = np.delete(all_cols, i)
            y = X_pop[:, i].astype(np.float32)
            if sequential:
                # Stage 2: fit A_i on peer residuals → unit i's residual
                Z_r = R[:, pop_cols].astype(np.float32)
                clf = Ridge(alpha=lambda_reg, fit_intercept=False, solver='svd')
                clf.fit(Z_r[idx_tv], R[idx_tv, i])
                AB_fold[i, pop_cols] = clf.coef_
                if not use_nl:
                    AB_fold[i, n_units:] = B_f[i]
                # Full test prediction: NL(or linear) B prediction + A·r
                if use_nl:
                    res_i   = stage1_fits[i]
                    b_te    = res_i['pred_fn'](
                        U_beh[idx_te].astype(np.float32) @ res_i['beta'],
                        res_i['nl_params']
                    )
                else:
                    b_te = U_beh[idx_te] @ B_f[i]
                y_hat = clf.predict(Z_r[idx_te]) + b_te
            else:
                Z = np.hstack([X_pop[:, pop_cols], U_beh]).astype(np.float32)
                clf = Ridge(alpha=lambda_reg, fit_intercept=False, solver='svd')
                clf.fit(Z[idx_tv], y[idx_tv])
                AB_fold[i, pop_cols]   = clf.coef_[:n_units - 1]
                AB_fold[i, n_units:]   = clf.coef_[n_units - 1:]
                y_hat = clf.predict(Z[idx_te])
            ss_res = np.sum((y[idx_te] - y_hat) ** 2)
            ss_tot = np.sum((y[idx_te] - y[idx_te].mean()) ** 2) + 1e-12
            r2_fold[i] = 1.0 - ss_res / ss_tot

        AB_folds.append(AB_fold)
        r2_folds.append(r2_fold)
        fold_data.append({
            'stage1_fits': stage1_fits, 'B_f': B_f,
            'A_fold': AB_fold[:, :n_units], 'R': R,
            'idx_te': idx_te, 'nl_type': nl_type,
        })

    AB = np.stack(AB_folds).mean(0)
    result = (AB[:, :n_units], AB[:, n_units:], np.stack(r2_folds).T)
    return result + (fold_data,) if return_per_fold else result
