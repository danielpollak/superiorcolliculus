"""guidance_law_fitting.py

Guidance law regression — OLS (Model I) or RMA (Model II).

Four guidance laws are supported:
  PN  (Parallel Navigation)       : thetadot ~ alphadot, delay column = tau1
  iPN (Integral PN)               : thetadot ~ alpha,    delay column = tau1
  PP  (Proportional Pursuit)      : thetadot ~ Phi,      delay column = tau
  dPP (Derivative PP / CATD)      : thetadot ~ Phidot,   delay column = tau

Typical usage
-------------
>>> from superiorcolliculus.behavior import guidance_law_fitting as glf
>>> fitter = glf.GuidanceLawFitter("OLS")   # or "RMA"
>>> reg_df = fitter.run_discrete_delay_analysis(
...     tautau_df, mode="tangent", predictor="alphadot",
...     taus=[0, 3], tau_col="tau1",
... )

Low-level RMA primitives (``type2_regression``, ``t2_fit``,
``t2_bilinear_fit``, ``t2_mixed_fit``) are module-level implementation
helpers used internally by ``GuidanceLawFitter``; they are not part of the
public API.
"""

import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from pylr2.regress2 import regress2
from scipy.odr import ODR, Model, RealData
from superiorcolliculus import processing
from superiorcolliculus.analysis import compute_vaf, compute_weighted_r2


# ---------------------------------------------------------------------------
# Low-level RMA primitives (implementation helpers — not public API)
# ---------------------------------------------------------------------------

def type2_regression(x: np.ndarray, y: np.ndarray):
    """Reduced Major Axis (Type II) regression without intercept.

    Returns
    -------
    slope : float
    r2    : float — Pearson r², rounded to 3 decimal places
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = regress2(x, y, _need_intercept=False)
    r2    = np.round(res["r"] ** 2, 3)
    slope = np.round(res["slope"], 2)
    return slope, r2


def _weighted_zscore(v, w):
    """Z-score v using weights w. Returns (v_z, weighted_mean, weighted_std)."""
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    w_norm = w / (w.sum() + 1e-12)
    mu  = float((w_norm * v).sum())
    std = float(np.sqrt((w_norm * (v - mu) ** 2).sum()))
    std = std if std > 1e-12 else 1.0
    return (v - mu) / std, mu, std


def t2_fit(x, y, w=None):
    """Model II regression via ODRPACK, single predictor, no intercept.

    Returns
    -------
    slope    : float
    slope_se : float
    r2       : float — in-sample weighted R²
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones(len(x))
    w = np.asarray(w, dtype=float)
    eps = 1e-12

    x_z, _, std_x = _weighted_zscore(x, w)
    y_z, _, std_y = _weighted_zscore(y, w)

    sd_y = 1.0 / (np.sqrt(w) + eps)
    sd_x = sd_y * (std_y / (std_x + eps))

    model = Model(lambda B, xi: B[0] * xi)
    data  = RealData(x_z, y_z, sx=sd_x, sy=sd_y)
    out   = ODR(data, model, beta0=[1.0]).run()

    slope_z    = float(out.beta[0])
    slope_se_z = float(out.sd_beta[0]) if out.sd_beta is not None else float("nan")
    slope      = slope_z    * std_y / (std_x + eps)
    slope_se   = slope_se_z * std_y / (std_x + eps)

    y_hat    = slope_z * x_z
    wsum     = w.sum()
    mean_w_y = float((w * y_z).sum() / wsum)
    ss_res   = float((w * (y_z - y_hat) ** 2).sum())
    ss_tot   = float((w * (y_z - mean_w_y) ** 2).sum())
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return slope, slope_se, r2


def t2_bilinear_fit(x, s, y, w=None):
    """Model II bilinear fit: θ̇ = (β₀ + β₁·s)·x via ODRPACK, no intercept.

    Returns
    -------
    predict_fn  : callable(x_te, s_te) → y_hat in original units
    slope_at_fn : callable(s_val) → effective gain β(s_val) in original units
    r2          : float — in-sample weighted R²
    """
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones(len(x))
    w  = np.asarray(w, dtype=float)
    x2 = s * x
    eps = 1e-12

    x1_z, mu_x1, std_x1 = _weighted_zscore(x,  w)
    x2_z, mu_x2, std_x2 = _weighted_zscore(x2, w)
    y_z,  mu_y,  std_y  = _weighted_zscore(y,  w)

    sd_y  = 1.0 / (np.sqrt(w) + eps)
    sd_x1 = sd_y * (std_y / (std_x1 + eps))
    sd_x2 = sd_y * (std_y / (std_x2 + eps))

    model = Model(lambda B, xv: B[0] * xv[0] + B[1] * xv[1])
    data  = RealData(
        np.vstack([x1_z, x2_z]), y_z,
        sx=np.vstack([sd_x1, sd_x2]), sy=sd_y,
    )
    out = ODR(data, model, beta0=[0.5, 0.5]).run()
    B0, B1 = float(out.beta[0]), float(out.beta[1])

    y_hat_z  = B0 * x1_z + B1 * x2_z
    wsum     = w.sum()
    mean_w_y = float((w * y_z).sum() / wsum)
    ss_res   = float((w * (y_z - y_hat_z) ** 2).sum())
    ss_tot   = float((w * (y_z - mean_w_y) ** 2).sum())
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def predict_fn(x_te, s_te):
        x_te  = np.asarray(x_te, dtype=float)
        s_te  = np.asarray(s_te, dtype=float)
        x1_z_ = (x_te - mu_x1) / std_x1
        x2_z_ = (s_te * x_te - mu_x2) / std_x2
        return (B0 * x1_z_ + B1 * x2_z_) * std_y + mu_y

    def slope_at_fn(s_val):
        return (B0 / std_x1 + B1 * s_val / std_x2) * std_y

    return predict_fn, slope_at_fn, r2


def t2_mixed_fit(x1, x2, y, w=None):
    """Model II fit for θ̇ = β₀·x1 + β₁·x2 via ODRPACK, no intercept.

    Returns
    -------
    coef       : ndarray shape (2,) — slopes in original units [β₀, β₁]
    predict_fn : callable(x1_te, x2_te) → y_hat in original units
    r2         : float — in-sample weighted R²
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y  = np.asarray(y,  dtype=float)
    if w is None:
        w = np.ones(len(x1))
    w = np.asarray(w, dtype=float)
    eps = 1e-12

    x1_z, mu_x1, std_x1 = _weighted_zscore(x1, w)
    x2_z, mu_x2, std_x2 = _weighted_zscore(x2, w)
    y_z,  mu_y,  std_y  = _weighted_zscore(y,  w)

    sd_y  = 1.0 / (np.sqrt(w) + eps)
    sd_x1 = sd_y * (std_y / (std_x1 + eps))
    sd_x2 = sd_y * (std_y / (std_x2 + eps))

    model = Model(lambda B, xv: B[0] * xv[0] + B[1] * xv[1])
    data  = RealData(
        np.vstack([x1_z, x2_z]), y_z,
        sx=np.vstack([sd_x1, sd_x2]), sy=sd_y,
    )
    out = ODR(data, model, beta0=[0.5, 0.5]).run()
    B0, B1 = float(out.beta[0]), float(out.beta[1])

    coef = np.array([B0 * std_y / (std_x1 + eps), B1 * std_y / (std_x2 + eps)])

    y_hat_z  = B0 * x1_z + B1 * x2_z
    wsum     = w.sum()
    mean_w_y = float((w * y_z).sum() / wsum)
    ss_res   = float((w * (y_z - y_hat_z) ** 2).sum())
    ss_tot   = float((w * (y_z - mean_w_y) ** 2).sum())
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def predict_fn(x1_te, x2_te):
        x1_te = np.asarray(x1_te, dtype=float)
        x2_te = np.asarray(x2_te, dtype=float)
        x1_z_ = (x1_te - mu_x1) / std_x1
        x2_z_ = (x2_te - mu_x2) / std_x2
        return (B0 * x1_z_ + B1 * x2_z_) * std_y + mu_y

    return coef, predict_fn, r2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GuidanceLawFitter:
    """Guidance law regression with selectable regression model.

    Parameters
    ----------
    model : {"OLS", "RMA"}
        "OLS" (default) — ordinary least squares; correct for VAF / prediction.
        "RMA" — reduced major axis (Type II); correct for unbiased parameter
                estimation when both predictor and response are noisy.
    """

    def __init__(self, model: str = "OLS"):
        if model not in ("OLS", "RMA"):
            raise ValueError(f"model must be 'OLS' or 'RMA', got {model!r}")
        self.model = model

    # ------------------------------------------------------------------
    # Internal dispatch primitives
    # ------------------------------------------------------------------

    def _fit_slope(self, x, y, w=None):
        """Single-predictor slope, no intercept.

        Returns
        -------
        slope    : float — in original units
        slope_se : float — standard error (nan for OLS)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.model == "OLS":
            X = x.reshape(-1, 1)
            if w is not None:
                sw = np.sqrt(np.asarray(w, dtype=float))
                X = X * sw[:, None]
                y = y * sw
            slope = float(np.linalg.lstsq(X, y, rcond=None)[0][0])
            return slope, float("nan")
        else:
            slope, slope_se, _ = t2_fit(x, y, w)
            return slope, slope_se

    def _fit_bilinear(self, x, s, y, w=None):
        """Bilinear fit: θ̇ = (β₀ + β₁·s)·x, no intercept.

        Returns
        -------
        predict_fn  : callable(x_te, s_te) → y_hat
        slope_at_fn : callable(s_val) → effective gain at speed s_val
        r2          : float
        """
        x = np.asarray(x, dtype=float)
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.model == "OLS":
            X = np.column_stack([x, s * x])
            y_fit = y.copy()
            if w is not None:
                sw = np.sqrt(np.asarray(w, dtype=float))
                X = X * sw[:, None]
                y_fit = y_fit * sw
            coef = np.linalg.lstsq(X, y_fit, rcond=None)[0]
            b0, b1 = float(coef[0]), float(coef[1])
            y_hat = b0 * x + b1 * s * x
            r2 = 1.0 - np.var(y - y_hat) / np.var(y) if np.var(y) > 0 else float("nan")

            def predict_fn(x_te, s_te):
                return b0 * np.asarray(x_te) + b1 * np.asarray(s_te) * np.asarray(x_te)

            def slope_at_fn(s_val):
                return b0 + b1 * float(s_val)

            return predict_fn, slope_at_fn, r2
        else:
            return t2_bilinear_fit(x, s, y, w)

    # ------------------------------------------------------------------
    # Per-bin fit
    # ------------------------------------------------------------------

    def fit_guidance_law(
        self,
        df_bin: pd.DataFrame,
        predictor: str,
        response: str = "thetadot",
        weights: np.ndarray | None = None,
    ) -> dict:
        """Fit a guidance law model on a pre-filtered DataFrame.

        Parameters
        ----------
        df_bin    : rows for one (tau, mode, subject, speed_bin) cell
        predictor : column name — 'alphadot' (PN), 'Phi' (PP), 'Phidot' (CATD)
        response  : column name, default 'thetadot'
        weights   : optional per-sample weights (e.g. v_forward); passed to
                    _fit_slope for WLS.  Negative weights are clamped to zero.

        Returns
        -------
        dict with keys: slope, r2, vaf, n
            r2 is Pearson r² (consistent across OLS and RMA).
        """
        x = df_bin[predictor].values
        y = df_bin[response].values
        w = np.clip(weights, 0, None) if weights is not None else None
        slope, _ = self._fit_slope(x, y, w)
        vaf = compute_vaf(x * slope, y)
        r2  = float(np.corrcoef(x, y)[0, 1] ** 2) if len(x) > 1 else float("nan")
        r2_w = compute_weighted_r2(x, y, w) if w is not None else float("nan")
        return {"slope": slope, "r2": r2, "r2_weighted": r2_w, "vaf": vaf, "n": len(x)}

    def fit_mixed_guidance_law(
        self,
        df_bin: pd.DataFrame,
        predictors: list[str],
        response: str = "thetadot",
        weights: np.ndarray | None = None,
    ) -> dict:
        """Fit a multi-predictor guidance law via OLS/WLS (no intercept).

        Parameters
        ----------
        df_bin     : rows for one (tau, mode, subject, dist_bin) cell
        predictors : list of column names, e.g. ['alphadot', 'Phi']
        response   : column name, default 'thetadot'
        weights    : optional per-sample weights; negatives clamped to zero.

        Returns
        -------
        dict with keys: coef, predict_fn, vaf, n
            coef       : ndarray of shape (len(predictors),)
            predict_fn : callable(df) → y_hat, accepts a DataFrame with predictor columns
        """
        if self.model != "OLS":
            raise NotImplementedError("fit_mixed_guidance_law only supports OLS; use t2_mixed_fit for RMA.")
        X = df_bin[predictors].values
        y = df_bin[response].values
        w = np.clip(weights, 0, None) if weights is not None else None
        X_fit, y_fit = X.copy(), y.copy()
        if w is not None:
            sw = np.sqrt(w)
            X_fit = X_fit * sw[:, None]
            y_fit = y_fit * sw
        coef = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
        y_hat = X @ coef
        vaf = compute_vaf(y_hat, y)

        def predict_fn(df):
            return df[predictors].values @ coef

        return {"coef": coef, "predict_fn": predict_fn, "vaf": vaf, "n": len(y)}

    # ------------------------------------------------------------------
    # Full analyses
    # ------------------------------------------------------------------

    def run_discrete_delay_analysis(
        self,
        df: pd.DataFrame,
        mode: str,
        predictor: str,
        taus: list,
        tau_col: str,
        gl_label: str = None,
        response: str = "thetadot",
        variable: str = "v_forward",
        interval: float = 5.0,
        min_count: int = 20,
        v_min: float = 0.0,
        v_max: float = 50.0,
        n_folds: int = 5,
    ) -> pd.DataFrame:
        """Speed-binned, multi-tau regression for one mode × predictor.

        Parameters
        ----------
        df        : tautau_df (function re-filters internally)
        mode      : 'tangent', 'BD', or 'HD'
        predictor : 'alphadot' or 'Phi'
        taus      : delay values (frames) to include, e.g. [0, 3]
        tau_col   : delay column being varied ('tau1' for PN, 'tau' for PP)
        gl_label  : guidance law label in results; inferred from predictor if None
        response  : target column, default 'thetadot'
        variable  : binning column, default 'v_forward'
        interval  : bin width
        min_count : minimum rows per bin to include it
        v_min/max : speed range
        n_folds   : k for k-fold cross-validation over bouts

        Returns
        -------
        DataFrame — one row per (tau, subject, fold, speed_bin) with columns:
            r2, VAF, bilinear_VAF, m, GL, tau (ms), tau_frames,
            subject, fold, mode, i, pos_RC, std
        """
        if gl_label is None:
            gl_label = "PN" if predictor == "alphadot" else "PP"

        fixed_col = "tau" if tau_col == "tau1" else "tau1"
        df_mode   = df.loc[(df["mode"] == mode) & (df[fixed_col] == 0)].copy()
        df_mode   = df_mode.loc[df_mode[tau_col].isin(taus)]

        bout_cols = [c for c in ["avi", "event ind"] if c in df_mode.columns]

        rows   = []
        groups = df_mode.groupby([tau_col, "subject"])
        for (tau, subject), tt_df in tqdm.tqdm(groups, total=groups.ngroups,
                                                desc=f"{mode} {gl_label}"):
            tt_df = tt_df.copy().reset_index(drop=True)

            if bout_cols:
                tt_df["_bout_id"] = tt_df.groupby(bout_cols).ngroup()
            else:
                tt_df["_bout_id"] = 0

            bout_ids = tt_df["_bout_id"].unique()
            n_bouts  = len(bout_ids)
            if n_bouts < 2:
                continue

            valid_bins = []
            for lo in np.arange(v_min, v_max, interval):
                hi = lo + interval
                if ((tt_df[variable] > lo) & (tt_df[variable] <= hi)).sum() >= min_count:
                    valid_bins.append((lo, hi))
                else:
                    break
            if not valid_bins:
                continue

            k    = min(n_folds, n_bouts)
            rng  = np.random.default_rng(42)
            perm = rng.permutation(n_bouts)
            fold_assignments = np.array_split(perm, k)

            for fold_idx, test_idx in enumerate(fold_assignments):
                test_bout_ids = set(bout_ids[i] for i in test_idx)
                is_test       = tt_df["_bout_id"].isin(test_bout_ids)

                train_df = tt_df[~is_test]
                test_df  = tt_df[is_test]
                if len(train_df) < 2 or len(test_df) < 2:
                    continue

                fit_train  = self.fit_guidance_law(train_df, predictor=predictor, response=response)
                slope      = fit_train["slope"]

                predict_bl, _, _ = self._fit_bilinear(
                    train_df[predictor].values,
                    train_df[variable].values,
                    train_df[response].values,
                )

                test_df = test_df.copy()
                test_df["_y_hat"]    = slope * test_df[predictor].values
                test_df["_y_hat_bl"] = predict_bl(test_df[predictor].values, test_df[variable].values)

                for lo, hi in valid_bins:
                    in_bin = (test_df[variable] > lo) & (test_df[variable] <= hi)
                    if in_bin.sum() < 2:
                        continue

                    t_df   = test_df[in_bin]
                    vaf    = compute_vaf(t_df["_y_hat"].values,    t_df[response].values)
                    vaf_bl = compute_vaf(t_df["_y_hat_bl"].values, t_df[response].values)
                    pos_rc = np.median(t_df["pos_RC"].values) / processing.conversion_rate

                    rows.append({
                        "r2":           fit_train["r2"],
                        "VAF":          vaf,
                        "bilinear_VAF": vaf_bl,
                        "m":            slope,
                        "GL":           gl_label,
                        "tau (ms)":     round(tau * 1000 / 30),
                        "tau_frames":   tau,
                        "subject":      subject,
                        "fold":         fold_idx,
                        "mode":         mode,
                        "i":            f"{int(lo)} - {int(hi)}",
                        "pos_RC":       pos_rc,
                        "std":          np.std(t_df[response].values),
                    })

        return pd.DataFrame(rows)

    def run_pd_controller_analysis(
        self,
        df: pd.DataFrame,
        mode: str,
        tau1_values: list = None,
        tau2_values: list = None,
        n_folds: int = 5,
        min_count: int = 20,
        variable: str = "speed",
        interval: float = 5.0,
        v_max: float = 50.0,
        response: str = "thetadot",
    ) -> pd.DataFrame:
        """Grid search over (τ₁, τ₂) for PD controller: θ̇ = A·Φ(t−τ₁) + B·dΦ/dt(t−τ₂).

        Always uses OLS (two-predictor lstsq) regardless of self.model.

        Returns
        -------
        DataFrame — one row per (tau1, tau2, subject, fold, speed_bin) with columns:
            tau1_frames, tau2_frames, tau1_ms, tau2_ms,
            A, B, cond, VAF, subject, fold, mode, i, pos_RC
        """
        if tau1_values is None:
            tau1_values = list(range(0, 8))
        if tau2_values is None:
            tau2_values = list(range(0, 8))

        df_base = df.loc[
            (df["mode"] == mode) & (df["tau"] == 0) & (df["tau1"] == 0)
        ].copy().reset_index(drop=True)

        bout_cols = [c for c in ["avi", "event ind"] if c in df_base.columns]

        rows = []
        grid = [(t1, t2) for t1 in tau1_values for t2 in tau2_values]

        for tau1, tau2 in tqdm.tqdm(grid, desc=f"{mode} PD grid"):
            df_shifted = df_base.copy()

            if tau1 > 0:
                df_shifted["_phi"] = df_base.groupby(bout_cols)["Phi"].shift(tau1)
            else:
                df_shifted["_phi"] = df_base["Phi"]

            if tau2 > 0:
                df_shifted["_phidot"] = df_base.groupby(bout_cols)["Phidot"].shift(tau2)
            else:
                df_shifted["_phidot"] = df_base["Phidot"]

            df_shifted = df_shifted.dropna(subset=["_phi", "_phidot"])

            for subject, subj_df in df_shifted.groupby("subject"):
                subj_df = subj_df.copy().reset_index(drop=True)

                if bout_cols:
                    subj_df["_bout_id"] = subj_df.groupby(bout_cols).ngroup()
                else:
                    subj_df["_bout_id"] = 0

                bout_ids = subj_df["_bout_id"].unique()
                n_bouts  = len(bout_ids)
                if n_bouts < 2:
                    continue

                valid_bins = []
                for hi in range(int(interval), int(v_max) + 1, int(interval)):
                    lo = hi - interval
                    if ((subj_df[variable] > lo) & (subj_df[variable] <= hi)).sum() >= min_count:
                        valid_bins.append((lo, hi))
                    else:
                        break
                if not valid_bins:
                    continue

                k    = min(n_folds, n_bouts)
                rng  = np.random.default_rng(42)
                perm = rng.permutation(n_bouts)
                fold_assignments = np.array_split(perm, k)

                for fold_idx, test_idx in enumerate(fold_assignments):
                    test_bout_ids = set(bout_ids[i] for i in test_idx)
                    is_test       = subj_df["_bout_id"].isin(test_bout_ids)

                    train_df = subj_df[~is_test]
                    test_df  = subj_df[is_test]
                    if len(train_df) < 2 or len(test_df) < 2:
                        continue

                    X_train = np.column_stack([
                        train_df["_phi"].values,
                        train_df["_phidot"].values,
                    ])
                    y_train = train_df[response].values

                    coef, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
                    A_coef, B_coef = float(coef[0]), float(coef[1])
                    cond = float(np.linalg.cond(X_train.T @ X_train))

                    X_test = np.column_stack([
                        test_df["_phi"].values,
                        test_df["_phidot"].values,
                    ])
                    test_df = test_df.copy()
                    test_df["_y_hat"] = X_test @ coef

                    for lo, hi in valid_bins:
                        in_bin = (test_df[variable] > lo) & (test_df[variable] <= hi)
                        if in_bin.sum() < 2:
                            continue

                        t_df   = test_df[in_bin]
                        vaf    = compute_vaf(t_df["_y_hat"].values, t_df[response].values)
                        pos_rc = np.median(t_df["pos_RC"].values) / processing.conversion_rate

                        rows.append({
                            "tau1_frames": tau1,
                            "tau2_frames": tau2,
                            "tau1_ms":     round(tau1 * 1000 / 30),
                            "tau2_ms":     round(tau2 * 1000 / 30),
                            "A":           A_coef,
                            "B":           B_coef,
                            "cond":        cond,
                            "VAF":         vaf,
                            "subject":     subject,
                            "fold":        fold_idx,
                            "mode":        mode,
                            "i":           f"{int(lo)}-{int(hi)}",
                            "pos_RC":      pos_rc,
                        })

        return pd.DataFrame(rows)

    def compute_per_bin_vaf(
        self,
        tautau_df: pd.DataFrame,
        modes: list,
        taus: list,
        n_folds: int = 5,
        interval: float = 5.0,
        v_min: float = 0.0,
        v_max: float = 50.0,
        min_count: int = 20,
        variable: str = "v_forward",
    ) -> pd.DataFrame:
        """Fit a separate slope per speed bin and return cross-validated VAF.

        Contrast with ``run_discrete_delay_analysis``, which fits a *single*
        pooled slope on all training frames and evaluates it per bin.  Here
        each bin gets its own free slope, fitted only on training frames in
        that bin.

        Returns
        -------
        DataFrame — one row per (mode, GL, tau, subject, fold, speed_bin) with
        columns: GL, tau (ms), subject, fold, mode, i, VAF, model='per-bin'.
        """
        rows = []
        configs = [
            ("alphadot", "tau1", "PN"),
            ("Phi",      "tau",  "PP"),
            ("Phidot",   "tau",  "CATD"),
        ]

        for mode in modes:
            for predictor, tau_col, gl_label in configs:
                fixed_col = "tau" if tau_col == "tau1" else "tau1"
                df_m = tautau_df.loc[
                    (tautau_df["mode"] == mode) &
                    (tautau_df[fixed_col] == 0) &
                    tautau_df[tau_col].isin(taus)
                ].copy()

                bout_cols = [c for c in ("avi", "event ind") if c in df_m.columns]

                for (tau, subj), tt in df_m.groupby([tau_col, "subject"]):
                    tt = tt.copy().reset_index(drop=True)
                    tt["_bid"] = tt.groupby(bout_cols).ngroup() if bout_cols else 0
                    bids = tt["_bid"].unique()
                    if len(bids) < 2:
                        continue

                    valid_bins = []
                    for lo in np.arange(v_min, v_max, interval):
                        hi = lo + interval
                        if ((tt[variable] > lo) & (tt[variable] <= hi)).sum() >= min_count:
                            valid_bins.append((lo, hi))
                        else:
                            break
                    if not valid_bins:
                        continue

                    k    = min(n_folds, len(bids))
                    perm = np.random.default_rng(42).permutation(len(bids))
                    for fold_idx, test_idx in enumerate(np.array_split(perm, k)):
                        test_bids = set(bids[i] for i in test_idx)
                        is_test   = tt["_bid"].isin(test_bids)
                        tr, te    = tt[~is_test], tt[is_test]
                        if len(tr) < 2 or len(te) < 2:
                            continue

                        for lo, hi in valid_bins:
                            tr_b = tr[(tr[variable] > lo) & (tr[variable] <= hi)]
                            te_b = te[(te[variable] > lo) & (te[variable] <= hi)]
                            if len(tr_b) < min_count or len(te_b) < 2:
                                continue

                            slope = self.fit_guidance_law(tr_b, predictor=predictor)["slope"]
                            vaf   = compute_vaf(
                                slope * te_b[predictor].values,
                                te_b["thetadot"].values,
                            )
                            rows.append({
                                "GL":       gl_label,
                                "tau (ms)": round(tau * 1000 / 30),
                                "subject":  subj,
                                "fold":     fold_idx,
                                "mode":     mode,
                                "i":        f"{int(lo)} - {int(hi)}",
                                "VAF":      vaf,
                                "model":    "per-bin",
                            })

        return pd.DataFrame(rows)

    def compute_vaf_2d(
        self,
        df_pn: pd.DataFrame,
        df_pp: pd.DataFrame,
        d_edges: np.ndarray,
        d_labels: list,
        v_min: float = 0.0,
        v_max: float = 40.0,
        v_interval: float = 10.0,
        min_cell: int = 20,
        k_folds: int = 5,
        mode: str = None,
        tau_ms: int = None,
        rng_seed: int = 42,
    ) -> pd.DataFrame:
        """Speed × distance binned VAF via bout-level k-fold CV for PN and PP.

        Parameters
        ----------
        df_pn    : rows at canonical PN tau (e.g. tau==0, tau1==tau_val)
        df_pp    : rows at canonical PP tau (e.g. tau==tau_val, tau1==0)
        d_edges  : (n_bins+1,) distance bin edges — pass percentiles from real data
                   so simulated and real data share identical bin boundaries.
        d_labels : n_bins label strings, e.g. ["near", "mid", "far"]
        mode     : label stored in output DataFrame (e.g. "tangent")
        tau_ms   : delay in ms stored in output DataFrame

        Returns
        -------
        DataFrame with columns: mode, tau_ms, subject, fold, v_bin, d_bin, GL, VAF, gain
        """
        bout_cols = [c for c in ("avi", "event ind") if c in df_pn.columns]
        rows = []

        for subj in df_pn["subject"].unique():
            tt_pn = df_pn[df_pn["subject"] == subj].copy().reset_index(drop=True)
            tt_pp = df_pp[df_pp["subject"] == subj].copy().reset_index(drop=True)
            if len(tt_pn) < 10:
                continue
            for tt in [tt_pn, tt_pp]:
                if len(tt) > 0 and bout_cols:
                    tt["_bid"] = tt.groupby(bout_cols).ngroup()

            bids = tt_pn["_bid"].unique()
            if len(bids) < 2:
                continue
            k    = min(k_folds, len(bids))
            perm = np.random.default_rng(rng_seed).permutation(len(bids))

            for fi, ti in enumerate(np.array_split(perm, k)):
                test_bids = set(bids[i] for i in ti)
                tr_pn = tt_pn[~tt_pn["_bid"].isin(test_bids)]
                te_pn = tt_pn[ tt_pn["_bid"].isin(test_bids)].copy()
                tr_pp = tt_pp[~tt_pp["_bid"].isin(test_bids)]
                te_pp = tt_pp[ tt_pp["_bid"].isin(test_bids)].copy()
                if any(len(x) < 2 for x in [tr_pn, te_pn, tr_pp, te_pp]):
                    continue

                m_pn = self.fit_guidance_law(tr_pn, "alphadot")["slope"]
                m_pp = self.fit_guidance_law(tr_pp, "Phi")["slope"]
                te_pn["_yhat"] = te_pn["alphadot"].values * m_pn
                te_pp["_yhat"] = te_pp["Phi"].values      * m_pp

                for gl, te, m_val in [("PN", te_pn, m_pn), ("PP", te_pp, m_pp)]:
                    for lo in np.arange(v_min, v_max, v_interval):
                        hi    = lo + v_interval
                        vmask = (te["v_forward"] > lo) & (te["v_forward"] <= hi)
                        if vmask.sum() < min_cell:
                            continue
                        te_v = te[vmask]
                        for di, dl in enumerate(d_labels):
                            dmask = ((te_v["dist"] >= d_edges[di]) &
                                     (te_v["dist"] <= d_edges[di + 1]))
                            if dmask.sum() < min_cell:
                                continue
                            te_vd = te_v[dmask]
                            rows.append({
                                "mode":    mode,
                                "tau_ms":  tau_ms,
                                "subject": subj,
                                "fold":    fi,
                                "v_bin":   f"{int(lo)} - {int(hi)}",
                                "d_bin":   dl,
                                "GL":      gl,
                                "VAF":     compute_vaf(te_vd["_yhat"].values,
                                                       te_vd["thetadot"].values),
                                "gain":    m_val,
                            })

        return pd.DataFrame(rows)

    def run_all_guidance_laws(
        self,
        tautau_df: pd.DataFrame,
        modes: list = None,
        taus: list = None,
        kwargs: dict = None,
    ) -> pd.DataFrame:
        """Convenience wrapper: run PN + PP + CATD for all modes.

        Returns
        -------
        Combined DataFrame with a 'GL, speed (cm/s)' label column.
        """
        if modes is None:
            modes = ["tangent", "BD", "HD"]
        if taus is None:
            taus = [0, 3]
        if kwargs is None:
            kwargs = {}

        parts = []
        for mode in modes:
            parts.append(
                self.run_discrete_delay_analysis(
                    tautau_df, mode=mode, predictor="alphadot",
                    taus=taus, tau_col="tau1", **kwargs,
                )
            )
            parts.append(
                self.run_discrete_delay_analysis(
                    tautau_df, mode=mode, predictor="Phi",
                    taus=taus, tau_col="tau", **kwargs,
                )
            )
            parts.append(
                self.run_discrete_delay_analysis(
                    tautau_df, mode=mode, predictor="Phidot",
                    taus=taus, tau_col="tau", gl_label="CATD", **kwargs,
                )
            )

        reg_df = pd.concat(parts, ignore_index=True)
        reg_df["GL, speed (cm/s)"] = reg_df["GL"] + ", " + reg_df["i"]
        return reg_df
