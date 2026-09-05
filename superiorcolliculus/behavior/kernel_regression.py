"""kernel_regression.py

Behavioral kernel regression: regress thetadot against a causal, acausal, or
symmetric lag design matrix built from a scalar behavioral predictor (alphadot,
Phi, Phidot, or alpha).  This is the behavioral analog of regression_trainer.py, which uses
the same lag-matrix approach for spike data → behavior decoding.

Instead of testing one discrete delay at a time (as in SfN_2025_poster.ipynb),
the full temporal integration kernel is fit simultaneously with Ridge regression.
The causal/acausal comparison tests whether thetadot leads or lags the predictor.

Design matrix conventions (matching superiorcolliculus.processing.build_lag_matrix for N=1):
  causal    X[t, c] = signal[t - c]         c ∈ [0, L-1]   shape (T, L)
  acausal   X[t, c] = signal[t + c]         c ∈ [0, L-1]   shape (T, L)
  symmetric X[t, c] = signal[t - (c - L)]   c ∈ [0, 2L]    shape (T, 2L+1)
              c=0 → L frames future, c=L → current, c=2L → L frames past

Checkpoint format (parallel to regression_trainer.py):
  { "coef": (n_lags,), "intercept": float,
    "config": {"L": int, "n_lags": int, "mode": str, "alpha": float,
               "predictor": str, "behavior_mode": str, "speed_bin": str},
    "r2_train": float, "r2_test": float,
    "causal_r2": float, "acausal_r2": float, "symmetric_r2": float,
    "peak_lag_frames": int, "peak_lag_ms": float }

Typical usage
-------------
>>> import superiorcolliculus.behavior.kernel_regression as kr
>>> results = kr.run_kernel_analysis_by_mode_speed(
...     df=bout_df, mode="tangent", predictor="alphadot",
...     L=20, alpha=1.0, output_dir="data/behavior",
... )
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from sklearn.linear_model import Lasso, Ridge

from superiorcolliculus.analysis import compute_vaf, r2_score
from superiorcolliculus.processing import build_lag_matrix as _shared_build_lag_matrix

# No default data path here (this module is part of the installed
# superiorcolliculus package, not the main repo, so it has no built-in
# notion of any one repo's checkout location) -- output_dir is a required
# argument on run_kernel_analysis_by_mode_speed/run_all_kernel_analyses below.
SAMPLING_RATE = 30.0                       # Hz
DT_MS         = 1000.0 / SAMPLING_RATE     # 33.33 ms per frame

SCAN_ALPHAS = [0.01, 1.0, 100.0]
SCAN_L      = [3, 5, 7, 12]


# ---------------------------------------------------------------------------
# Lag matrix
# ---------------------------------------------------------------------------

def build_behavioral_lag_matrix(
    signal: np.ndarray,
    L: int,
    mode: str = "causal",
):
    """Build a causal, acausal, or symmetric lag matrix from a 1-D signal.

    Scalar analog of superiorcolliculus.processing.build_lag_matrix (which
    also takes a (N, T) spike matrix). Here signal has shape (T,), so the
    output has no neuron dimension.

    Boundary conditions: zero-padding (same as build_lag_matrix).

    Parameters
    ----------
    signal : 1-D array, shape (T,)
    L      : number of lags (half-width for symmetric mode)
    mode   : 'causal' | 'acausal' | 'symmetric'

    Returns
    -------
    X      : float64 array
               causal/acausal → shape (T, L)
               symmetric      → shape (T, 2L+1)
    n_lags : int — L for causal/acausal, 2L+1 for symmetric

    Thin wrapper around superiorcolliculus.processing.build_lag_matrix
    (dtype pinned to float64, 1-D input, to preserve prior behavior).
    """
    signal = np.asarray(signal, dtype=np.float64)
    return _shared_build_lag_matrix(signal, L, mode=mode, dtype=np.float64)


# ---------------------------------------------------------------------------
# Single-bout fit
# ---------------------------------------------------------------------------

def fit_behavioral_kernel(
    x_signal: np.ndarray,
    y_signal: np.ndarray,
    L: int,
    alpha: float,
    mode: str = "causal",
    train_fraction: float = 0.8,
    weights: np.ndarray = None,
    model_cls=Ridge,
) -> dict:
    """Fit a Ridge regression kernel: y_signal ~ X @ coef + intercept.

    Parameters
    ----------
    x_signal       : predictor time series, shape (T,)
    y_signal       : response time series, shape (T,)
    L              : number of lags
    alpha          : Ridge regularization strength
    mode           : 'causal' | 'acausal' | 'symmetric'
    train_fraction : fraction of frames used for training (first portion)
    weights        : optional sample weights shape (T,) applied to the
                     training split.  Test evaluation is always unweighted.
                     Typical usage: weights = speed / speed.mean()

    Returns
    -------
    dict with keys:
        coef       : (n_lags,) temporal integration kernel
        intercept  : float
        r2_train, r2_test, VAF_train, VAF_test : floats
        config     : dict; includes ``"weighted": True`` when weights provided
    Returns None if there are fewer than 10 test frames after splitting.
    """
    x_signal = np.asarray(x_signal, dtype=np.float64)
    y_signal = np.asarray(y_signal, dtype=np.float64)

    X, n_lags = build_behavioral_lag_matrix(x_signal, L, mode=mode)

    split = int(len(y_signal) * train_fraction)
    if len(y_signal) - split < 10:
        return None

    X_train, y_train = X[:split], y_signal[:split]
    X_test,  y_test  = X[split:], y_signal[split:]

    w_train = np.asarray(weights, dtype=np.float64)[:split] if weights is not None else None

    model = model_cls(alpha=alpha)
    model.fit(X_train, y_train, sample_weight=w_train)

    cfg = {"L": L, "n_lags": n_lags, "mode": mode, "alpha": alpha, "train_fraction": train_fraction}
    if weights is not None:
        cfg["weighted"] = True

    return {
        "coef":      model.coef_,
        "intercept": float(model.intercept_),
        "r2_train":  r2_score(y_train, model.predict(X_train)),
        "r2_test":   r2_score(y_test,  model.predict(X_test)),
        "VAF_train": compute_vaf(model.predict(X_train), y_train),
        "VAF_test":  compute_vaf(model.predict(X_test),  y_test),
        "sparsity":  float(np.mean(model.coef_ == 0)),
        "config":    cfg,
    }


def fit_speed_modulated_kernel(
    x_signal: np.ndarray,
    y_signal: np.ndarray,
    speed: np.ndarray,
    L: int,
    alpha: float,
    mode: str = "causal",
    train_fraction: float = 0.8,
    model_cls=Ridge,
) -> dict:
    """Bilinear speed-modulated kernel: K(s) = K0 + s_z * K1.

    Speed is z-scored before building the augmented design matrix so that both
    blocks of columns (X and speed_z * X) are on the same scale.  This is
    critical for Ridge regularization to treat K0 and K1 fairly; without it the
    speed-modulated block is 20-30x larger and Ridge under-penalises K1, leading
    to severe overfitting.

    The model is:  y_hat = (K0 + s_z * K1) @ x_lags + intercept
                         = [X, speed_z * X] @ [K0; K1] + intercept
    where  s_z = (s - speed_mean) / speed_std

    K0 is therefore the kernel at the mean speed; K1 is the change per one
    standard deviation of speed.

    Use kernel_at_speed(result, s) to evaluate K(s) in original speed units.

    Parameters
    ----------
    x_signal, y_signal : predictor and response time series, shape (T,)
    speed              : speed time series in cm/s, shape (T,)
    L, alpha, mode, train_fraction : same as fit_behavioral_kernel

    Returns
    -------
    dict with keys:
        K0              : (n_lags,) kernel at mean speed
        K1              : (n_lags,) change per s.d. of speed
        intercept       : float
        r2_train, r2_test, VAF_train, VAF_test : floats
        config          : dict with speed_mean, speed_std, and "bilinear": True
    Returns None if there are fewer than 10 test frames.
    """
    x_signal = np.asarray(x_signal, dtype=np.float64)
    y_signal = np.asarray(y_signal, dtype=np.float64)
    speed    = np.asarray(speed,    dtype=np.float64)

    speed_mean = speed.mean()
    speed_std  = speed.std()
    if speed_std == 0:
        speed_std = 1.0
    speed_z = (speed - speed_mean) / speed_std

    X, n_lags = build_behavioral_lag_matrix(x_signal, L, mode=mode)
    X_aug = np.hstack([X, speed_z[:, None] * X])    # shape (T, 2*n_lags)

    split = int(len(y_signal) * train_fraction)
    if len(y_signal) - split < 10:
        return None

    X_train, y_train = X_aug[:split], y_signal[:split]
    X_test,  y_test  = X_aug[split:], y_signal[split:]

    model = model_cls(alpha=alpha)
    model.fit(X_train, y_train)

    coef = model.coef_
    K0   = coef[:n_lags]
    K1   = coef[n_lags:]

    return {
        "K0":        K0,
        "K1":        K1,
        "intercept": float(model.intercept_),
        "r2_train":  r2_score(y_train, model.predict(X_train)),
        "r2_test":   r2_score(y_test,  model.predict(X_test)),
        "VAF_train": compute_vaf(model.predict(X_train), y_train),
        "VAF_test":  compute_vaf(model.predict(X_test),  y_test),
        "config": {
            "L": L, "n_lags": n_lags, "mode": mode,
            "alpha": alpha, "train_fraction": train_fraction,
            "bilinear": True,
            "speed_mean": float(speed_mean),
            "speed_std":  float(speed_std),
        },
    }


# ---------------------------------------------------------------------------
# Leave-one-bout-out (LOO) kernel fitting
# ---------------------------------------------------------------------------

def _collect_loo_data(
    bout_df: pd.DataFrame,
    target_avi,
    target_event_ind,
    subject: str,
    mode: str,
    predictor: str,
    response: str,
    speed_col: str | None,
    min_test_frames: int = 10,
    min_train_bouts: int = 2,
) -> dict | None:
    """Collect raw training and test arrays for LOO kernel fitting.

    Filters bout_df to the same subject + mode, excluding the target bout.
    Training data are kept as a list of per-bout raw arrays so that lag
    matrices can be built independently for each bout — this prevents
    boundary artifacts from leaking across bout edges when vstacking.

    Parameters
    ----------
    bout_df           : DataFrame containing all bouts
    target_avi        : avi identifier of the held-out bout
    target_event_ind  : event_ind of the held-out bout
    subject           : subject filter (matched against 'subject' column if present)
    mode              : bout mode filter (matched against 'mode' column)
    predictor         : column name for predictor signal
    response          : column name for response signal
    speed_col         : column name for speed, or None if not needed
    min_test_frames   : minimum frames required in the test bout
    min_train_bouts   : minimum number of training bouts required

    Returns
    -------
    dict with keys:
        train_bouts : list of dicts, each with 'x', 'y', and (if speed_col)
                      'speed' arrays for one training bout
        x_test      : (T_test,) predictor array for the held-out bout
        y_test      : (T_test,) response array for the held-out bout
        speed_test  : (T_test,) speed array, or None
    Returns None if there are fewer than min_train_bouts training bouts or
    fewer than min_test_frames frames in the test bout.
    """
    mask = bout_df["mode"] == mode
    if "subject" in bout_df.columns:
        mask &= bout_df["subject"] == subject

    target_mask = (bout_df["avi"] == target_avi) & (bout_df["event ind"] == target_event_ind)
    train_df = bout_df.loc[mask & ~target_mask]
    test_df  = bout_df.loc[mask &  target_mask]

    if len(test_df) < min_test_frames:
        return None

    bout_id_cols = [c for c in ("avi", "event ind") if c in train_df.columns]

    train_bouts = []
    groups = train_df.groupby(bout_id_cols) if bout_id_cols else [("all", train_df)]
    for _, grp in groups:
        entry = {
            "x": grp[predictor].values.astype(np.float64),
            "y": grp[response].values.astype(np.float64),
        }
        if speed_col is not None:
            entry["speed"] = grp[speed_col].values.astype(np.float64)
        train_bouts.append(entry)

    if len(train_bouts) < min_train_bouts:
        return None

    return {
        "train_bouts": train_bouts,
        "x_test":      test_df[predictor].values.astype(np.float64),
        "y_test":      test_df[response].values.astype(np.float64),
        "speed_test":  test_df[speed_col].values.astype(np.float64) if speed_col is not None else None,
    }


def fit_behavioral_kernel_loo(
    bout_df: pd.DataFrame,
    target_avi,
    target_event_ind,
    subject: str,
    mode: str,
    predictor: str,
    response: str = "thetadot",
    L: int = 20,
    alpha: float = 100.0,
    kernel_mode: str = "symmetric",
    speed_col: str = None,
    model_cls=Ridge,
) -> dict | None:
    """LOO plain kernel fit: train on all same-subject/mode bouts except the target.

    The lag matrix is built independently for each training bout before
    vstacking, so no signal leaks across bout boundaries.

    Parameters
    ----------
    bout_df           : DataFrame containing all bouts (will be filtered internally)
    target_avi        : avi identifier of the held-out bout
    target_event_ind  : event_ind of the held-out bout
    subject           : subject used to restrict training bouts
    mode              : bout mode used to restrict training bouts
    predictor         : column name for predictor signal
    response          : column name for response signal
    L                 : kernel half-width in frames
    alpha             : Ridge regularisation strength
    kernel_mode       : 'causal' | 'acausal' | 'symmetric'
    speed_col         : optional column name for speed.  When provided, training
                        frames are weighted by speed / mean(speed), biasing the
                        kernel toward fast frames.  Test evaluation is unweighted.

    Returns
    -------
    dict with the same keys as fit_behavioral_kernel, plus:
        n_train_bouts : int — number of bouts used for training
        config        : includes ``"loo": True`` and ``"weighted": True`` if
                        speed_col was provided
    Returns None if insufficient training data (see _collect_loo_data).
    """
    data = _collect_loo_data(
        bout_df, target_avi, target_event_ind, subject, mode,
        predictor, response, speed_col=speed_col,
    )
    if data is None:
        return None

    # Minimum raw frames needed to build a lag matrix without empty slices
    min_frames = 2 * L + 1

    # Test bout too short for this L → skip
    if len(data["x_test"]) < min_frames:
        return None

    X_list, y_list, w_list = [], [], []
    n_lags = None
    for bout in data["train_bouts"]:
        if len(bout["x"]) < min_frames:
            continue   # too short to build lag matrix at this L
        X_b, n_lags = build_behavioral_lag_matrix(bout["x"], L, mode=kernel_mode)
        X_list.append(X_b)
        y_list.append(bout["y"])
        if speed_col is not None:
            w_list.append(bout["speed"])

    if len(X_list) < 2:
        return None   # fewer than 2 valid training bouts remain

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    w_train = None
    if speed_col is not None:
        w_all   = np.concatenate(w_list)
        mean_w  = w_all.mean() or 1.0
        w_train = w_all / mean_w

    model = model_cls(alpha=alpha)
    model.fit(X_train, y_train, sample_weight=w_train)

    X_test, _ = build_behavioral_lag_matrix(data["x_test"], L, mode=kernel_mode)
    y_test     = data["y_test"]

    y_hat_train = model.predict(X_train)
    y_hat_test  = model.predict(X_test)

    return {
        "coef":          model.coef_,
        "intercept":     float(model.intercept_),
        "r2_train":      r2_score(y_train, y_hat_train),
        "r2_test":       r2_score(y_test,  y_hat_test),
        "VAF_train":     compute_vaf(y_hat_train, y_train),
        "VAF_test":      compute_vaf(y_hat_test,  y_test),
        "sparsity":      float(np.mean(model.coef_ == 0)),
        "n_train_bouts": len(X_list),
        "config": {
            "L": L, "n_lags": n_lags, "mode": kernel_mode,
            "alpha": alpha, "loo": True,
            **( {"weighted": True} if speed_col is not None else {} ),
        },
    }


def fit_speed_modulated_kernel_loo(
    bout_df: pd.DataFrame,
    target_avi,
    target_event_ind,
    subject: str,
    mode: str,
    predictor: str,
    response: str = "thetadot",
    speed_col: str = "speed",
    L: int = 20,
    alpha: float = 100.0,
    kernel_mode: str = "symmetric",
    speed_weighted: bool = False,
    model_cls=Ridge,
) -> dict | None:
    """LOO bilinear speed-modulated kernel fit.

    Speed is z-scored using statistics from the training bouts only; the same
    mean and std are then applied to the held-out test bout.  This mirrors
    fit_speed_modulated_kernel but with cross-bout generalisation instead of
    an in-bout temporal split.

    Parameters
    ----------
    bout_df           : DataFrame containing all bouts
    target_avi        : avi identifier of the held-out bout
    target_event_ind  : event_ind of the held-out bout
    subject           : subject used to restrict training bouts
    mode              : bout mode used to restrict training bouts
    predictor         : column name for predictor signal
    response          : column name for response signal
    speed_col         : column name for speed (cm/s)
    L                 : kernel half-width in frames
    alpha             : Ridge regularisation strength
    kernel_mode       : 'causal' | 'acausal' | 'symmetric'
    speed_weighted    : if True, training frames are additionally weighted by
                        speed / mean(speed) in the Ridge loss — on top of speed
                        already acting as a structural feature via K1.

    Returns
    -------
    dict with the same keys as fit_speed_modulated_kernel, plus:
        n_train_bouts : int
        config        : includes ``"loo": True`` and ``"weighted": True`` if
                        speed_weighted is True
    Returns None if insufficient training data (see _collect_loo_data).
    """
    data = _collect_loo_data(
        bout_df, target_avi, target_event_ind, subject, mode,
        predictor, response, speed_col=speed_col,
    )
    if data is None:
        return None

    min_frames = 2 * L + 1

    if len(data["x_test"]) < min_frames:
        return None

    # Speed normalisation from training frames only (long bouts only)
    valid_bouts = [b for b in data["train_bouts"] if len(b["x"]) >= min_frames]
    if len(valid_bouts) < 2:
        return None

    all_train_speed = np.concatenate([b["speed"] for b in valid_bouts])
    speed_mean = float(all_train_speed.mean())
    speed_std  = float(all_train_speed.std()) or 1.0

    X_list, y_list = [], []
    n_lags = None
    for bout in valid_bouts:
        X_b, n_lags = build_behavioral_lag_matrix(bout["x"], L, mode=kernel_mode)
        spd_z = (bout["speed"] - speed_mean) / speed_std
        X_list.append(np.hstack([X_b, spd_z[:, None] * X_b]))
        y_list.append(bout["y"])

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    w_train = None
    if speed_weighted:
        mean_w  = all_train_speed.mean() or 1.0
        w_train = all_train_speed / mean_w

    model = model_cls(alpha=alpha)
    model.fit(X_train, y_train, sample_weight=w_train)

    X_test_base, _ = build_behavioral_lag_matrix(data["x_test"], L, mode=kernel_mode)
    spd_z_test = (data["speed_test"] - speed_mean) / speed_std
    X_test     = np.hstack([X_test_base, spd_z_test[:, None] * X_test_base])
    y_test     = data["y_test"]

    y_hat_train = model.predict(X_train)
    y_hat_test  = model.predict(X_test)

    coef = model.coef_
    K0   = coef[:n_lags]
    K1   = coef[n_lags:]

    return {
        "K0":          K0,
        "K1":          K1,
        "intercept":   float(model.intercept_),
        "r2_train":    r2_score(y_train, y_hat_train),
        "r2_test":     r2_score(y_test,  y_hat_test),
        "VAF_train":   compute_vaf(y_hat_train, y_train),
        "VAF_test":    compute_vaf(y_hat_test,  y_test),
        "n_train_bouts": len(data["train_bouts"]),
        "config": {
            "L": L, "n_lags": n_lags, "mode": kernel_mode,
            "alpha": alpha, "loo": True, "bilinear": True,
            "speed_mean": speed_mean, "speed_std": speed_std,
            **( {"weighted": True} if speed_weighted else {} ),
        },
    }


# ---------------------------------------------------------------------------
# Mixed (two-predictor) kernel fitting
# ---------------------------------------------------------------------------

def fit_mixed_kernel_loo(
    bout_df: pd.DataFrame,
    target_avi,
    target_event_ind,
    subject: str,
    mode: str,
    predictor_A: str,
    predictor_B: str,
    response: str = "thetadot",
    L: int = 7,
    alpha: float = 100.0,
    kernel_mode: str = "symmetric",
    model_cls=Ridge,
) -> dict | None:
    """LOO Ridge fit with combined design matrix [X_A | X_B].

    Fits a single Ridge model using both predictor_A and predictor_B as
    simultaneous predictors.  The design matrix is built by horizontally
    concatenating the lag matrices:
        X_mixed = [X_A (T x n_lags), X_B (T x n_lags)]

    Ridge jointly estimates the optimal temporal kernel for each predictor.
    coef_A and coef_B are the marginal kernels — each encoding the unique
    contribution of its predictor controlling for the other.

    Parameters
    ----------
    bout_df           : DataFrame with all bouts
    target_avi        : avi identifier of the held-out bout
    target_event_ind  : event_ind of the held-out bout
    subject           : subject used to restrict training bouts
    mode              : bout mode used to restrict training bouts
    predictor_A       : first predictor column (e.g. 'alphadot')
    predictor_B       : second predictor column (e.g. 'Phi')
    response          : response column (default 'thetadot')
    L                 : kernel half-width in frames
    alpha             : Ridge regularisation strength
    kernel_mode       : 'causal' | 'acausal' | 'symmetric'

    Returns
    -------
    dict with keys:
        coef_A        : (n_lags,) kernel for predictor_A
        coef_B        : (n_lags,) kernel for predictor_B
        intercept     : float
        r2_train, r2_test, VAF_train, VAF_test : floats
        n_train_bouts : int
        config        : dict with "mixed": True, "predictor_A", "predictor_B"
    Returns None if insufficient training data.
    """
    mask = bout_df["mode"] == mode
    if "subject" in bout_df.columns:
        mask &= bout_df["subject"] == subject

    target_mask = (
        (bout_df["avi"] == target_avi)
        & (bout_df["event ind"] == target_event_ind)
    )
    train_df = bout_df.loc[mask & ~target_mask]
    test_df  = bout_df.loc[mask &  target_mask]

    min_frames = 2 * L + 1
    if len(test_df) < min_frames:
        return None

    bout_id_cols = [c for c in ("avi", "event ind") if c in train_df.columns]
    groups = (train_df.groupby(bout_id_cols) if bout_id_cols
              else [("all", train_df)])

    X_list, y_list = [], []
    n_lags = None

    for _, grp in groups:
        if len(grp) < min_frames:
            continue
        xA = grp[predictor_A].values.astype(np.float64)
        xB = grp[predictor_B].values.astype(np.float64)
        y  = grp[response].values.astype(np.float64)
        XA, n_lags = build_behavioral_lag_matrix(xA, L, mode=kernel_mode)
        XB, _      = build_behavioral_lag_matrix(xB, L, mode=kernel_mode)
        X_list.append(np.hstack([XA, XB]))
        y_list.append(y)

    if len(X_list) < 2 or n_lags is None:
        return None

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    model = model_cls(alpha=alpha)
    model.fit(X_train, y_train)

    x_A_test = test_df[predictor_A].values.astype(np.float64)
    x_B_test = test_df[predictor_B].values.astype(np.float64)
    y_test   = test_df[response].values.astype(np.float64)

    XA_test, _ = build_behavioral_lag_matrix(x_A_test, L, mode=kernel_mode)
    XB_test, _ = build_behavioral_lag_matrix(x_B_test, L, mode=kernel_mode)
    X_test = np.hstack([XA_test, XB_test])

    y_hat_train = model.predict(X_train)
    y_hat_test  = model.predict(X_test)

    coef = model.coef_

    return {
        "coef_A":        coef[:n_lags],
        "coef_B":        coef[n_lags:],
        "intercept":     float(model.intercept_),
        "r2_train":      r2_score(y_train, y_hat_train),
        "r2_test":       r2_score(y_test,  y_hat_test),
        "VAF_train":     compute_vaf(y_hat_train, y_train),
        "VAF_test":      compute_vaf(y_hat_test,  y_test),
        "n_train_bouts": len(X_list),
        "config": {
            "L": L, "n_lags": n_lags, "mode": kernel_mode,
            "alpha": alpha, "loo": True, "mixed": True,
            "predictor_A": predictor_A, "predictor_B": predictor_B,
        },
    }


def collect_all_predictions_loo(
    bout_df: pd.DataFrame,
    subject: str,
    mode: str,
    predictor_A: str,
    predictor_B: str,
    response: str = "thetadot",
    L: int = 7,
    alpha: float = 100.0,
    kernel_mode: str = "symmetric",
    min_valid_frames: int = 30,
    model_cls=Ridge,
) -> dict | None:
    """Pool LOO predictions for all bouts of one subject/mode.

    For each held-out bout, trains three Ridge models on all remaining bouts:
        A     : predictor_A only
        B     : predictor_B only
        mixed : both predictors jointly

    Lag matrices for each training bout are built independently to avoid
    cross-bout boundary artifacts (same convention as fit_behavioral_kernel_loo).
    All three models share the same training loop for efficiency.

    Parameters
    ----------
    bout_df           : DataFrame with all bouts
    subject           : subject identifier
    mode              : bout mode to restrict to
    predictor_A       : first predictor column (e.g. 'alphadot')
    predictor_B       : second predictor column (e.g. 'Phi')
    response          : response column
    L, alpha, kernel_mode : kernel hyperparameters
    min_valid_frames  : minimum frames in the test bout

    Returns
    -------
    dict with:
        bouts         : list of per-bout dicts, each with arrays:
                          y_true, y_hat_A, y_hat_B, y_hat_mixed, x_A, x_B
        y_true        : (N,) concatenated true response
        y_hat_A       : (N,) concatenated predictions from A only
        y_hat_B       : (N,) concatenated predictions from B only
        y_hat_mixed   : (N,) concatenated predictions from A+B
        n_bouts       : int
    Returns None if no valid bouts.
    """
    mask = bout_df["mode"] == mode
    if "subject" in bout_df.columns:
        mask &= bout_df["subject"] == subject
    subj_df = bout_df.loc[mask]

    bout_id_cols = [c for c in ("avi", "event ind") if c in subj_df.columns]
    bout_keys = [
        (k if isinstance(k, tuple) else (k,))
        for k, _ in subj_df.groupby(bout_id_cols)
    ]

    min_frames = 2 * L + 1
    bouts_out  = []

    for key in tqdm.tqdm(bout_keys, desc=f"LOO predictions {subject}/{mode}"):
        avi       = key[0]
        event_ind = key[1] if len(key) > 1 else None

        target_mask = (
            (bout_df["avi"] == avi)
            & (bout_df["event ind"] == event_ind)
        )
        test_df  = bout_df.loc[mask &  target_mask]
        train_df = bout_df.loc[mask & ~target_mask]

        if len(test_df) < max(min_valid_frames, min_frames):
            continue

        # Build lag matrices per training bout (no cross-bout leakage)
        XA_list, XB_list, y_list = [], [], []
        n_lags = None
        tr_id_cols = [c for c in ("avi", "event ind") if c in train_df.columns]
        tr_groups  = (train_df.groupby(tr_id_cols) if tr_id_cols
                      else [("all", train_df)])

        for _, grp in tr_groups:
            if len(grp) < min_frames:
                continue
            xA = grp[predictor_A].values.astype(np.float64)
            xB = grp[predictor_B].values.astype(np.float64)
            y  = grp[response].values.astype(np.float64)
            XA_b, n_lags = build_behavioral_lag_matrix(xA, L, mode=kernel_mode)
            XB_b, _      = build_behavioral_lag_matrix(xB, L, mode=kernel_mode)
            XA_list.append(XA_b)
            XB_list.append(XB_b)
            y_list.append(y)

        if len(XA_list) < 2 or n_lags is None:
            continue

        XA_tr   = np.vstack(XA_list)
        XB_tr   = np.vstack(XB_list)
        X_mx_tr = np.hstack([XA_tr, XB_tr])
        y_tr    = np.concatenate(y_list)

        # Fit all three models (shared training data)
        ridge = model_cls(alpha=alpha)
        ridge.fit(XA_tr, y_tr);   coef_A  = ridge.coef_.copy(); ic_A  = float(ridge.intercept_)
        ridge.fit(XB_tr, y_tr);   coef_B  = ridge.coef_.copy(); ic_B  = float(ridge.intercept_)
        ridge.fit(X_mx_tr, y_tr); coef_mx = ridge.coef_.copy(); ic_mx = float(ridge.intercept_)

        # Predict on held-out test bout
        x_A_test = test_df[predictor_A].values.astype(np.float64)
        x_B_test = test_df[predictor_B].values.astype(np.float64)
        y_test   = test_df[response].values.astype(np.float64)

        XA_test, _ = build_behavioral_lag_matrix(x_A_test, L, mode=kernel_mode)
        XB_test, _ = build_behavioral_lag_matrix(x_B_test, L, mode=kernel_mode)
        X_mx_test  = np.hstack([XA_test, XB_test])

        bouts_out.append({
            "y_true":        y_test,
            "y_hat_A":       XA_test   @ coef_A            + ic_A,
            "y_hat_B":       XB_test   @ coef_B            + ic_B,
            "y_hat_mixed":   X_mx_test @ coef_mx           + ic_mx,
            "x_A":           x_A_test,
            "x_B":           x_B_test,
            # Bout identity (for raw data lookup)
            "avi":           avi,
            "event_ind":     event_ind,
            # Kernels from this LOO fold (for shape comparison)
            "coef_A_alone":  coef_A,                 # predictor_A kernel, A-only model
            "coef_B_alone":  coef_B,                 # predictor_B kernel, B-only model
            "coef_A_mixed":  coef_mx[:n_lags],       # predictor_A kernel, mixed model
            "coef_B_mixed":  coef_mx[n_lags:],       # predictor_B kernel, mixed model
            # Intercepts (needed to reconstruct predictions externally)
            "ic_A":          ic_A,
            "ic_B":          ic_B,
            "ic_mixed":      ic_mx,
        })

    if not bouts_out:
        return None

    return {
        "bouts":       bouts_out,
        "y_true":      np.concatenate([b["y_true"]      for b in bouts_out]),
        "y_hat_A":     np.concatenate([b["y_hat_A"]     for b in bouts_out]),
        "y_hat_B":     np.concatenate([b["y_hat_B"]     for b in bouts_out]),
        "y_hat_mixed": np.concatenate([b["y_hat_mixed"] for b in bouts_out]),
        "n_bouts":     len(bouts_out),
    }


# ---------------------------------------------------------------------------
# Reichardt-Poggio additive static gain model
# ---------------------------------------------------------------------------

def _fourier_basis(x_norm: np.ndarray, n_basis: int) -> np.ndarray:
    """Fourier design matrix for x_norm in [-1, 1].

    Columns: [sin(πx), ..., sin(n·πx), cos(πx), ..., cos(n·πx)]
    where n = n_basis // 2.  n_basis should be even.
    """
    n_h = n_basis // 2
    k   = np.arange(1, n_h + 1)
    ang = np.outer(x_norm, k * np.pi)
    return np.hstack([np.sin(ang), np.cos(ang)])


def fit_rp_static_gain(
    phi: np.ndarray,
    phidot: np.ndarray,
    omega: np.ndarray,
    tau_frames: int = 0,
    n_basis: int = 8,
    alpha: float = 1.0,
    phi_range: float = 180.0,
    phidot_clip_pct: float = 99.0,
    train_fraction: float = 0.8,
    model_cls=Ridge,
) -> dict | None:
    """Fit the Reichardt-Poggio multiplicative gain model.

    Estimates nonlinear gain functions g and h such that:

        omega(t) ≈ g(phi(t − tau)) + h(phi(t − tau)) * phidot(t − tau) + intercept

    where phi is the head-centric retinal position error (Phi column) and
    phidot = dPhi/dt is the retinal slip velocity.  Both g and h are functions
    of phi represented by a shared Fourier basis (n_basis // 2 sine + cosine
    harmonics), fit jointly with Ridge regression.  The design matrix is:

        X = [B_phi(phi), phidot_norm * B_phi(phi)]   shape (T, 2*n_basis)

    so the regression remains linear in the coefficients.

    Parameters
    ----------
    phi            : retinal position error in degrees, shape (T,)
    phidot         : retinal slip velocity in deg/s (= dPhi/dt), shape (T,)
    omega          : mouse turning rate in deg/s, shape (T,)
    tau_frames     : sensorimotor delay in frames (0 = contemporaneous)
    n_basis        : Fourier basis functions per component; must be even
    alpha          : Ridge regularisation strength
    phi_range      : half-range for phi normalisation (default 180 degrees)
    phidot_clip_pct: percentile of |phidot| used as normalisation scale
    train_fraction : fraction of frames used for training (first portion)

    Returns
    -------
    dict with keys:
        coef_g, coef_h  : (n_basis,) basis coefficients for g(phi) and h(phi)
        intercept       : float
        g_grid, h_grid  : (100,) gain functions evaluated on phi_grid
        phi_grid        : (100,) phi values in degrees (shared x-axis for g and h)
        phidot_scale    : float, normalisation scale for phidot
        r2_train, r2_test, VAF_train, VAF_test : floats
        config          : hyperparameter dict; includes ``"rp_model": True``
    Returns None if fewer than 10 test frames remain after the train split.
    """
    phi    = np.asarray(phi,    dtype=np.float64)
    phidot = np.asarray(phidot, dtype=np.float64)
    omega  = np.asarray(omega,  dtype=np.float64)

    if tau_frames > 0:
        phi_in    = phi[:-tau_frames]
        phidot_in = phidot[:-tau_frames]
        omega_in  = omega[tau_frames:]
    else:
        phi_in, phidot_in, omega_in = phi, phidot, omega

    phi_norm     = phi_in / phi_range
    phidot_scale = float(np.percentile(np.abs(phidot_in), phidot_clip_pct)) or 1.0
    phidot_norm  = np.clip(phidot_in / phidot_scale, -1.0, 1.0)

    B_phi = _fourier_basis(phi_norm, n_basis)
    X = np.hstack([B_phi, phidot_norm[:, None] * B_phi])

    split = int(len(omega_in) * train_fraction)
    if len(omega_in) - split < 10:
        return None

    model = model_cls(alpha=alpha)
    model.fit(X[:split], omega_in[:split])

    coef_g = model.coef_[:n_basis]
    coef_h = model.coef_[n_basis:]

    phi_grid  = np.linspace(-phi_range, phi_range, 100)
    B_grid    = _fourier_basis(phi_grid / phi_range, n_basis)
    g_grid    = B_grid @ coef_g
    h_grid    = B_grid @ coef_h

    y_hat_train = model.predict(X[:split])
    y_hat_test  = model.predict(X[split:])

    return {
        "coef_g":       coef_g,
        "coef_h":       coef_h,
        "intercept":    float(model.intercept_),
        "g_grid":       g_grid,
        "h_grid":       h_grid,
        "phi_grid":     phi_grid,
        "phidot_scale": phidot_scale,
        "r2_train":     r2_score(omega_in[:split], y_hat_train),
        "r2_test":      r2_score(omega_in[split:], y_hat_test),
        "VAF_train":    compute_vaf(y_hat_train, omega_in[:split]),
        "VAF_test":     compute_vaf(y_hat_test,  omega_in[split:]),
        "config": {
            "tau_frames":      tau_frames,
            "n_basis":         n_basis,
            "alpha":           alpha,
            "phi_range":       phi_range,
            "phidot_scale":    phidot_scale,
            "phidot_clip_pct": phidot_clip_pct,
            "train_fraction":  train_fraction,
            "rp_model":        True,
        },
    }


def fit_rp_static_gain_loo(
    bout_df: pd.DataFrame,
    target_avi,
    target_event_ind,
    subject: str,
    mode: str,
    phi_col: str = "Phi",
    phidot_col: str | None = None,
    response: str = "thetadot",
    tau_frames: int = 0,
    n_basis: int = 8,
    alpha: float = 1.0,
    phi_range: float = 180.0,
    phidot_clip_pct: float = 99.0,
    model_cls=Ridge,
) -> dict | None:
    """LOO fit of the R&P additive static gain model.

    Trains on all same-subject/mode bouts except the held-out target bout.
    The phidot normalisation scale is derived from training data only and
    applied to the test bout, preventing leakage.

    Parameters
    ----------
    bout_df           : DataFrame containing all bouts
    target_avi        : avi identifier of the held-out bout
    target_event_ind  : event_ind of the held-out bout
    subject           : subject used to restrict training bouts
    mode              : bout mode used to restrict training bouts
    phi_col           : column name for retinal position error (default 'Phi')
    phidot_col        : column name for retinal slip velocity; if None,
                        computed per bout as dPhi/dt via np.gradient
    response          : response column (default 'thetadot')
    tau_frames        : sensorimotor delay in frames
    n_basis           : Fourier basis functions per predictor (must be even)
    alpha             : Ridge regularisation strength
    phi_range         : half-range for phi normalisation
    phidot_clip_pct   : percentile of |phidot| from training data for scale

    Returns
    -------
    dict with the same keys as fit_rp_static_gain, plus:
        n_train_bouts : int
        config        : includes ``"loo": True, "rp_model": True``
    Returns None if fewer than 2 valid training bouts or test bout too short.
    """
    mask = bout_df["mode"] == mode
    if "subject" in bout_df.columns:
        mask &= bout_df["subject"] == subject

    target_mask = (bout_df["avi"] == target_avi) & (bout_df["event ind"] == target_event_ind)
    train_df = bout_df.loc[mask & ~target_mask]
    test_df  = bout_df.loc[mask &  target_mask]

    min_frames = max(tau_frames + 1, 10)
    if len(test_df) < min_frames:
        return None

    bout_id_cols = [c for c in ("avi", "event ind") if c in train_df.columns]
    groups = train_df.groupby(bout_id_cols) if bout_id_cols else [("all", train_df)]

    raw_bouts, phidot_pool = [], []
    for _, grp in groups:
        if len(grp) < min_frames:
            continue
        phi_b    = grp[phi_col].values.astype(np.float64)
        phidot_b = (grp[phidot_col].values.astype(np.float64) if phidot_col is not None
                    else np.gradient(phi_b, 1.0 / SAMPLING_RATE))
        omega_b  = grp[response].values.astype(np.float64)
        if tau_frames > 0:
            phi_b, phidot_b, omega_b = phi_b[:-tau_frames], phidot_b[:-tau_frames], omega_b[tau_frames:]
        if len(omega_b) < 10:
            continue
        raw_bouts.append((phi_b, phidot_b, omega_b))
        phidot_pool.append(phidot_b)

    if len(raw_bouts) < 2:
        return None

    # phidot_scale from training frames only
    phidot_scale = float(np.percentile(np.abs(np.concatenate(phidot_pool)), phidot_clip_pct)) or 1.0

    X_list, y_list = [], []
    for phi_b, phidot_b, omega_b in raw_bouts:
        phi_norm    = phi_b / phi_range
        phidot_norm = np.clip(phidot_b / phidot_scale, -1.0, 1.0)
        B = _fourier_basis(phi_norm, n_basis)
        X_list.append(np.hstack([B, phidot_norm[:, None] * B]))
        y_list.append(omega_b)

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    model = model_cls(alpha=alpha)
    model.fit(X_train, y_train)

    # Test bout
    phi_t    = test_df[phi_col].values.astype(np.float64)
    phidot_t = (test_df[phidot_col].values.astype(np.float64) if phidot_col is not None
                else np.gradient(phi_t, 1.0 / SAMPLING_RATE))
    omega_t  = test_df[response].values.astype(np.float64)
    if tau_frames > 0:
        phi_t, phidot_t, omega_t = phi_t[:-tau_frames], phidot_t[:-tau_frames], omega_t[tau_frames:]
    if len(omega_t) < 10:
        return None

    phi_t_norm    = phi_t / phi_range
    phidot_t_norm = np.clip(phidot_t / phidot_scale, -1.0, 1.0)
    B_test  = _fourier_basis(phi_t_norm, n_basis)
    X_test  = np.hstack([B_test, phidot_t_norm[:, None] * B_test])

    coef_g = model.coef_[:n_basis]
    coef_h = model.coef_[n_basis:]

    phi_grid = np.linspace(-phi_range, phi_range, 100)
    B_grid   = _fourier_basis(phi_grid / phi_range, n_basis)
    g_grid   = B_grid @ coef_g
    h_grid   = B_grid @ coef_h

    y_hat_train = model.predict(X_train)
    y_hat_test  = model.predict(X_test)

    return {
        "coef_g":        coef_g,
        "coef_h":        coef_h,
        "intercept":     float(model.intercept_),
        "g_grid":        g_grid,
        "h_grid":        h_grid,
        "phi_grid":      phi_grid,
        "phidot_scale":  phidot_scale,
        "r2_train":      r2_score(y_train,  y_hat_train),
        "r2_test":       r2_score(omega_t,  y_hat_test),
        "VAF_train":     compute_vaf(y_hat_train, y_train),
        "VAF_test":      compute_vaf(y_hat_test,  omega_t),
        "n_train_bouts": len(raw_bouts),
        "config": {
            "tau_frames":      tau_frames,
            "n_basis":         n_basis,
            "alpha":           alpha,
            "phi_range":       phi_range,
            "phidot_scale":    phidot_scale,
            "phidot_clip_pct": phidot_clip_pct,
            "loo":             True,
            "rp_model":        True,
        },
    }


def kernel_at_speed(bilinear_result: dict, s: float) -> np.ndarray:
    """Evaluate the bilinear kernel K(s) = K0 + s_z * K1.

    Converts s to the z-scored units used during fitting before applying the
    bilinear formula, so s should be in the same original units as the speed
    array passed to fit_speed_modulated_kernel (cm/s).

    Parameters
    ----------
    bilinear_result : dict returned by fit_speed_modulated_kernel
    s               : speed value in cm/s

    Returns
    -------
    (n_lags,) array representing the temporal integration kernel at speed s
    """
    cfg  = bilinear_result["config"]
    s_z  = (s - cfg["speed_mean"]) / cfg["speed_std"]
    return bilinear_result["K0"] + s_z * bilinear_result["K1"]


# ---------------------------------------------------------------------------
# Causal / acausal comparison
# ---------------------------------------------------------------------------

def compare_causal_acausal(
    x: np.ndarray,
    y: np.ndarray,
    L: int,
    alpha: float,
    model_cls=Ridge,
) -> dict:
    """Fit causal, acausal, and symmetric kernels and compare test r².

    Scientific interpretation
    -------------------------
    acausal_r2 >> causal_r2  →  thetadot *leads* the predictor (or the
                                predictor lags a common upstream variable).
    causal_r2 ≈ acausal_r2  →  contemporaneous or bidirectional relationship.

    The peak of the causal kernel (argmax of absolute value) estimates the
    dominant temporal lag, which should align with the best discrete tau from
    run_discrete_delay_analysis.

    Returns
    -------
    dict with keys:
        causal_r2, acausal_r2, symmetric_r2  : test-set r²
        causal_kernel, acausal_kernel         : (L,) arrays
        symmetric_kernel                      : (2L+1,) array
        peak_causal_lag_frames                : int
        peak_causal_lag_ms                    : float
    Returns None if any fit fails (bout too short).
    """
    causal_fit   = fit_behavioral_kernel(x, y, L, alpha, mode="causal",    model_cls=model_cls)
    acausal_fit  = fit_behavioral_kernel(x, y, L, alpha, mode="acausal",   model_cls=model_cls)
    symmetric_fit = fit_behavioral_kernel(x, y, L, alpha, mode="symmetric", model_cls=model_cls)

    if any(f is None for f in [causal_fit, acausal_fit, symmetric_fit]):
        return None

    causal_kernel = causal_fit["coef"]
    peak_frame    = int(np.argmax(np.abs(causal_kernel)))

    return {
        "causal_r2":       causal_fit["r2_test"],
        "acausal_r2":      acausal_fit["r2_test"],
        "symmetric_r2":    symmetric_fit["r2_test"],
        "causal_VAF":      causal_fit["VAF_test"],
        "acausal_VAF":     acausal_fit["VAF_test"],
        "symmetric_VAF":   symmetric_fit["VAF_test"],
        "causal_kernel":   causal_kernel,
        "acausal_kernel":  acausal_fit["coef"],
        "symmetric_kernel": symmetric_fit["coef"],
        "peak_causal_lag_frames": peak_frame,
        "peak_causal_lag_ms":     peak_frame * DT_MS,
    }


# ---------------------------------------------------------------------------
# Parameter scan
# ---------------------------------------------------------------------------

def scan_kernel_parameters(
    x: np.ndarray,
    y: np.ndarray,
    L_values: list = None,
    alpha_values: list = None,
    mode: str = "symmetric",
    model_cls=Ridge,
    weights: np.ndarray = None,
) -> pd.DataFrame:
    """Grid-search over L × alpha for a single time series pair.

    Mirrors scan_parameters() in regression_trainer.py.

    Returns
    -------
    DataFrame with columns: L, alpha, r2_train, r2_test, mode
    Sorted by r2_test descending.
    """
    if L_values is None:
        L_values = SCAN_L
    if alpha_values is None:
        alpha_values = SCAN_ALPHAS

    rows = []
    for L in L_values:
        for alpha in alpha_values:
            fit = fit_behavioral_kernel(x, y, L, alpha, mode=mode, model_cls=model_cls, weights=weights)
            if fit is None:
                continue
            rows.append({
                "L": L, "alpha": alpha,
                "r2_train": fit["r2_train"],
                "r2_test":  fit["r2_test"],
                "VAF_train": fit["VAF_train"],
                "VAF_test":  fit["VAF_test"],
                "mode": mode,
            })

    return pd.DataFrame(rows).sort_values("VAF_test", ascending=False)


def scan_kernel_parameters_loo(
    bout_df: pd.DataFrame,
    subject: str,
    mode: str,
    predictor: str,
    response: str = "thetadot",
    speed_col: str = None,
    L_values: list = None,
    alpha_values: list = None,
    kernel_mode: str = "symmetric",
    min_valid_frames: int = 30,
    model_cls=Ridge,
) -> pd.DataFrame:
    """Grid-search over L × alpha using leave-one-bout-out CV.

    For each (L, alpha) pair, every bout for this subject/mode is held out in
    turn; the kernel is trained on the remaining bouts and evaluated on the
    held-out bout.  Mean and std of VAF_test across bouts are returned.

    This gives an unbiased estimate of generalisation performance because
    train and test bouts are fully separated — unlike the single-bout
    train/test split in scan_kernel_parameters, which inflates test scores
    due to within-bout autocorrelation.

    Parameters
    ----------
    bout_df           : DataFrame from bb.load_tautau_df()
    subject           : subject identifier (matched against 'subject' column)
    mode              : bout mode to restrict to
    predictor         : predictor column name
    response          : response column name
    speed_col         : if provided, Ridge loss is weighted by speed / mean(speed)
    L_values          : list of kernel half-widths to scan; defaults to SCAN_L
    alpha_values      : list of Ridge alpha values to scan; defaults to SCAN_ALPHAS
    kernel_mode       : 'causal' | 'acausal' | 'symmetric'
    min_valid_frames  : minimum frames in test bout (passed to _collect_loo_data)

    Returns
    -------
    DataFrame with columns:
        L, alpha, mean_VAF_test, std_VAF_test, n_bouts, mean_r2_test, mode
    Sorted by mean_VAF_test descending.
    """
    if L_values is None:
        L_values = SCAN_L
    if alpha_values is None:
        alpha_values = SCAN_ALPHAS

    # Identify all valid bout identifiers for this subject × mode
    mask = bout_df["mode"] == mode
    if "subject" in bout_df.columns:
        mask &= bout_df["subject"] == subject
    subj_df = bout_df.loc[mask]

    bout_id_cols = [c for c in ("avi", "event ind") if c in subj_df.columns]
    bout_keys = [(k if isinstance(k, tuple) else (k,))
                 for k, _ in subj_df.groupby(bout_id_cols)]

    rows = []
    pbar = tqdm.tqdm(total=len(L_values) * len(alpha_values) * len(bout_keys), desc=f"Scanning LOO kernel params for {subject} {mode}")
    for L in L_values:
        for alpha in alpha_values:
            vafs, r2s = [], []
            for key in bout_keys:
                pbar.update(1)
                avi       = key[0]
                event_ind = key[1] if len(key) > 1 else None
                fit = fit_behavioral_kernel_loo(
                    bout_df, avi, event_ind, subject, mode,
                    predictor=predictor, response=response,
                    L=L, alpha=alpha, kernel_mode=kernel_mode,
                    speed_col=speed_col, model_cls=model_cls,
                )
                if fit is None:
                    continue
                vafs.append(fit["VAF_test"])
                r2s.append(fit["r2_test"])

            if not vafs:
                continue
            rows.append({
                "L":             L,
                "alpha":         alpha,
                "mean_VAF_test": float(np.mean(vafs)),
                "std_VAF_test":  float(np.std(vafs)),
                "n_bouts":       len(vafs),
                "mean_r2_test":  float(np.mean(r2s)),
                "mode":          mode,
            })

    return pd.DataFrame(rows).sort_values("mean_VAF_test", ascending=False)


# ---------------------------------------------------------------------------
# Full mode × speed analysis
# ---------------------------------------------------------------------------

DEFAULT_KERNEL_CONFIGS = [
    ("causal",    False),
    ("causal",    True),
    ("acausal",   False),
    ("acausal",   True),
    ("symmetric", False),
    ("symmetric", True),
]


def _config_key(km: str, weighted: bool) -> str:
    """Canonical column-name fragment for one (kernel_mode, weighted) config."""
    return f"{km}_weighted" if weighted else km


def run_kernel_analysis_by_mode_speed(
    df: pd.DataFrame,
    mode: str,
    predictor: str,
    L: int,
    alpha: float,
    output_dir: Path,
    response: str = "thetadot",
    variable: str = "speed",
    interval: float = 5.0,
    min_count: int = 20,
    v_max: float = 50.0,
    kernel_configs: list | None = None,
    save: bool = True,
    n_folds: int = 5,
    model_cls=Ridge,
) -> pd.DataFrame:
    """Run kernel regression for one behavioral mode × predictor, binned by variable.

    Fits each (kernel_mode, speed_weighted) configuration in kernel_configs using
    k-fold bout cross-validation.  Kernels are fit on full training bouts (no
    internal frame split); performance is evaluated on held-out test bouts,
    binned by `variable`.

    Parameters
    ----------
    df             : source DataFrame (filtered internally to mode + τ==0)
    mode           : 'tangent', 'BD', or 'HD'
    predictor      : predictor column (e.g. 'alphadot', 'Phi')
    L              : kernel half-width in frames
    alpha          : Ridge regularisation strength
    response       : response column, default 'thetadot'
    variable       : column to bin by, default 'speed'
    interval, min_count, v_max : binning parameters
    kernel_configs : list of (kernel_mode, speed_weighted) tuples.
                     kernel_mode ∈ {'causal', 'acausal', 'symmetric'}.
                     speed_weighted: Ridge loss weighted by w_t = speed_t / mean(speed).
                     Defaults to DEFAULT_KERNEL_CONFIGS (all 6 combinations).
    output_dir     : directory for pickle output
    save           : write {mode}_{predictor}_kernel_results.pkl if True
    n_folds        : cross-validation folds (bouts as units)
    model_cls      : Ridge-compatible estimator with sample_weight support

    Returns
    -------
    DataFrame with one row per (subject, fold, speed_bin).
    For each (km, weighted) in kernel_configs, columns are added:
        {key}_VAF, {key}_r2        — test-fold performance
        mean_{key}_kernel          — mean kernel across ALL bouts (visualisation)
    where  key = '{km}_weighted'  if weighted  else  '{km}'.
    """
    if kernel_configs is None:
        kernel_configs = DEFAULT_KERNEL_CONFIGS

    unique_modes = list(dict.fromkeys(km for km, _ in kernel_configs))

    df_clean = df.loc[
        (df["mode"] == mode) &
        (df["tau"]  == 0) &
        (df["tau1"] == 0)
    ].copy()

    bout_cols           = ["avi", "event ind", "subject"]
    available_bout_cols = [c for c in bout_cols if c in df_clean.columns]
    inner_bout_cols     = [c for c in available_bout_cols if c != "subject"]

    if "subject" in df_clean.columns:
        subject_groups = [(s, sg) for s, sg in df_clean.groupby("subject")]
    else:
        subject_groups = [("all", df_clean)]

    def _fit_bout(x, y, speed_vals):
        """Fit all requested kernel configs on one bout (no frame split)."""
        sm = speed_vals.mean(); ss = speed_vals.std() or 1.0
        w  = speed_vals / (sm or 1.0)

        lag_mats = {
            km: build_behavioral_lag_matrix(x, L, mode=km)[0]
            for km in unique_modes
        }

        r   = model_cls(alpha=alpha)
        out = {"speed_mean": sm, "speed_std": ss}

        for km, weighted in kernel_configs:
            r.fit(lag_mats[km], y, sample_weight=(w if weighted else None))
            key = _config_key(km, weighted)
            out[f"coef_{key}"] = r.coef_.copy()
            out[f"ic_{key}"]   = float(r.intercept_)

        # Peak lag from unweighted causal (reference / diagnostics)
        if "coef_causal" in out:
            out["peak_lag"] = int(np.argmax(np.abs(out["coef_causal"])))
        else:
            first_key = _config_key(*kernel_configs[0])
            out["peak_lag"] = int(np.argmax(np.abs(out[f"coef_{first_key}"])))

        return out

    all_subject_rows = []

    for subject, subj_df in subject_groups:
        bout_list = (
            list(subj_df.groupby(inner_bout_cols)) if inner_bout_cols
            else [(("all",), subj_df)]
        )
        bout_list = [(k, b) for k, b in bout_list if len(b) >= max(2 * L + 2, 15)]
        n_bouts   = len(bout_list)
        if n_bouts < 2:
            continue

        # ── Fit all bouts for visualisation kernels ───────────────────────────
        all_fits = [_fit_bout(
            b[predictor].values, b[response].values, b[variable].values
        ) for _, b in bout_list]

        mean_kernels = {
            f"mean_{_config_key(km, wt)}_kernel": np.mean(
                [f[f"coef_{_config_key(km, wt)}"] for f in all_fits], axis=0
            )
            for km, wt in kernel_configs
        }
        median_peak_lag = float(np.median([f["peak_lag"] for f in all_fits]))

        # ── Precompute valid speed bins ───────────────────────────────────────
        all_speed  = np.concatenate([b[variable].values for _, b in bout_list])
        valid_bins = []
        for hi in range(int(interval), int(v_max) + 1, int(interval)):
            lo = hi - interval
            if ((all_speed > lo) & (all_speed <= hi)).sum() >= min_count:
                valid_bins.append((lo, hi))
            else:
                break
        if not valid_bins:
            continue

        # ── k-fold assignment ─────────────────────────────────────────────────
        k    = min(n_folds, n_bouts)
        rng  = np.random.default_rng(42)
        perm = rng.permutation(n_bouts)
        fold_assignments = np.array_split(perm, k)

        for fold_idx, test_idx in enumerate(fold_assignments):
            test_set   = set(test_idx.tolist())
            train_list = [i for i in range(n_bouts) if i not in test_set]
            if not train_list:
                continue

            # Mean kernel + intercept from training bouts
            tf = [all_fits[i] for i in train_list]
            mk = {}
            ic = {}
            for km, weighted in kernel_configs:
                key    = _config_key(km, weighted)
                mk[key] = np.mean([f[f"coef_{key}"] for f in tf], axis=0)
                ic[key] = float(np.mean([f[f"ic_{key}"] for f in tf]))

            # Predictions on held-out test bouts
            pool_y_true = []
            pool_preds  = {_config_key(km, wt): [] for km, wt in kernel_configs}
            pool_speed  = []

            for i in test_set:
                _, bout_df_i = bout_list[i]
                x_i   = bout_df_i[predictor].values
                y_i   = bout_df_i[response].values
                spd_i = bout_df_i[variable].values

                lag_mats_i = {
                    km: build_behavioral_lag_matrix(x_i, L, mode=km)[0]
                    for km in unique_modes
                }
                pool_y_true.append(y_i)
                pool_speed.append(spd_i)
                for km, weighted in kernel_configs:
                    key = _config_key(km, weighted)
                    pool_preds[key].append(lag_mats_i[km] @ mk[key] + ic[key])

            if not pool_y_true:
                continue

            y_true_all = np.concatenate(pool_y_true)
            speed_all  = np.concatenate(pool_speed)
            y_hats     = {key: np.concatenate(arrs)
                          for key, arrs in pool_preds.items()}

            for lo, hi in valid_bins:
                in_bin = (speed_all > lo) & (speed_all <= hi)
                if in_bin.sum() < 2:
                    continue

                row = {
                    "subject":                subject,
                    "fold":                   fold_idx,
                    "speed_bin":              f"{int(lo)}-{int(hi)}",
                    "n_bouts":                len(train_list),
                    "n_frames_total":         int(in_bin.sum()),
                    "median_peak_lag_frames": median_peak_lag,
                    "median_peak_lag_ms":     median_peak_lag * DT_MS,
                    "mode":                   mode,
                    "predictor":              predictor,
                    "L":                      L,
                    "alpha":                  alpha,
                }
                for key, y_hat in y_hats.items():
                    row[f"{key}_VAF"] = float(compute_vaf(y_hat[in_bin], y_true_all[in_bin]))
                    row[f"{key}_r2"]  = r2_score(y_true_all[in_bin], y_hat[in_bin])
                row.update(mean_kernels)
                all_subject_rows.append(row)

    if not all_subject_rows:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_subject_rows)

    if save and len(results_df) > 0:
        out_path = Path(output_dir) / f"{mode}_{predictor}_kernel_results.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(results_df, f)
        n_subj = results_df["subject"].nunique()
        print(f"  saved {len(results_df)} rows ({n_subj} subj, {n_folds}-fold CV) → {out_path.name}")

    return results_df


def run_all_kernel_analyses(
    df: pd.DataFrame,
    modes: list = None,
    L: int = 12,
    alpha: float = 100.0,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper: run all four guidance laws for all modes.

    Calls run_kernel_analysis_by_mode_speed for each of the four predictors
    (alphadot/PN, Phi/PP, Phidot/dPP, alpha/iPN) across all requested modes
    and concatenates the results.  Mirrors run_all_guidance_laws() in
    guidance_law_fitting.py.

    Parameters
    ----------
    df    : source DataFrame (tautau_df or equivalent)
    modes : list of mode strings; defaults to ['tangent', 'BD', 'HD']
    L     : kernel half-width in frames
    alpha : Ridge regularisation strength
    **kwargs : forwarded to run_kernel_analysis_by_mode_speed -- must include
               output_dir (required there, no default)

    Returns
    -------
    Combined DataFrame with a 'predictor' column identifying the law.
    """
    if modes is None:
        modes = ["tangent", "BD", "HD"]

    predictors = ["alphadot", "Phi", "Phidot", "alpha"]
    parts = []
    for mode in modes:
        for predictor in predictors:
            result = run_kernel_analysis_by_mode_speed(
                df, mode=mode, predictor=predictor, L=L, alpha=alpha, **kwargs
            )
            if len(result) > 0:
                parts.append(result)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
