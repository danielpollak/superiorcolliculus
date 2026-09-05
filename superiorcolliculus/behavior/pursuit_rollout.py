#!/usr/bin/env python3
"""pursuit_rollout.py — Forward-simulation engine for guidance-law fitting.

Parametric PP/PN/CATD/Mixed forward simulation + multiple-shooting fit,
consumed by analysis/behavior/deck/data.py and analysis/behavior/pp_pn_phase_space.py
in the main repo (superior_colliculus). (Its own docstring previously claimed
it was also shared with a `trajectory_fitter.py`, which doesn't exist in this
repo -- stale claim, corrected here.)

`process_bout`/`load_bouts`/`compute_alphadot_from_positions` (tautau_df bout
extraction -> the bout-dict format `simulate_bout`/`fit_law` below consume)
were absorbed in from `weak_sindy_pursuit.py` when that file's SINDy-specific
identification code was retired -- these three were never actually
SINDy-specific, just general bout-prep utilities that happened to live there.

Pursuer state: (x, y, theta) in cm and radians.
  x_{t+1}     = x_t + v_t * cos(theta_t) * dt
  y_{t+1}     = y_t + v_t * sin(theta_t) * dt
  theta_{t+1} = theta_t + G(phi, alpha_dot) * dt

Speed v(t) is observed (not modelled). Puck trajectory is ground truth.
Integration is forward Euler at the observed frame rate (dt = 1/FPS).

Guidance laws
-------------
  pp    : theta_dot = k * phi
  pn    : theta_dot = N * alpha_dot
  catd  : theta_dot = N_c * phi_dot   (fitted in PN form; reported as CATD)
  mixed : theta_dot = k * phi + N * alpha_dot

CATD note: CATD(N_c) is identical to PN(N) with N = N_c / (1 + N_c).
Optimization always runs in PN parameterization; convert after fitting.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from superiorcolliculus.processing import conversion_rate

# ── public API ────────────────────────────────────────────────────────────────
__all__ = [
    "LAWS",
    "default_params",
    "catd_to_pn",
    "pn_to_catd",
    "compute_alphadot_from_positions",
    "process_bout",
    "load_bouts",
    "prep_tgt_vel",
    "simulate_window",
    "simulate_bout",
    "compute_vpt",
    "compute_vpt_distribution",
    "bout_loss",
    "fit_law",
    "compute_hairs",
    "compute_teeth",
]

LAWS = ("pp", "pn", "catd", "mixed")

# ── parameter helpers ─────────────────────────────────────────────────────────

def default_params(law: str) -> np.ndarray:
    defaults = {"pp": [1.0], "pn": [3.0], "catd": [3.0], "mixed": [1.0, 3.0]}
    return np.array(defaults[law], dtype=float)


def catd_to_pn(N_c: float) -> float:
    """Convert CATD gain to equivalent PN gain: N = N_c / (1 + N_c)."""
    return N_c / (1.0 + N_c)


def pn_to_catd(N: float) -> float:
    """Convert PN gain to equivalent CATD gain: N_c = N / (1 - N)."""
    if abs(1.0 - N) < 1e-12:
        return float("inf")
    return N / (1.0 - N)


# ── angle helper ──────────────────────────────────────────────────────────────

def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ── data preparation ──────────────────────────────────────────────────────────

def prep_tgt_vel(bout: dict) -> np.ndarray:
    """Central-difference puck velocity in cm/s, as complex array."""
    tgt = bout["tgt_cm"]
    times = np.arange(len(tgt)) * bout["dt"]
    vx = np.gradient(tgt.real, times)
    vy = np.gradient(tgt.imag, times)
    return vx + 1j * vy


def _to_complex(series: pd.Series) -> np.ndarray:
    arr = series.values
    if arr.dtype == object:
        return np.array(arr, dtype=complex)
    return arr.astype(complex)


def compute_alphadot_from_positions(
    pos_cm: np.ndarray,   # complex, cm
    tgt_cm: np.ndarray,   # complex, cm
    r_cm: np.ndarray,     # range in cm (already available in DataFrame)
    fps: float,
) -> np.ndarray:
    """alpha_dot = v_perp / r  via the cross-product formula [rad/s].

    Cross product in 2D: (r_vec × dr_vec/dt) / r^2
    where r_vec = tgt - pos (LOS vector).

    We differentiate Cartesian positions (already DLC-smoothed), NOT the
    wrapped angle alpha, to avoid footfall-noise aliasing.
    """
    dx = tgt_cm.real - pos_cm.real   # LOS components (cm)
    dy = tgt_cm.imag - pos_cm.imag

    # Central differences on LOS vector (units: cm/frame)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Cross product / r^2  → rad/frame; multiply by fps → rad/s
    cross = dx * ddy - dy * ddx
    alphadot_rad_s = (cross / np.maximum(r_cm**2, 0.1)) * fps
    return alphadot_rad_s


def process_bout(grp: pd.DataFrame, fps: float, delay_frames: int = 0) -> Optional[dict]:
    """Extract time-series features for one tautau_df bout into the bout-dict
    format simulate_bout/fit_law/bout_loss above consume.

    With delay_frames=τ, visual inputs (phi, alphadot) are taken from t-τ and
    motor-state inputs (v, r, ṙ) are returned in both current (t) and delayed
    (t-τ) forms so a downstream fit can combine them freely (this delay-split
    convention originates from weak_sindy_pursuit.py's library construction;
    kept as-is here even though most current callers use delay_frames=0).

    Returns None if required columns are missing, contain NaN, or the bout is
    too short for the requested delay.
    """
    dt = 1.0 / fps

    # ── positions ─────────────────────────────────────────────────────────
    has_positions = "pos" in grp.columns and "target" in grp.columns
    if has_positions:
        try:
            pos_px  = _to_complex(grp["pos"])
            tgt_px  = _to_complex(grp["target"])
        except (TypeError, ValueError):
            has_positions = False

    if has_positions:
        if np.any(np.isnan(pos_px.real) | np.isnan(pos_px.imag)):
            has_positions = False
        if np.any(np.isnan(tgt_px.real) | np.isnan(tgt_px.imag)):
            has_positions = False

    # ── range and range-rate (cm, cm/s) ───────────────────────────────────
    r = np.clip(grp["dist"].values.astype(float), 0.5, None)
    rdot = np.gradient(r) * fps   # cm/s; negative = closing

    # ── heading: complex phasor → unwrapped radians ───────────────────────
    theta_phasor = _to_complex(grp["theta"])
    theta_rad = np.unwrap(np.angle(theta_phasor))

    # ── bearing (deg → rad) ───────────────────────────────────────────────
    phi_rad = np.radians(grp["Phi"].values.astype(float))

    # ── speed (cm/s) ──────────────────────────────────────────────────────
    v = np.clip(grp["speed"].values.astype(float), 0.0, None)

    # ── alphadot (rad/s) — prefer position-derived, fall back to column ───
    if has_positions:
        pos_cm = pos_px / conversion_rate
        tgt_cm = tgt_px / conversion_rate
        alphadot_rad_s = compute_alphadot_from_positions(pos_cm, tgt_cm, r, fps)
    else:
        warnings.warn(
            "pos/target columns missing or invalid — "
            "falling back to alphadot column (may be noisier)."
        )
        alphadot_rad_s = np.radians(grp["alphadot"].values.astype(float))
        pos_cm = None
        tgt_cm = None

    thetadot_rad_s = np.radians(grp["thetadot"].values.astype(float))

    # ── drop NaN rows, find valid contiguous region ────────────────────────
    cols_to_check = np.column_stack([theta_rad, phi_rad, alphadot_rad_s, v, r])
    valid = ~np.any(np.isnan(cols_to_check), axis=1)
    if valid.sum() < 10:
        return None
    idx = np.where(valid)[0]
    s, e = idx[0], idx[-1] + 1   # valid region [s, e)

    # Trim all arrays to valid region
    theta_rad      = theta_rad[s:e]
    thetadot_rad_s = thetadot_rad_s[s:e]
    phi_rad        = phi_rad[s:e]
    alphadot_rad_s = alphadot_rad_s[s:e]
    v              = v[s:e]
    r              = r[s:e]
    rdot           = rdot[s:e]
    pos_cm_arr     = pos_cm[s:e] if pos_cm is not None else None
    tgt_cm_arr     = tgt_cm[s:e] if tgt_cm is not None else None
    L = e - s

    # ── apply sensorimotor delay split ────────────────────────────────────
    tau = delay_frames
    if tau > 0:
        if L <= tau:
            return None
        # Visual inputs shifted to t-τ (first L-τ values)
        phi_rad        = phi_rad[:-tau]
        alphadot_rad_s = alphadot_rad_s[:-tau]
        # Response + current motor state advanced by τ (last L-τ values)
        thetadot_rad_s = thetadot_rad_s[tau:]
        theta_rad      = theta_rad[tau:]
        v_cur          = v[tau:]
        r_cur          = r[tau:]
        rdot_cur       = rdot[tau:]
        # Delayed motor state (first L-τ values, aligned with visual inputs)
        v_del          = v[:-tau]
        r_del          = r[:-tau]
        rdot_del       = rdot[:-tau]
        if pos_cm_arr is not None:
            pos_cm_arr = pos_cm_arr[tau:]
            tgt_cm_arr = tgt_cm_arr[tau:]
        T_out = L - tau
    else:
        v_cur = v_del = v
        r_cur = r_del = r
        rdot_cur = rdot_del = rdot
        T_out = L

    return dict(
        T=T_out,
        theta_rad=theta_rad,
        thetadot_rad_s=thetadot_rad_s,
        phi_rad=phi_rad,
        alphadot_rad_s=alphadot_rad_s,
        v_cur=v_cur,
        v_del=v_del,
        r_cur=r_cur,
        r_del=r_del,
        rdot_cur=rdot_cur,
        rdot_del=rdot_del,
        info_weight=np.abs(alphadot_rad_s),
        pos_cm=pos_cm_arr,
        tgt_cm=tgt_cm_arr,
        dt=dt,
    )


def load_bouts(
    path: Path, mode: str, fps: float, min_frames: int, delay_frames: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Load tautau_df, split into train/test bouts (80/20 at bout level).

    Uses tau=0, tau1=0 slice (all variables at current time); the
    sensorimotor delay split is applied inside process_bout.
    """
    df = pd.read_pickle(path)
    df = df[(df["mode"] == mode) & (df["tau"] == 0) & (df["tau1"] == 0)].copy()
    df = df.reset_index(drop=True)

    bout_cols = [c for c in ("avi", "event ind") if c in df.columns]
    if not bout_cols:
        raise ValueError("No bout ID columns found (expected 'avi' and/or 'event ind')")
    df["_bid"] = df.groupby(bout_cols).ngroup()

    bouts = []
    for bid, grp in df.groupby("_bid"):
        grp = grp.reset_index(drop=True)
        if len(grp) < min_frames + delay_frames:
            continue
        bout = process_bout(grp, fps, delay_frames=delay_frames)
        if bout is not None:
            bout["bout_id"] = int(bid)
            bout["delay_frames"] = delay_frames
            bouts.append(bout)

    if not bouts:
        raise RuntimeError(
            f"No valid bouts found in {path} for mode='{mode}'. "
            "Check --mode and --min_bout_frames."
        )

    # 80/20 bout-level split (fixed seed)
    rng_split = np.random.default_rng(42)
    idx = rng_split.permutation(len(bouts))
    n_train = max(1, int(0.8 * len(bouts)))
    train_bouts = [bouts[i] for i in idx[:n_train]]
    test_bouts  = [bouts[i] for i in idx[n_train:]]

    print(
        f"Loaded {len(bouts)} bouts "
        f"({len(train_bouts)} train / {len(test_bouts)} test) "
        f"from {path.name}"
    )
    return train_bouts, test_bouts


# ── single-window Euler simulation ───────────────────────────────────────────

def simulate_window(
    state0: np.ndarray,    # [x0, y0, theta0]  cm, cm, rad
    obs_pos: np.ndarray,   # complex (T,) cm  — ground truth within window
    tgt_pos: np.ndarray,   # complex (T,) cm  — puck position within window
    tgt_vel: np.ndarray,   # complex (T,) cm/s — puck velocity within window
    v: np.ndarray,         # float   (T,) cm/s — mouse speed within window
    dt: float,             # seconds per frame
    law: str,
    params: np.ndarray,
    delay_frames: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Forward Euler rollout through one multiple-shooting window.

    Returns
    -------
    sim_pos   : complex ndarray shape (T,), simulated positions
    window_sse: sum of squared position errors (cm²)
    """
    if law not in LAWS:
        raise ValueError(f"Unknown law '{law}'. Choose from {LAWS}.")

    # CATD → PN
    if law == "catd":
        law = "pn"
        params = np.array([catd_to_pn(params[0])])

    if law == "pp":
        k, = params
    elif law == "pn":
        N, = params
    elif law == "mixed":
        k, N = params

    T = len(obs_pos)
    sim_pos = np.empty(T, dtype=complex)

    x, y, theta = float(state0[0]), float(state0[1]), float(state0[2])
    sim_pos[0] = complex(x, y)

    for t in range(T - 1):
        t_d  = t - delay_frames
        t_lo = max(0, int(np.floor(t_d)))
        t_hi = min(T - 1, t_lo + 1)
        frac = t_d - t_lo
        tx   = (1-frac)*tgt_pos[t_lo].real + frac*tgt_pos[t_hi].real
        ty   = (1-frac)*tgt_pos[t_lo].imag + frac*tgt_pos[t_hi].imag
        tdx  = (1-frac)*tgt_vel[t_lo].real + frac*tgt_vel[t_hi].real
        tdy  = (1-frac)*tgt_vel[t_lo].imag + frac*tgt_vel[t_hi].imag
        vt   = max(float(v[t]), 0.0)

        dx = tx - x
        dy = ty - y
        r2 = max(dx * dx + dy * dy, 0.25)

        alpha     = np.arctan2(dy, dx)
        phi       = _wrap(alpha - theta)
        alpha_dot = (dx * (tdy - vt * np.sin(theta)) -
                     dy * (tdx - vt * np.cos(theta))) / r2

        if law == "pp":
            theta_dot = k * phi
        elif law == "pn":
            theta_dot = N * alpha_dot
        else:  # mixed
            theta_dot = k * phi + N * alpha_dot

        x      = x + vt * np.cos(theta) * dt
        y      = y + vt * np.sin(theta) * dt
        theta  = _wrap(theta + theta_dot * dt)
        sim_pos[t + 1] = complex(x, y)

    sse = float(np.sum(np.abs(sim_pos - obs_pos) ** 2))
    return sim_pos, sse


# ── full-bout simulation (returns heading trajectory) ─────────────────────────

def simulate_bout(
    bout: dict,
    law: str,
    params: np.ndarray,
    delay_frames: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward Euler simulation over a full bout; returns (sim_pos_cm, theta_rad).

    Unlike compute_vpt / simulate_window, also returns the heading trajectory
    so kinematics can be recomputed from the simulated state for regression.

    Parameters
    ----------
    bout   : dict with keys pos_cm, tgt_cm, v_cur, dt (and optionally theta_rad)
    law    : one of LAWS
    params : parameter array matching the law
    delay_frames : sensorimotor delay in frames (float; sub-frame interpolated)

    Returns
    -------
    sim_pos_cm : complex ndarray shape (T,) — simulated positions in cm
    theta_rad  : float ndarray shape (T,)   — simulated heading in radians
    """
    if bout.get("pos_cm") is None or bout.get("tgt_cm") is None:
        raise ValueError("bout must have pos_cm and tgt_cm")

    if law not in LAWS:
        raise ValueError(f"Unknown law {law!r}. Choose from {LAWS}.")

    if law == "catd":
        law    = "pn"
        params = np.array([catd_to_pn(params[0])])

    if law == "pp":
        k,   = params
    elif law == "pn":
        N,   = params
    elif law == "mixed":
        k, N = params

    tgt_vel      = prep_tgt_vel(bout)
    obs_pos      = bout["pos_cm"]
    tgt_pos      = bout["tgt_cm"]
    v            = bout["v_cur"]
    theta_stored = bout.get("theta_rad")
    dt           = bout["dt"]
    T            = len(obs_pos)

    theta0 = (float(theta_stored[0]) if theta_stored is not None
              else float(np.angle(tgt_pos[0] - obs_pos[0])))

    sim_pos   = np.empty(T, dtype=complex)
    theta_arr = np.empty(T, dtype=float)
    x, y, theta = float(obs_pos[0].real), float(obs_pos[0].imag), theta0
    sim_pos[0]   = complex(x, y)
    theta_arr[0] = theta

    for t in range(T - 1):
        t_d  = t - delay_frames
        t_lo = max(0, int(np.floor(t_d)))
        t_hi = min(T - 1, t_lo + 1)
        frac = t_d - t_lo
        tx   = (1-frac)*tgt_pos[t_lo].real + frac*tgt_pos[t_hi].real
        ty   = (1-frac)*tgt_pos[t_lo].imag + frac*tgt_pos[t_hi].imag
        tdx  = (1-frac)*tgt_vel[t_lo].real + frac*tgt_vel[t_hi].real
        tdy  = (1-frac)*tgt_vel[t_lo].imag + frac*tgt_vel[t_hi].imag
        vt   = max(float(v[t]), 0.0)

        dx        = tx - x
        dy        = ty - y
        r2        = max(dx * dx + dy * dy, 0.25)
        alpha     = np.arctan2(dy, dx)
        phi       = _wrap(alpha - theta)
        alpha_dot = (dx * (tdy - vt * np.sin(theta)) -
                     dy * (tdx - vt * np.cos(theta))) / r2

        if law == "pp":
            theta_dot = k * phi
        elif law == "pn":
            theta_dot = N * alpha_dot
        else:
            theta_dot = k * phi + N * alpha_dot

        x             = x + vt * np.cos(theta) * dt
        y             = y + vt * np.sin(theta) * dt
        theta         = _wrap(theta + theta_dot * dt)
        sim_pos[t+1]   = complex(x, y)
        theta_arr[t+1] = theta

    return sim_pos, theta_arr


# ── single-shot VPT ───────────────────────────────────────────────────────────

def compute_vpt(
    bout: dict,
    law: str,
    params: np.ndarray,
    eps_cm: float,
    tgt_vel: np.ndarray | None = None,
    delay_frames: float = 0.0,
) -> float:
    """Full-bout rollout from true initial state; return VPT (s).

    VPT = time of first frame where |pos_sim - pos_obs| > eps_cm.
    Returns bout duration if simulation never diverges.
    Returns nan if pos_cm / tgt_cm are missing.
    """
    if bout.get("pos_cm") is None or bout.get("tgt_cm") is None:
        return float("nan")

    if tgt_vel is None:
        tgt_vel = prep_tgt_vel(bout)

    dt  = bout["dt"]
    obs = bout["pos_cm"]
    tgt = bout["tgt_cm"]
    v   = bout["v_cur"]
    T   = len(obs)

    theta_rad = bout.get("theta_rad")
    theta0 = float(theta_rad[0]) if theta_rad is not None else float(np.angle(tgt[0] - obs[0]))
    state0 = np.array([obs[0].real, obs[0].imag, theta0])

    sim_pos, _ = simulate_window(state0, obs, tgt, tgt_vel, v, dt, law, params,
                                  delay_frames=delay_frames)

    deviations = np.abs(sim_pos - obs)
    exceed = np.where(deviations > eps_cm)[0]
    times  = np.arange(T) * dt
    if len(exceed) == 0:
        return float(times[-1])
    return float(times[exceed[0]])


def compute_vpt_distribution(
    bouts: list[dict],
    law: str,
    params: np.ndarray,
    eps_cm: float,
    delay_frames: float = 0.0,
) -> np.ndarray:
    """VPT (s) for each bout. NaN entries skipped in reporting."""
    vpts = []
    for bout in bouts:
        tgt_vel = prep_tgt_vel(bout)
        vpt = compute_vpt(bout, law, params, eps_cm, tgt_vel=tgt_vel,
                          delay_frames=delay_frames)
        vpts.append(vpt)
    return np.array(vpts)


# ── fitting ───────────────────────────────────────────────────────────────────

def bout_loss(
    bout: dict,
    tgt_vel: np.ndarray,
    window_frames: int,
    law: str,
    params: np.ndarray,
    delay_frames: float = 0.0,
    cap_frames: int | None = None,
) -> float:
    """Multiple-shooting loss for one bout.

    If cap_frames is set, only the first cap_frames of each window contribute
    to the error — aligning the loss with the capped-VPT evaluation metric.
    """
    obs_pos   = bout["pos_cm"]
    tgt_pos   = bout["tgt_cm"]
    v         = bout["v_cur"]
    theta_rad = bout.get("theta_rad")
    dt        = bout["dt"]
    T         = bout["T"]
    sse       = 0.0
    for w in range(max(1, T // window_frames)):
        s = w * window_frames
        e = min(s + window_frames + 1, T)
        if e - s < 2:
            continue
        x0     = float(obs_pos[s].real)
        y0     = float(obs_pos[s].imag)
        theta0 = (
            float(theta_rad[s]) if theta_rad is not None
            else float(np.angle(tgt_pos[s] - obs_pos[s]))
        )
        sim_pos, sse_w = simulate_window(
            np.array([x0, y0, theta0]),
            obs_pos[s:e], tgt_pos[s:e], tgt_vel[s:e], v[s:e],
            dt, law, params, delay_frames=delay_frames,
        )
        if cap_frames is not None:
            n_cap = min(cap_frames, e - s)
            sse_w = float(np.sum(np.abs(sim_pos[:n_cap] - obs_pos[s:s + n_cap]) ** 2))
        if np.isfinite(sse_w):
            sse += sse_w
    return sse


def fit_law(
    law: str,
    bouts: list[dict],
    window_frames: int,
    fit_delay: bool = True,
    delay_frames_fixed: float = 0.0,
    delay_init: float = 2.0,
    delay_bounds: tuple[float, float] = (0.0, 5.0),
    cap_frames: int | None = None,
) -> tuple[np.ndarray, float, float]:
    """L-BFGS-B multiple-shooting fit. Returns (law_params, delay_frames, loss).

    cap_frames: if set, each window's error is summed only over the first
    cap_frames frames (see bout_loss). Aligns training loss with capped VPT.
    """
    p0_law     = default_params(law)
    bounds_law = [(-20.0, 20.0)] * len(p0_law)

    if fit_delay:
        p0     = np.append(p0_law, delay_init)
        bounds = bounds_law + [delay_bounds]
    else:
        p0     = p0_law
        bounds = bounds_law

    tgt_vels = [prep_tgt_vel(b) for b in bouts]

    def _loss(params_full):
        if fit_delay:
            lp, d = params_full[:-1], float(params_full[-1])
        else:
            lp, d = params_full, delay_frames_fixed
        return sum(bout_loss(b, tv, window_frames, law, lp, d, cap_frames)
                   for b, tv in zip(bouts, tgt_vels))

    result = minimize(
        _loss, p0,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-12},
    )

    if fit_delay:
        law_params   = result.x[:-1].copy()
        delay_frames = float(result.x[-1])
    else:
        law_params   = result.x.copy()
        delay_frames = delay_frames_fixed

    return law_params, delay_frames, float(result.fun)


# ── diagnostic segment builders ───────────────────────────────────────────────

def compute_hairs(
    bout: dict,
    law: str,
    params: np.ndarray,
    eps_cm: float,
    delay_frames: float = 0.0,
) -> list[dict]:
    """Sequential restart-from-divergence segments for hairiness plots.

    Each hair runs until |sim - obs| > eps_cm, then restarts at true state.
    Returns list of dicts with keys: start_frame, end_frame, sim_pos,
    duration_s, mean_vperpoverr.
    """
    tgt_vel   = prep_tgt_vel(bout)
    obs_pos   = bout["pos_cm"]
    tgt_pos   = bout["tgt_cm"]
    v         = bout["v_cur"]
    theta_rad = bout.get("theta_rad")
    alphadot  = np.abs(bout["alphadot_rad_s"])
    dt        = bout["dt"]
    T         = bout["T"]
    times     = np.arange(T) * dt

    hairs = []
    cur   = 0
    while cur < T - 1:
        x0     = float(obs_pos[cur].real)
        y0     = float(obs_pos[cur].imag)
        theta0 = (
            float(theta_rad[cur]) if theta_rad is not None
            else float(np.angle(tgt_pos[cur] - obs_pos[cur]))
        )
        sim_pos, _ = simulate_window(
            np.array([x0, y0, theta0]),
            obs_pos[cur:], tgt_pos[cur:], tgt_vel[cur:], v[cur:],
            dt, law, params, delay_frames=delay_frames,
        )
        errors  = np.abs(sim_pos - obs_pos[cur:])
        exceed  = np.where(errors > eps_cm)[0]
        div_rel = int(exceed[0]) if len(exceed) > 0 else len(sim_pos) - 1

        end             = cur + div_rel
        seg_alphadot    = alphadot[cur : end + 1]
        mean_vperpoverr = float(np.mean(seg_alphadot)) if len(seg_alphadot) else 0.0

        hairs.append(dict(
            start_frame     = cur,
            end_frame       = end,
            sim_pos         = sim_pos[: div_rel + 1],
            duration_s      = float(times[end] - times[cur]),
            mean_vperpoverr = mean_vperpoverr,
        ))
        cur = end
        if cur >= T - 1:
            break
    return hairs


def compute_teeth(
    bout: dict,
    law: str,
    params: np.ndarray,
    eps_cm: float,
    W: int,
    delay_frames: float = 0.0,
) -> list[dict]:
    """Fixed-interval independent comb teeth for comb plots.

    Returns list of dicts with keys: start_frame, end_frame, sim_pos,
    duration_frac (fraction of W survived before divergence).
    """
    tgt_vel   = prep_tgt_vel(bout)
    obs_pos   = bout["pos_cm"]
    tgt_pos   = bout["tgt_cm"]
    v         = bout["v_cur"]
    theta_rad = bout.get("theta_rad")
    dt        = bout["dt"]
    T         = bout["T"]

    teeth = []
    for s in range(0, T - 1, W):
        max_end = min(s + W, T - 1)
        if max_end <= s:
            continue
        x0     = float(obs_pos[s].real)
        y0     = float(obs_pos[s].imag)
        theta0 = (
            float(theta_rad[s]) if theta_rad is not None
            else float(np.angle(tgt_pos[s] - obs_pos[s]))
        )
        sim_pos, _ = simulate_window(
            np.array([x0, y0, theta0]),
            obs_pos[s : max_end + 1], tgt_pos[s : max_end + 1],
            tgt_vel[s : max_end + 1], v[s : max_end + 1],
            dt, law, params, delay_frames=delay_frames,
        )
        errors  = np.abs(sim_pos - obs_pos[s : max_end + 1])
        exceed  = np.where(errors > eps_cm)[0]
        div_rel = int(exceed[0]) if len(exceed) > 0 else len(sim_pos) - 1

        n_frames      = max_end - s
        duration_frac = min(div_rel / max(n_frames, 1), 1.0)

        teeth.append(dict(
            start_frame   = s,
            end_frame     = s + div_rel,
            sim_pos       = sim_pos[: div_rel + 1],
            duration_frac = duration_frac,
        ))
    return teeth
