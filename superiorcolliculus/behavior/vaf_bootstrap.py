"""Per-bout leave-one-out guidance-law fitting + two-stage cluster bootstrap
significance testing for paired model VAF comparisons (e.g. PP vs. PN).

Moved out of analysis/behavior/deck/data.py (the LOO-per-bout fit,
`run_loo_discrete`/`run_loo_discrete_frames`) and
dissemination/CCN_2026_poster/CCN_2026_behavior.ipynb (the bootstrap/
significance layer -- `_cluster_bootstrap_vaf`, `bootstrap_vaf_df`,
`holm_correction`, `pair_by_replicate`, `paired_bootstrap_test`,
`compute_bootstrap_significance` -- previously notebook-local, with no
importable home) so this methodology, used by the poster's "Flagship plot",
is a documented, reusable peer of guidance_law_fitting.py's k-fold-CV-based
GuidanceLawFitter -- not poster-specific. See .claude/CLAUDE.md's Behavioral
Analysis Modules section.

`run_loo_discrete`/`run_loo_discrete_frames` intentionally differ from
GuidanceLawFitter's k-fold CV rather than replacing it: this module fits one
model per bout (train on every OTHER bout from that subject, test on the
held-out bout), so every single frame gets a genuinely held-out prediction --
GuidanceLawFitter's k-fold CV fits k models total (default 5), each trained on
~80% of bouts, which is coarser and leaves some frames' held-out status
dependent on a random fold assignment. Both are correct for their own
purposes; they are not merged into one class here.

`deck/data.py` still exposes `run_loo_discrete`/`run_loo_discrete_frames` as
thin `@MEMORY.cache`-wrapped re-exports of the implementations below, so
existing deck consumers (`deck/figures.py`, the `deck/explore_*.py` scripts)
are unaffected by this move -- this module itself is deliberately
cache-free/dependency-light (no joblib), so callers that don't need deck's
disk cache (e.g. this project's poster notebooks) can import it directly
without pulling in deck's cache/tqdm/joblib dependency chain (the same reason
`analysis/behavior/pp_pn_phase_space.py` gives for not importing `deck`).
"""
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from sklearn.linear_model import LinearRegression

from superiorcolliculus.analysis import (
    compute_vaf, holm_correction, two_stage_cluster_bootstrap, paired_bootstrap_test,
)

__all__ = [
    "run_loo_discrete", "run_loo_discrete_frames",
    "bootstrap_vaf_df", "pair_by_replicate", "compute_bootstrap_significance",
]


def run_loo_discrete(df: pd.DataFrame, predictor_cols: list, response: str = "thetadot") -> pd.DataFrame:
    """Per-subject, per-bout leave-one-bout-out OLS, no intercept.

    Training is restricted to the held-out bout's own subject (excluding that
    bout), matching GuidanceLawFitter.compute_vaf_2d and
    fit_behavioral_kernel_loo -- gains vary substantially across subjects
    (roughly 1x-3x for PN in this dataset), so a single slope pooled across
    every subject underfits all of them.

    No intercept: PN/PP/Mixed predict thetadot as a pure proportional response
    to the sensory signal (no baseline turning bias), matching
    GuidanceLawFitter._fit_slope. Reported metric is VAF (compute_vaf), not R2
    -- R2 is undefined without penalizing residual bias, but a no-intercept
    fit has no guarantee of zero-mean residuals even in-sample, so VAF is the
    metric that actually matches this model.
    """
    results = []
    bout_groups = list(df.groupby(["avi", "event ind"]))
    for (avi, event_ind), grp in tqdm.tqdm(bout_groups, desc=f"LOO discrete {predictor_cols}"):
        subj  = grp["subject"].iloc[0]
        mask  = (df["avi"] == avi) & (df["event ind"] == event_ind)
        train = df[(df["subject"] == subj) & ~mask].dropna(subset=predictor_cols + [response])
        test  = grp.dropna(subset=predictor_cols + [response])
        if len(train) < 20 or len(test) < 5:
            continue
        reg   = LinearRegression(fit_intercept=False).fit(
                    train[predictor_cols].values, train[response].values)
        y_hat = reg.predict(test[predictor_cols].values)
        y     = test[response].values
        results.append({
            "avi":          avi,
            "event ind":    event_ind,
            "subject":      test["subject"].iloc[0],
            "vaf_test":     compute_vaf(y_hat, y),
            "rmse_test":    float(np.sqrt(np.mean((y - y_hat) ** 2))),
            "speed_median": float(test["speed"].median()),
            "dist_median":  float(test["dist"].median()),
        })
    return pd.DataFrame(results)


def run_loo_discrete_frames(df: pd.DataFrame, predictor_cols: list, response: str = "thetadot") -> pd.DataFrame:
    """Per-subject, per-bout leave-one-bout-out OLS, no intercept -- frame-level output.

    Same fit as run_loo_discrete (training excludes the held-out bout, restricted
    to that bout's subject), but returns one row per held-out frame -- carrying
    that frame's own Phi/dist/speed and (y, y_hat) -- instead of one row per bout
    with a bout-median VAF and bout-median condition. Lets callers bin by each
    frame's own instantaneous condition instead of a bout-wide summary, which
    matters because Phi/dist/speed all vary substantially within a bout.
    """
    rows = []
    bout_groups = list(df.groupby(["avi", "event ind"]))
    for (avi, event_ind), grp in tqdm.tqdm(bout_groups, desc=f"LOO discrete frames {predictor_cols}"):
        subj  = grp["subject"].iloc[0]
        mask  = (df["avi"] == avi) & (df["event ind"] == event_ind)
        train = df[(df["subject"] == subj) & ~mask].dropna(subset=predictor_cols + [response])
        test  = grp.dropna(subset=predictor_cols + [response, "Phi", "dist", "speed"])
        if len(train) < 20 or len(test) < 5:
            continue
        reg   = LinearRegression(fit_intercept=False).fit(
                    train[predictor_cols].values, train[response].values)
        y_hat = reg.predict(test[predictor_cols].values)
        rows.append(pd.DataFrame({
            "avi":       avi,
            "event ind": event_ind,
            "subject":   subj,
            "y":         test[response].values,
            "y_hat":     y_hat,
            "Phi":       test["Phi"].values,
            "dist":      test["dist"].values,
            "speed":     test["speed"].values,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bootstrap_vaf_df(frame_pool: pd.DataFrame, group_cols: list, n_boot: int = 500, seed: int = 0,
                      diagnostics=None) -> pd.DataFrame:
    """Two-stage cluster bootstrap VAF (resample subjects, then (avi, event
    ind) bouts within subject -- see superiorcolliculus.analysis.
    two_stage_cluster_bootstrap), grouped by group_cols + model. One row per
    bootstrap replicate per group -- long-format, e.g. for a paired
    VAF_PP-vs-VAF_PN scatter."""
    by = group_cols + ["model"]
    rows = []
    for key, sub in frame_pool.groupby(by, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        diag_key = dict(zip(by, keys))
        vafs = two_stage_cluster_bootstrap(
            sub, group_col="subject", bout_cols=("avi", "event ind"), value_cols=("y", "y_hat"),
            metric_fn=compute_vaf, n_boot=n_boot, seed=seed, diagnostics=diagnostics, diag_key=diag_key,
        )
        if len(vafs) == 0:
            continue
        row = dict(zip(by, keys))
        row["vaf"] = vafs
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def pair_by_replicate(boot_df: pd.DataFrame, bin_col: str, order, model_col: str = "model",
                       model_names=("PP", "PN")) -> pd.DataFrame:
    """Zip two models' raw per-replicate VAF arrays by position (bootstrap
    replicate index), not by any explicit join key -- bootstrap_vaf_df/
    two_stage_cluster_bootstrap write each bin's `n_boot` replicates into
    `row["vaf"]` in the same b=0..n_boot-1 order they were generated in, so
    position IS the pairing. `model_names`: the two model_col values to pair,
    in (first, second) order -- output columns are named
    `vaf_{model_names[0].lower()}`/`vaf_{model_names[1].lower()}`."""
    m0, m1 = model_names
    col0, col1 = f"vaf_{m0.lower()}", f"vaf_{m1.lower()}"
    frames = []
    for b in order:
        v0 = boot_df.loc[(boot_df[bin_col] == b) & (boot_df[model_col] == m0), "vaf"].values
        v1 = boot_df.loc[(boot_df[bin_col] == b) & (boot_df[model_col] == m1), "vaf"].values
        n = min(len(v0), len(v1))
        if n == 0:
            continue
        frames.append(pd.DataFrame({bin_col: b, col0: v0[:n], col1: v1[:n]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[bin_col, col0, col1])


def _diag_lookup(diag_list, bin_col, bin_val):
    """Pull each model's (n_animals, frame_budget_cap) for one bin out of a
    bootstrap-diagnostics list, for reporting alongside the p-value."""
    out = {}
    for d in diag_list:
        if d["bin"].get(bin_col) == bin_val:
            out[d["bin"].get("model")] = (d["n_animals"], d["min_frames"])
    return out


def compute_bootstrap_significance(paired_df: pd.DataFrame, diag_list, bin_col: str, order, column_name: str,
                                    value_cols=("vaf_pp", "vaf_pn")) -> pd.DataFrame:
    """Per-bin paired-bootstrap significance (superiorcolliculus.analysis.
    paired_bootstrap_test) plus each model's bootstrap sample-size diagnostics,
    one row per bin in `order`. Holm-correct the returned `p` column across
    bins yourself (see superiorcolliculus.analysis.holm_correction) -- this
    function only computes the raw per-bin p-values."""
    col0, col1 = value_cols
    rows = []
    for b in order:
        sub = paired_df[paired_df[bin_col] == b]
        res = paired_bootstrap_test(sub[col0].values, sub[col1].values)
        res["column"] = column_name
        res["bin"] = b
        diag = _diag_lookup(diag_list, bin_col, b)
        n_pp, min_pp = diag.get("PP", (np.nan, np.nan))
        n_pn, min_pn = diag.get("PN", (np.nan, np.nan))
        res.update(n_animals_pp=n_pp, min_frames_pp=min_pp, n_animals_pn=n_pn, min_frames_pn=min_pn)
        rows.append(res)
    return pd.DataFrame(rows)
