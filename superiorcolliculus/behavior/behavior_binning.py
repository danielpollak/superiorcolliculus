"""behavior_binning.py

Load tautau_df, split by behavioral mode, and bin by any continuous variable.
Saves intermediate results wherever the caller directs (see `output_dir`
below) so downstream analysis cells can be re-run without re-processing the
full dataset.

Typical usage
-------------
>>> import superiorcolliculus.behavior.behavior_binning as bb
>>> tautau_df, tau_df, bout_df = bb.load_tautau_df("data/behavior/tautau_df_0.pkl")
>>> binned = bb.run_binning_pipeline(tautau_df, output_dir="data/behavior")
>>> speed_bins = bb.load_binned("data/behavior/tangent_speed_bins.pkl")

No default data path is hardcoded here (this module is part of the installed
superiorcolliculus package, not the main repo, so it has no built-in notion
of any one repo's checkout location) -- callers always pass their own path.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES     = ["tangent", "BD", "HD"]
INTERVAL  = 5     # cm/s (or cm) per bin
MIN_COUNT = 20    # minimum rows per bin; bins below this are excluded
SPEED_MAX = 50    # upper edge of last speed bin (cm/s)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tautau_df(path: Path):
    """Load tautau_df.pkl and return three views.

    Returns
    -------
    tautau_df : full DataFrame (all tau × tau1 combinations)
    tau_df    : rows where tau1 == 0  (PN analysis: varies tau)
    bout_df   : rows where tau == 0 AND tau1 == 0  (zero-delay reference)
    """
    with open(path, "rb") as f:
        tautau_df = pickle.load(f)
    tau_df  = tautau_df.loc[tautau_df["tau1"] == 0].copy()
    bout_df = tautau_df.loc[(tautau_df["tau1"] == 0) & (tautau_df["tau"] == 0)].copy()
    return tautau_df, tau_df, bout_df


# ---------------------------------------------------------------------------
# Splitting and binning
# ---------------------------------------------------------------------------

def split_by_mode(df: pd.DataFrame) -> dict:
    """Partition df on the 'mode' column.

    Returns
    -------
    dict mapping mode string → sub-DataFrame (copy).
    """
    return {mode: df.loc[df["mode"] == mode].copy() for mode in df["mode"].unique()}


def bin_by_variable(
    df: pd.DataFrame,
    variable: str = "speed",
    interval: float = INTERVAL,
    min_count: int = MIN_COUNT,
    v_max: float = SPEED_MAX,
) -> dict:
    """Bin df by a continuous variable into equal-width intervals.

    Replicates the notebook loop exactly (range(interval, v_max+1, interval),
    breaks when a bin has fewer than min_count rows).

    Parameters
    ----------
    df        : DataFrame to bin
    variable  : column name.  For distance, add the column before calling:
                    df['distance'] = np.abs(df['pos'] - df['target'])
    interval  : bin width
    min_count : bins with fewer rows than this are excluded AND the loop stops
                (matching the notebook's `break` behaviour — sparse high-speed
                bins are dropped once the first under-populated bin is hit)
    v_max     : upper edge of the last bin (inclusive)

    Returns
    -------
    dict mapping "{lo}-{hi}" → sub-DataFrame (copy).
    """
    bins = {}
    for hi in range(int(interval), int(v_max) + 1, int(interval)):
        lo = hi - interval
        sub = df.loc[(df[variable] > lo) & (df[variable] <= hi)].copy()
        if len(sub) < min_count:
            break
        bins[f"{int(lo)}-{int(hi)}"] = sub
    return bins


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_binned(data: dict, filepath: Path) -> None:
    """Pickle data to filepath, creating parent directories as needed."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_binned(filepath: Path) -> dict:
    """Load and return a pickled dict from filepath."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_binning_pipeline(
    tautau_df: pd.DataFrame,
    output_dir: Path,
    modes: list = None,
    variables: list = None,
    interval: float = INTERVAL,
    min_count: int = MIN_COUNT,
    v_max: float = SPEED_MAX,
) -> dict:
    """Split and bin for all mode × variable combinations; save to disk.

    For distance binning, pass variables=['distance'] and make sure the
    DataFrame contains a 'distance' column:
        tautau_df['distance'] = np.abs(tautau_df['pos'] - tautau_df['target'])

    Parameters
    ----------
    tautau_df  : full source DataFrame
    modes      : list of mode strings; defaults to MODES
    variables  : list of column names to bin on; defaults to ['speed']
    output_dir : where to write {mode}_{variable}_bins.pkl
    interval, min_count, v_max : forwarded to bin_by_variable

    Returns
    -------
    Nested dict: result[mode][variable] = {bin_label: sub_df}
    """
    if modes is None:
        modes = MODES
    if variables is None:
        variables = ["speed"]

    result = {}
    for mode in modes:
        mode_df = tautau_df.loc[tautau_df["mode"] == mode].copy()
        result[mode] = {}
        for variable in variables:
            bins = bin_by_variable(
                mode_df, variable=variable,
                interval=interval, min_count=min_count, v_max=v_max,
            )
            result[mode][variable] = bins
            out_path = Path(output_dir) / f"{mode}_{variable}_bins.pkl"
            save_binned(bins, out_path)
            print(f"  saved {len(bins)} bins → {out_path.name}")
    return result
