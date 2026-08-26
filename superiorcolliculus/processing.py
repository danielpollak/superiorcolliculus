import os
import numpy as np

import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
import warnings

import scipy.ndimage as nd
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt
import scipy.io
from numba import njit

from .analysis import *

conversion_rate = 820 / 18 / 2.54  # pixels per cm


def interpolate_cubic_spline(data):
    # Indices of valid (non-NaN) and invalid (NaN) values
    valid_mask = ~np.isnan(data)
    invalid_mask = np.isnan(data)

    # Extract the valid x and y values
    x_valid = np.arange(len(data))[valid_mask]
    y_valid = data[valid_mask]

    # Create cubic spline interpolation
    cubic_spline = CubicSpline(x_valid, y_valid)

    # Interpolate the missing values (NaNs)
    x_invalid = np.arange(len(data))[invalid_mask]
    data[invalid_mask] = cubic_spline(x_invalid)
    return data


def smooth(arr, sigma=2):
    """Just uses scipy's gaussian filter for non-angle data

    Does different things if arr is complex or not.
    Parameters
    ----------
    arr: np.array, can be complex
    sigma: int or float sigma for gaussian smoothing
    """
    return nd.gaussian_filter1d(arr, sigma)


def semiflatten(x, d=0):
    """Move axis d to the last position and collapse all remaining axes into one.

    Returns (y, s) where y is 2-D (K, L) with L = x.shape[d] and K = product
    of all other dimensions.  s is an opaque token for semiunflatten.
    """
    x = np.asarray(x)
    d = d % x.ndim
    rest = tuple(s for i, s in enumerate(x.shape) if i != d)
    y = np.moveaxis(x, d, -1).reshape(-1, x.shape[d])
    return y, (d, np.array(rest, dtype=int))


def semiunflatten(y, s):
    """Inverse of semiflatten: restore original shape and axis order."""
    d, rest = s
    x = y.reshape((*rest, y.shape[-1]))
    return np.moveaxis(x, -1, d)


def fupsample(x, n, t=None, axis=0, taper=0.0):
    """Fourier (sinc) upsampling by integer factor n along a given axis.

    Inserts n-1 evenly-spaced samples between each pair of original samples by
    zero-padding the DFT spectrum.  The result is a band-limited interpolant
    that passes exactly through all original sample values.

    Parameters
    ----------
    x : array-like
        Input signal (real or complex).
    n : int
        Upsampling factor (n=1 returns x unchanged).
    t : 1-D array, optional
        Time-stamps of x along `axis`.  If given, returns interpolated
        time-stamps instead of pseudo-indices.
    axis : int
        Axis along which to upsample (default 0).
    taper : float in [0, 1)
        Fraction of the half-spectrum to roll off with a raised-cosine window
        before zero-padding.  Attenuates the highest-frequency content to
        reduce Gibbs ringing without distorting time-domain signal values.
        0 (default) = pure sinc interpolation; ~0.1–0.2 is typically enough.

    Returns
    -------
    y : ndarray
        Upsampled signal with shape[axis] == x.shape[axis] * n.
    idx_or_t : 1-D ndarray
        Pseudo-indices (0, 1/n, 2/n, …) or interpolated time-stamps.
    """
    x = np.asarray(x)
    y, s = semiflatten(x, axis)  # y: (K, L), axis d is now the last dim
    K, L = y.shape

    if n == 1:
        return semiunflatten(y, s), (t if t is not None else np.arange(L))

    f = np.fft.fft(y, axis=1)
    half = L // 2

    if taper > 0:
        # Raised-cosine rolloff applied to the K bins nearest the Nyquist
        # boundary on both sides of the spectrum.  This smooths the hard
        # rectangular cutoff that causes Gibbs ringing, without touching
        # any time-domain values directly.
        K_tap = min(max(1, int(half * taper)), half - 1)
        # rolloff[0] = 1, rolloff[-1] ≈ 0  (falls smoothly over K_tap bins)
        rolloff = 0.5 * (1 + np.cos(np.pi * np.arange(K_tap) / K_tap))
        f[:, half - K_tap:half] *= rolloff          # positive freqs near Nyquist
        f[:, [half]] = 0                             # zero Nyquist itself
        f[:, half + 1:half + 1 + K_tap] *= rolloff[::-1]  # mirror: negative freqs

    if n % 2:
        # Odd upsampling factor: the Nyquist bin has no symmetric counterpart
        # in the new grid, so split its energy equally between the two adjacent
        # bins on either side of the zero-pad block.
        nyq = f[:, [half]] / 2
        pad = np.zeros((K, L * (n - 1) - 1), dtype=f.dtype)
        f = np.concatenate([f[:, :half], nyq, pad, nyq, f[:, half + 1:]], axis=1)
    else:
        # Even upsampling factor: the original Nyquist bin maps cleanly to the
        # Nyquist of the upsampled grid; insert zeros between positive and
        # negative frequency halves.
        pad = np.zeros((K, L * (n - 1)), dtype=f.dtype)
        f = np.concatenate([f[:, :half], pad, f[:, half:]], axis=1)

    # For real inputs, the imaginary residual is floating-point noise; discard it.
    # For complex inputs (e.g. x + j*y position phasors), keep the full result.
    y_up = np.fft.ifft(f, axis=1) * n
    if not np.iscomplexobj(x):
        y_up = y_up.real
    idx = np.arange(L * n) / n
    y_up = semiunflatten(y_up, s)

    if t is None:
        return y_up, idx
    return y_up, np.interp(idx, np.arange(len(t)), t)


def medfilt_smooth(arr, MEDFILT_KERNEL_SIZE=5):
    """"""
    assert np.iscomplexobj(arr), "Array must be complex"
    return medfilt(arr.real, MEDFILT_KERNEL_SIZE) + medfilt(arr.imag, MEDFILT_KERNEL_SIZE)*1j


def build_lag_matrix(data, L, mode="causal", dtype=np.float64):
    """Build a causal, acausal, or symmetric lag matrix.

    Accepts either a single timeseries (T,) or a stack of timeseries (N, T)
    — e.g. one row per spike train or per behavioral signal.

    mode="causal"    X[t, i*n_lags + c] = data[i, t - c]
                     c in [0, L-1]; c=0 current, c>0 past frames.
                     shape (T, N*L).

    mode="acausal"   X[t, i*n_lags + c] = data[i, t + c]
                     c in [0, L-1]; c=0 current, c>0 future frames.
                     shape (T, N*L).

    mode="symmetric" X[t, i*n_lags + c] = data[i, t - (c - L)]
                     c in [0, 2L]; c=0 -> L frames future, c=L -> current,
                     c=2L -> L frames past. shape (T, N*(2L+1)).

    Causal and acausal produce identical shapes (T, N*L), so metrics computed
    on them are directly comparable. Symmetric has twice as many features
    per row. Boundary conditions: zero-padding.

    For 1-D input (T,), N=1 and the output is simply (T, n_lags) with no
    row-interleaving — this is the scalar/single-signal case (e.g. one
    behavioral timeseries).

    Parameters
    ----------
    data  : (T,) or (N, T) array-like
    L     : int — number of lags (half-width for symmetric)
    mode  : {"causal", "acausal", "symmetric"}
    dtype : output dtype (default float64; pass np.float32 for large
            multi-row matrices where memory matters, e.g. spike trains)

    Returns
    -------
    X      : (T, N*n_lags) (or (T, n_lags) for 1-D input) array of `dtype`
    n_lags : int — L for causal/acausal, 2L+1 for symmetric
    """
    data = np.asarray(data, dtype=dtype)
    if data.ndim == 1:
        data = data[None, :]
    N, T = data.shape

    if mode == "symmetric":
        n_lags = 2 * L + 1
        X = np.zeros((T, N * n_lags), dtype=dtype)
        for c in range(n_lags):
            offset = c - L          # negative=future, zero=current, positive=past
            if offset == 0:
                X[:, c::n_lags] = data.T
            elif offset > 0:        # past: data[i, t - offset]
                X[offset:, c::n_lags] = data[:, :-offset].T
            else:                   # future: data[i, t + |offset|]
                X[:T + offset, c::n_lags] = data[:, -offset:].T
    else:
        n_lags = L
        X = np.zeros((T, N * n_lags), dtype=dtype)
        for c in range(L):
            if c == 0:
                X[:, c::n_lags] = data.T
            elif mode == "causal":  # past: data[i, t - c]
                X[c:, c::n_lags] = data[:, :-c].T
            else:                   # future: data[i, t + c]
                X[:T - c, c::n_lags] = data[:, c:].T
    return X, n_lags


def phasor(angle):
    """
    Parameters
    ----------
    angle: Angle in degrees
    
    Returns
    -------
    np.complex128 """
    return np.exp(1j * np.deg2rad(angle))

@njit
def get_angular_difference_scalar(u, v):
    eps = 1e-12
    unorm = np.abs(u)
    vnorm = np.abs(v)
    if unorm < eps or vnorm < eps:
        return 0.0
    u /= unorm
    v /= vnorm
    return (180.0 / np.pi) * np.angle(u / v)

@njit
def get_angular_difference_array(u, v):
    """Returns the angular difference between two phasors in degrees.
    Parameters
    ----------
    u, v: np.array of phasors

    Returns
    -------
    np.array of angular differences in degrees
    """
    eps = 1e-12
    out = np.empty(len(u))
    for i in range(len(u)):
        unorm = np.abs(u[i])
        vnorm = np.abs(v[i])
        if unorm < eps or vnorm < eps:
            out[i] = 0.0
        else:
            out[i] = (180.0 / np.pi) * np.angle(u[i] / unorm / (v[i] / vnorm))
    return out


def dangle_dt(angles):
    """Returns the derivative of angles with respect to time
    Parameters
    ----------
    angles: np.array
        Phasors
    Returns
    -------
    np.array (angles in degrees)
        Derivative of angles with respect to time"""
    assert np.iscomplexobj(angles), "Angles must be complex, not degrees"
    assert len(angles) > 1, "Angles must have at least 2 elements"
    diffs = get_angular_difference_array(angles[1:], angles[:-1])
    
    return np.concatenate(([diffs[0]], diffs))


# Generate histogram of speeds
def get_speed_timeseries(arr:np.complex128, fps):
    """gets a timeseries of average speed with intervals of `interval`
    Parameters
    ----------
    arr: (np.complex) xy positions
    fps: (float or array) frames per second of the video
    """

    # Inter frame interval
    IFI = 1 / fps # s

    # Inter frame distances
    IFD = np.abs(np.diff(arr, prepend=arr[0])) / conversion_rate  # cm
    
    # Calculate speeds
    speeds = IFD / IFI # pixels/s
    return speeds


def read_DLC_csv(path):
    """Reads DLC csv and returns a pandas dataframe"""
    # Get length of header
    sniffer_df = pd.read_csv(path, header=None, low_memory=False)

    # Number of elements in the first 9 elements of the first column
    # This is to determine how many header rows there are before it starts doing '0', '1', '2', etc.
    header_len = sum([len(elem) > 1 for elem in sniffer_df.iloc[:9,0]])

    # Read csv to dataframe
    df = pd.read_csv(path, header=list(range(header_len)), encoding="latin-1")

    # Columns have only relevant info
    new_cols = [None] * len(df.columns)
    for col_i, column in enumerate(df.columns):
        new_cols[col_i] = tuple(
            [df.columns[col_i][tuple_i] for tuple_i in range(len(df.columns[col_i])) if tuple_i > 0])

    df.columns = new_cols
    return df



def boris_to_array(b_df:pd.DataFrame, n_frames:int):
    """ Convert a Boris DataFrame to an array of behaviors.
    
    Parameters
    ----------
    b_df : pd.DataFrame
        DataFrame containing Boris event data with columns "Image index", "Behavior type", and "Behavior".
    n_frames : int
        Total number of frames in the video.
    Returns
    -------
    np.ndarray
        A 1D numpy array where each element corresponds to a behavior at each time point.
    """
    # Assert even number of rows
    assert b_df.shape[0] % 2 == 0, "Boris DataFrame should have an even number of rows, got {}".format(b_df.shape[0])
    
    # Assert monotonicity 
    assert sorted(b_df["Image index"]) == list(b_df["Image index"]), "Image index is not monotonic"

    # Assert there are no zero intervals. That would make things trickier.
    # assert all(np.diff(b_df["Image index"]) > 0), f"There are zero Image index intervals in the Image index at {np.where(np.diff(b_df['Image index']) <= 0)[0]}"
    
    # Init output
    arr_l = [["other"] * int(b_df.iloc[0]["Image index"])]

    for i in np.arange(0, len(b_df)-2, 2):
        bh_arr = boris_to_array_helper(b_df, i)
        
        # Append the behavior for the duration
        arr_l.append(bh_arr)

        # Append other for the interval between behaviors
        arr_l.append(
            np.array(["other"] * int( 
                b_df.iloc[i + 2]["Image index"] -  b_df.iloc[i + 1]["Image index"]
            ), dtype=object)
        )

    # Now remember, i + 2 at the end is the SECOND TO LAST ROW. Let's add the last interval together, shall we?
    bh_arr = boris_to_array_helper(b_df, len(b_df) - 2)

    # Append the behavior for the duration
    arr_l.append(bh_arr)

    curr_len = len(np.concatenate(arr_l))
    assert curr_len <= n_frames, "Concatenated array length exceeds n_frames: {} > {}".format(len(np.concatenate(arr_l)), n_frames)

    if curr_len < n_frames:
        # If the last behavior does not fill the array, fill it with "other"
        arr_l.append(
            np.array(["other"] * (n_frames - curr_len), dtype=object)
        )
    return np.concatenate(arr_l)


def boris_to_array_helper(b_df:pd.DataFrame, i:int):
    """ Helper function to convert a pair of rows in a Boris DataFrame to an array of behaviors."""
    start_row, stop_row = b_df.iloc[i], b_df.iloc[i+1]

    assert start_row["Behavior type"] == "START"
    assert stop_row["Behavior type"] == "STOP", "Start and stop rows do not match at index {}".format(i + 2)
    assert start_row["Behavior"] == stop_row["Behavior"]
    
    duration = int(stop_row["Image index"] - start_row["Image index"])
    bh = start_row["Behavior"]

    return np.array([bh] * duration, dtype=object)


def boris_windows_from_arr(boris_arr):
    """ Get the offsets of behaviors from a boris array.
    Essentially the opposite of boris_to_array_helper, but it's
    useful because sometimes you have basically cast away the boris dataframe. 

    Parameters
    ----------
    boris_arr: np.array of behaviors, where each behavior is an integer
    
    Returns:
    --------
    windows: list of tuples (start, stop, behavior)
    """
    # Get offsets for behaviors
    offset_d = {bh:[] for bh in np.unique(boris_arr)}
    for bh in np.unique(boris_arr):
        bh_inds = np.where(boris_arr == bh)[0]
        non_bh_inds = np.where(boris_arr != bh)[0]

        offsets = np.where(np.diff(bh_inds) > 1)

        # Actual indicies where behavior STOPS
        offset_d[bh].extend(bh_inds[offsets[0]])

    # Concatenate and sort offsets and behaviors
    offsets = []
    behaviors = []
    for bh, offset_l in offset_d.items():
        offsets.extend(offset_l)
        behaviors.extend([bh] * len(offset_l))
    
    bh_inds = np.argsort(offsets)
    sorted_offsets = np.array(offsets)[bh_inds]
    sorted_behaviors = np.array(behaviors)[bh_inds]
    assert len(sorted_offsets) == len(sorted_behaviors), "Offsets and behaviors lengths do not match"

    # Get windows of behaviors
    windows = [(0, sorted_offsets[0], sorted_behaviors[0])]
    for i in range(0,len(sorted_offsets)-1):
        behavior= sorted_behaviors[i]
        windows.append((sorted_offsets[i], sorted_offsets[i+1], behavior))

    # assert len(windows) - 1 == len(sorted_offsets), "Windows and offsets lengths do not match"
    print(len(windows), len(sorted_offsets))
    return windows 


def remove_behaviors(df, behaviors):
    if len(behaviors) == 0:
        raise Exception("behaviors must be a list of behaviors to remove")
    
    df = remove_behavior(df, behaviors[0])
    if len(behaviors) == 1:
        return df
    else:
        return remove_behaviors(df, behaviors[1:])


def remove_behavior(df, behavior):
    "remove entries from df with Behavior field values of `behavior`"
    return df[df["Behavior"] != behavior]


# Preprocessing
def clean_boris(event_df):
    # Clean up nans in Image Index
    for row_i in range(len(event_df)):
        if np.isnan(event_df.iloc[row_i]["Image index"]):
            event_df.at[row_i, "Image index"] = event_df.iloc[row_i + 1]["Image index"].copy()

    return event_df


def match_avi_boris_dlc(path):
    """
    Match avi files with boris, dlc files, and cv2 files
    Parameters
    ----------
    path: (str) pattern corresponding to all avi files in a folder or folders
    Returns
    -------
    list of tuples: (avi_file, boris_tsv_file, DLC_csv_file, cv2_pursuit_target
    """
    # Holder var
    matched_file_l = []
    all_files = glob.glob(path)
    for avi_file in glob.glob(path+".avi"):
        # get pattern
        pattern = avi_file.split(".avi")[0]

        # Get tsvs
        boris_tsv_files = [file for file in all_files if pattern in file and ((".avi.tsv" in file) or ("_boris_events.tsv" in file))]
        boris_tsv_file = boris_tsv_files[0] if len(boris_tsv_files) > 0 else None

        # Get DLC csvs
        DLC_csv_files = [file for file in all_files if pattern in file and "el.csv" in file]
        DLC_csv_file = DLC_csv_files[0] if len(DLC_csv_files) > 0 else None

        # Get pursuit target cv2 files
        cv2_pursuit_target_files = [file for file in all_files if pattern in file and "_keypoints.csv" in file]
        cv2_pursuit_target_file = cv2_pursuit_target_files[0] if len(cv2_pursuit_target_files) > 0 else None

        matched_file_l.append((
            avi_file,
            boris_tsv_file,
            DLC_csv_file,
            cv2_pursuit_target_file))

    return matched_file_l


def extract_clips(input_file, output_folder=r"\\datanas\family\SC\Figs",
                  csv_output_folder=r"\\datanas\family\SC\data_agg\behavior"):
    """"""
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 200
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else: 
        detector = cv2.SimpleBlobDetector_create(params)

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if output_folder is not None:
        video_output_file = f"{output_folder}/{os.path.basename(input_file[:-4])}.mp4"
        # Initialize VideoWriter for the subclip
        out = cv2.VideoWriter(video_output_file, fourcc, fps, (width, height))

    keypoint_l = [None] * int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret = 1
    counter = -1
    with tqdm.tqdm(total=len(keypoint_l)) as pbar:
        while 1:
            # Read until end of video
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1
            pbar.update(1)

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Crop out inconvenient screw
            gray = gray[:,:1000]

            # Blob detection
            keypoints = detector.detect(gray)

            # Append keypoints to list
            if len(keypoints) > 0:
                # Empirically it looks like the keypoints are sorted by rightmost extent, descending.
                keypoint_l[counter] = keypoints[0].pt
            else:
                keypoint_l[counter] = (np.nan, np.nan)

            if output_folder is not None:
                # Draw blobs on frame
                frame = cv2.drawKeypoints(
                    frame, keypoints, np.array([]), (0,0,255),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # Flip frame so it matches data
                frame = np.flipud(frame)
                out.write(frame)

    # Release video writer
    if output_folder is not None:
        out.release()

    # Release current video capture object
    cap.release()

    # Close progressbar
    pbar.close()

    # Save keypoints to dataframe and to csv
    keypoint_df = pd.DataFrame(keypoint_l, columns=["x", "y"])


    return keypoint_df


def flatten_l(xss):
    """Flattens a list"""
    return [x for xs in xss for x in xs]


def rename_df_cols(df):
    """Renames dataframe columns so that they are only two items long"""
    df.columns = [tuple(col[-2:]) for col in df.columns]
    return df


def get_likelihood(df, bodypart):
    """
    Parameters
    ----------
    df: (pandas.DataFrame) DLC dataframe, parsed with read_DLC_csv
    bodypart: (str) Bodypart to get
    
    Returns
    -------
    np.array (complex numbers)
    """
    '''Sometimes dlc dfs have three or two elements in their column names.
    Here I will simply check the second to last element'''
    # First, make all columns equal to the original values last two elements.
    df = rename_df_cols(df)
    L = df[(bodypart, "likelihood")]
    return L


def get_bodypart(df, bodypart):
    """
    Parameters
    ----------
    df: (pandas.DataFrame) DLC dataframe, parsed with read_DLC_csv
    bodypart: (str) Bodypart to get
    
    Returns
    -------
    np.array (complex numbers)
    """
    '''Sometimes dlc dfs have three or two elements in their column names.
    Here I will simply check the second to last element'''
    # First, make all columns equal to the original values last two elements.
    L = get_likelihood(df, bodypart)
    arr = df[[(bodypart, "x"), (bodypart, "y")]].values
    return arr[:,0] + 1j * arr[:,1], L


def clean_target_positions(df):
    df.columns = [tuple(col[-2:]) for col in df.columns]

    mag_x, mag_y = df[('mag', 'x')].values, df[('mag', 'y')].values
    roach_x, roach_y = df[('roach', 'x')].values, df[('roach', 'y')].values
    roach_L, mag_L = df[('roach', 'likelihood')].values, df[('mag', 'likelihood')].values

    # target_x, target_y = roach_x.copy(), roach_y.copy()
    target_x, target_y = mag_x.copy(), mag_y.copy()
    
    # If the distance between two points is greater than 100, it's an outlier
    dpos_dt = (np.diff(target_x, prepend=target_x[0]) ** 2 + np.diff(target_y, prepend=target_y[0]))**0.5
    mag_roach_dist = np.sqrt((mag_x - roach_x) **2 + (mag_y - roach_y) ** 2)

    aberrant_inds = np.where(
        (dpos_dt > 30) |
        # (mag_roach_dist > 40) | 
        ((roach_L < .9) & (mag_L < .9)))[0]
    
    target_x[aberrant_inds], target_y[aberrant_inds] = np.nan, np.nan

    for _ in range(3):

        # Now dilate the nans
        dilated_mask = nd.binary_dilation(np.isnan(target_x), structure=np.ones(2))
        target_x[np.where(dilated_mask)[0]], target_y[np.where(dilated_mask)[0]] = np.nan, np.nan

    # target_x, target_y = scipy.signal.medfilt(target_x, 5), scipy.signal.medfilt(target_y, 5)
    return target_x, target_y


def get_relevant_measures(dlc_csv, conversion_rate):
    """
    Parameters:
    -----------
    dlc_csv: str, path to dlc output
    conversion_rate: float, pixels per cm
    use_cv2_target_keypoints: bool, whether to use cv2 target keypoints

    Returns
    -------
    LOS: np.array, phasor of target position
    head_vector: np.array, phasor of head position
    body_vector: np.array, phasor of body position
    ego_angle: np.array, egocentric head angle
    target_bearing: np.array, head-target angle, in degs
    head_angle: np.array, allocentric head angle, in degs
    dist: np.array, distance between head and target
    (neck_x, neck_y): tuple, x and y positions of neck
    (snout_x, snout_y): tuple, x and y positions of snout
    (base_x, base_y): tuple, x and y positions of base
    (target_x, target_y): tuple, x and y positions of target
    df: pandas.DataFrame, whole DLC dataframe
    """

    df = read_DLC_csv(dlc_csv)

    # Get mouse and target positions
    neck =  interpolate_cubic_spline(get_bodypart(df,   "spine1")[0])
    snout = interpolate_cubic_spline(get_bodypart(df,    "snout")[0])
    base =  interpolate_cubic_spline(get_bodypart(df, "tailbase")[0])

    # Get target positions
    # Remove target outliers by using the mag, and then for when the likelihood of the roach is greater than the mag, use the roach.
    if ("mag", "x") in df.columns.tolist():
        target, _ = get_bodypart(df, "mag")

        # If the likelihood of the roach is greater than the mag, use the roach
        roach_L, mag_L = get_likelihood(df, "roach"), get_likelihood(df, "mag")
        swap_in_roach_inds = np.where(roach_L > mag_L)[0]
        swap_in_roach_vals = get_bodypart(df, "roach")[0][swap_in_roach_inds]
        target[swap_in_roach_inds] = swap_in_roach_vals

    elif ("puck", "x") in df.columns.tolist():
        target, L = get_bodypart(df, "puck")

    else:
        raise ValueError("No target found in DLC data. Please check the DLC csv file.")
    

    # If the distance between two points is greater than 100, it's an outlier
    def remove_outliers(arr, thres=100):
        aberrant_inds = np.where(np.abs(np.diff(arr)) > thres)[0]
        arr[aberrant_inds] = np.nan
        arr = interpolate_cubic_spline(arr)
        return arr
    
    # remove outliers
    target = remove_outliers(target)
    neck = remove_outliers(neck)
    snout = remove_outliers(snout)
    base = remove_outliers(base)

    # # Smooth
    neck = smooth(neck)
    # neck = medfilt_smooth(neck)
    snout = smooth(snout)
    # snout = medfilt_smooth(snout)
    base = smooth(base)
    # base = medfilt_smooth(base)
    target = smooth(target)
    # target = medfilt_smooth(target)

    # suppress warnings using with statement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Get distances
        dist = np.abs(neck - target) / conversion_rate # complex
       
        # Get angles
        LOS = target - neck
        head_vector = snout - neck 
        body_vector = neck - base

        # Egocentric head angle
        ego_angle = get_angular_difference_array(head_vector, body_vector)

        # Allocentric head angle
        head_angle = np.angle(head_vector, deg=True)

        # head-target head angle
        target_bearing = get_angular_difference_array(LOS, head_vector)
        
    return {
        'LOS':LOS, 'head_vector':head_vector, 'body_vector':body_vector, 'ego_angle':ego_angle, 'target_bearing':target_bearing,
        'head_angle':head_angle, 'dist':dist, "neck": (neck.real, neck.imag),
        "snout": (snout.real, snout.imag), "base": (base.real, base.imag), "target":(target.real, target.imag), 'df':df
    }


def relevant_measures_to_tuples(rm_d):
    """Relevant measures were originally a tuple, but now there are too many of them.
    They have to be dictionaries now, but it is useful to be able to unpack them."""
    (
        LOS, head_vector, body_vector, ego_angle, target_bearing, head_angle, dist, 
        (neck_x, neck_y), (snout_x, snout_y), (base_x, base_y), (target_x, target_y), df
    ) = rm_d['LOS'], rm_d['head_vector'], rm_d['body_vector'], rm_d['ego_angle'], \
            rm_d['target_bearing'], rm_d['head_angle'], rm_d['dist'], \
            rm_d['neck'], rm_d['snout'], rm_d['base'], rm_d['target'], rm_d['df']
    
    return (LOS, head_vector, body_vector, ego_angle, target_bearing, head_angle, dist, 
        (neck_x, neck_y), (snout_x, snout_y), (base_x, base_y),  (target_x, target_y), df)

def get_boris_bout(boris_df, bout_ind, behavior="approach"):
    assert behavior in boris_df.Behavior.unique()
    behavior_df = boris_df.loc[boris_df.Behavior==behavior, :]
    bout = behavior_df.iloc[bout_ind * 2 : bout_ind * 2 + 2, :]
    return bout


def behavior_bout_generator(avi_boris_dlc, conversion_rate, behavior="approach", warnings=True):
    """
    """
    for avi_index, (avi_file, boris_tsv, dlc_csv, cv2_csv) in enumerate(avi_boris_dlc):
        # Identify missing data
        if boris_tsv is None:
            if warnings:
                print(f"{avi_index}th file missing boris data: {avi_file}")
            continue

        if dlc_csv is None:
            if warnings:
                print(f"{avi_index}th file missing DLC data: {avi_file}")
            continue

        if warnings and cv2_csv is None:
            if warnings:
                print(f"{avi_index}th file missing cv2 keypoint data: {avi_file}")

        # Open tsv
        boris_df = pd.read_csv(boris_tsv, sep='\t')

        # Clean boris
        boris_df = clean_boris(boris_df)
        
        # Remove irrelevant behaviors
        if behavior != "all":
            boris_df = boris_df.loc[boris_df["Behavior"] == behavior,:]


        
        relevant_measures = get_relevant_measures(dlc_csv, conversion_rate)

        (   LOS, head_vector, body_vector, ego_angle, target_bearing, head_angle, dist,
            (neck_x, neck_y), (snout_x, snout_y), (base_x, base_y), (target_x, target_y), df
        ) = relevant_measures_to_tuples(relevant_measures)

        for bout_index in np.arange(0, len(boris_df), 2):
            # Get csv of target positions
            win = boris_df.iloc[bout_index:bout_index+2]["Image index"].values.astype(int)
            
            n_frames = np.diff(win)[0]

            # Windowing DLC parameters win head angle, snout position, body vector, head vector
            win_head_angle = head_angle[win[0]:win[1]]
            win_snout_x = snout_x[win[0]:win[1]]
            win_snout_y = snout_y[win[0]:win[1]]
            win_body_vector = body_vector[win[0]:win[1]]
            win_head_vector = head_vector[win[0]:win[1]]

            # Target vector is just target position
            win_target_position = (target_x + 1j * target_y)[win[0]:win[1]]
            win_target_bearing = target_bearing[win[0]:win[1]]

            yield {
                'relevant_measures':relevant_measures, 'win_head_angle':win_head_angle,
                'win_snout':(win_snout_x, win_snout_y),
                'win_body_vector':win_body_vector, 'win_head_vector':win_head_vector, 'win_target_position':win_target_position,
                'win_target_bearing':win_target_bearing, 'win':win, 'n_frames':n_frames,
                'avi':avi_file, 'boris_tsv':boris_tsv, 'dlc_csv':dlc_csv, 'event_ind':bout_index//2, 'boris_df':boris_df
            }



def vectorized_schmidt_trigger(input_signal, lower_threshold, upper_threshold):
    """
    Vectorized Schmidt Trigger implementation using NumPy.
    Currently only finds falling edges

    :param input_signal: NumPy array of input values.
    :param lower_threshold: Lower threshold for switching.
    :param upper_threshold: Upper threshold for switching.
    :return: NumPy array of output states.
    """
    # Initialize the output array with False
    output = np.full_like(input_signal, False, dtype=bool)

    # Identify indices where input crosses the upper and lower thresholds
    upper_crossings = np.where(input_signal > upper_threshold)[0]
    lower_crossings = np.where(input_signal < lower_threshold)[0]
    
    '''
    # If rising edge, just swap the crossings
    if not falling:
        temp = upper_crossings
        upper_crossings = lower_crossings
        lower_crossings = temp
    '''
    # Toggle the state at each crossing
    for uc in tqdm.tqdm(upper_crossings):
        output[uc:] = True  # Set to True from this index onwards
        # Find the next lower crossing after this upper crossing
        next_lower_crossing = lower_crossings[lower_crossings > uc]
        if next_lower_crossing.size > 0:
            output[next_lower_crossing[0]:] = False  # Reset to False from the next lower crossing

    return output.astype(int)


def get_date_time(datetime_str):
    """Takes OEGUI datetime string and returns date and time"""
    date = datetime_str[:8]
    time = datetime_str[8:]
    # Get month, day, year and HMS
    month, day, year = date[:2], date[2:4], date[4:]
    hour, minute, second = time[:2], time[2:4], time[4:]
    return np.datetime64(f"{year}-{month}-{day} {hour}:{minute}:{second}")


def pretty_print_datetime(datetime_str):
    """Takes OEGUI datetime string and returns date and time"""
    month, day, year, hour, minute, second = get_date_time(datetime_str)


"""NPIX operations"""
def get_cluster_info(data_path):
    """Loads cluster information"""
    # Load the sorting data
    unit_df = pd.read_csv(data_path + "\\cluster_info.tsv", sep='\t')

    unit_df = unit_df.sort_values("ch")

    # Load channelmap data
    ChanMap = scipy.io.loadmat(
        r"C:\Users\dan\Documents\MATLAB\Kilosort-2.5\configFiles\neuropixPhase3B2_kilosortChanMap.mat")
    del ChanMap["__header__"], ChanMap["__version__"]
    del ChanMap["__globals__"], ChanMap["name"], ChanMap["chanMap"], ChanMap["connected"]
    probe_df = pd.DataFrame({key: val.flatten() for key, val in ChanMap.items()})
    probe_df.columns = ['ch', 'shankInd', 'xcoords', 'ycoords']

    # Annotate sorting data with channelmap data
    udf = pd.merge(unit_df, probe_df, how="left", on="ch")
    return udf



def extract_waveforms(recording, sorting, unit_df, pre=0.001, post=0.002, n_waveforms=100, label="good", sr=30_000, bounds=None):
    """
    Extract waveforms from a recording.

    Parameters:
    recording (spikeinterface.BaseRecording): The recording object.
    sorting (object): The sorting object.
    unit_id (int): The ID of the unit.
    pre (float, optional): The duration of the pre-spike window in seconds. Defaults to 0.001.
    post (float, optional): The duration of the post-spike window in seconds. Defaults to 0.002.
    n_waveforms (int, optional): The number of waveforms to extract. Defaults to 100.
    good (bool, optional): Flag to indicate whether to extract waveforms from good units only. Defaults to True.
    sr (int, optional): The sampling rate of the recording in Hz. Defaults to 30_000.
    bounds (tuple(int)): bounds in 

    Returns:
    dict: A dictionary containing the extracted waveforms, where the keys are unit IDs and the values are 2D arrays of waveforms.

    Raises:
    AssertionError: If the recording is not filtered.

    """
    assert recording.is_filtered()

    pre, post = pre * sr, post * sr
    id_column = "id" if "id" in unit_df.columns else "cluster_id"

    waveform_d = {row[id_column]: np.zeros((n_waveforms, int(pre+post))) for _, row in unit_df.iterrows() if row.group==label}
    unit_df = unit_df.loc[unit_df.group==label]

    for row_i, row in tqdm.tqdm(unit_df.iterrows(), total=len(unit_df)):
        spiketrain = sorting.get_unit_spike_train(unit_id=row[id_column]).flatten()
        if bounds is not None:
            spiketrain = spiketrain[(spiketrain > bounds[0]) & (spiketrain < bounds[1])]
            if len(spiketrain) == 0:
                continue
        
        # Randomly sample spikes
        sub_spiketrain = np.random.choice(spiketrain, n_waveforms, replace=True)

        for spike_i, spike in enumerate(sub_spiketrain):  
            waveform_d[row[id_column]][spike_i,:] = recording.get_traces(
                start_frame=int(spike-pre), end_frame=int(spike+post),
                channel_ids=[row.ch]
            ).T
    return waveform_d


def get_NP1():
    import probeinterface
    total=384 #, 960
    NP1 = probeinterface.generate_multi_columns_probe(
        4, num_contact_per_column=total//4, xpitch=16, ypitch=40,
        y_shift_per_column=[0,20,0,20], contact_shapes="square", contact_shape_params={"width":12},)
    return NP1


def plot_NP1(NP1, clean=True, ax=None):
    from probeinterface.plotting import plot_probe

    if ax is None:
        _, ax = plt.subplots()

    plot_probe(NP1, with_contact_id=True, title="NP1.0", ax=ax)
    
    if clean:
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_ylabel("")

        # removing bounding box on axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    return ax


def plot_units_on_probe(unit_df, NP1, waveform_d, normalize=True, ax=None, color=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,25))

    ax = plot_NP1(NP1, clean=False, ax=ax)

    id_col = "cluster_id" if "cluster_id" in unit_df.columns else "id"
    for id_i, (id, waveforms) in enumerate(waveform_d.items()):
        # Get channel coordinates (jitter them?)
        ch = unit_df.loc[unit_df[id_col]==id, "ch"].values[0]
        xcoords = unit_df.loc[unit_df[id_col]==id, "xcoords"].values[0]
        ycoords = unit_df.loc[unit_df[id_col]==id, "ycoords"].values[0]
        
        # Get the average waveform
        avg_waveform = np.mean(waveforms, axis=0)[10:80]

        # Normalize shape
        avg_waveform = avg_waveform 
        
        if normalize:
            avg_waveform = avg_waveform / np.max(avg_waveform) * 10


        x = np.linspace(0, 20, num=len(avg_waveform))        
        ax.plot(x + xcoords, avg_waveform.T + ycoords, linewidth=1, color=color)
        ax.plot(x + xcoords, waveforms.T + ycoords, linewidth=1, color=color)
    return ax


def parse_param_string(param_string):
    """
    Parses a parameter string from a Kilosort2 config file.
    Parameters
    ----------
    param_string : str
        The parameter string to parse, should be in form "[p1, p2]".
    Returns
    -------
    np.array
    """ 
    return np.fromstring(param_string.strip("()"), sep=" ") # []


