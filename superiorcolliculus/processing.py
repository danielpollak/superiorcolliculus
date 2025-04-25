import pandas as pd
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
import warnings
import os
import scipy.io
import scipy.ndimage as nd
from numba import njit

from .analysis import *

conversion_rate = 820 / 18 / 2.54  # pixels per cm

from scipy.interpolate import CubicSpline

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


def smooth(angle_arr, box_pts=8):
    """Maintains original length of array
    mode = "same" instead of full or valid because full produces an array of the wrong length, valid induces a phase shift, and
    "same" produces an array that is the wrong length but does not produce a phase shift. I then add back the first value
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(angle_arr, box, mode='same')
    return y_smooth[box_pts//2:-(box_pts//2)]


# def smooth(data, window_width=8):
#     """Edited from 
#     https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
#     in order for the output to be the same length as the input. The naive implementation cuts the first window_width-1
#     from the output array in terms of length. I add it back in by copying the first value window_width-1 times
#     """
#     cumsum_vec = np.nancumsum(np.insert(data, 0, 0)) 
#     ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
#     ma_vec_evened = np.concatenate((
#             np.ones((len(data)-len(ma_vec))//2) * ma_vec[0],
#             ma_vec,
#             np.ones((len(data)-len(ma_vec))//2 + 1) * ma_vec[-1]
#         ))

#     return ma_vec_evened[:-1]



def phasor(angle):
    """Angle in degrees"""
    return np.exp(1j * np.deg2rad(angle))

# @njit
# def get_angular_difference(u, v):
#     """
#     Parameters
#     ----------
#     u, v: np.complex128
#         Unit phasors representing angles
#     Returns
#     -------
#     np.array (angles, in degrees)
#         The angular difference between the phasors in degrees
#     """

#     # Compute the difference using division of complex numbers
#     eps = 1e-12
#     u_norm = np.abs(u)
#     v_norm = np.abs(v)
#     """old"""
#     if u_norm < eps or v_norm < eps:
#         return np.array([0.0])  # or np.nan, or raise a controlled exception
#     """old"""

#     """new"""
#     u_norm(np.where(u_norm < eps)) = eps
#     v_norm(np.where(v_norm < eps)) = eps
#     """new"""

#     u = u / u_norm
#     v = v / v_norm
#     # Calculate the angle of the resulting phasor in degrees
#     difference_phasor = u / v
#     angular_difference = (180.0 / np.pi) * np.angle(difference_phasor)

#     return angular_difference # deg
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
def get_speed_timeseries(arr_x, arr_y, interval=5, fps=30):
    """gets a timeseries of average speed with intervals of `interval`
    Parameters
    ----------
    arr_(x/y): (array) x/y positions
    interval: (int) number of frames to average over
    """

    inter_frame_interval = 1 / fps * interval

    inter_frame_distances = np.array([np.sqrt((arr_x[i] - arr_x[i+interval])**2 + (arr_y[i] - arr_y[i+interval])**2) for i in range(len(arr_x)-interval)])

    speeds = inter_frame_distances / inter_frame_interval # pixels/s
    speeds = speeds / conversion_rate # cm / s
    return speeds


def read_DLC_csv(path):
    """Reads DLC csv and returns a pandas dataframe"""
    # Read csv to dataframe
    df = pd.read_csv(path, header=[0, 1, 2, 3], encoding="latin-1")

    # Columns have only relevant info
    new_cols = [None] * len(df.columns)
    for col_i, column in enumerate(df.columns):
        new_cols[col_i] = tuple(
            [df.columns[col_i][tuple_i] for tuple_i in range(len(df.columns[col_i])) if tuple_i > 0])

    df.columns = new_cols
    return df


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


def get_bodypart(df, bodypart):
    """
    Parameters
    ----------
    df: (pandas.DataFrame) DLC dataframe
    bodypart: (str) Bodypart to get
    
    Returns
    -------
    np.array (complex numbers)
    """
    if ("mouse", bodypart, "x") in df.columns:
        arr = df[[("mouse", bodypart, "x"), ("mouse", bodypart, "y")]].values
    elif ("single", bodypart, "x") in df.columns:
        arr = df[[("single", bodypart, "x"), ("single", bodypart, "y")]].values
    else:
        raise Exception(f"Bodypart {bodypart} not found in dataframe")
    
    return arr[:,0] + 1j * arr[:,1]


def clean_target_positions(df):
    mag_x, mag_y = df[('single', 'mag', 'x')].values, df[('single', 'mag', 'y')].values
    roach_x, roach_y = df[('single', 'roach', 'x')].values, df[('single', 'roach', 'y')].values
    roach_L, mag_L = df[('single', 'roach', 'likelihood')].values, df[('single', 'mag', 'likelihood')].values

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

    target_x, target_y = scipy.signal.medfilt(target_x, 5), scipy.signal.medfilt(target_y, 5)
    return target_x, target_y


def get_relevant_measures(dlc_csv, conversion_rate, use_cv2_target_keypoints=False):
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
    mouse_speeds: np.array, speed of mouse
    target_speeds: np.array, speed of target
    (neck_x, neck_y): tuple, x and y positions of neck
    (snout_x, snout_y): tuple, x and y positions of snout
    (base_x, base_y): tuple, x and y positions of base
    (target_x, target_y): tuple, x and y positions of target
    df: pandas.DataFrame, whole DLC dataframe
    """

    df = read_DLC_csv(dlc_csv)
    # Get mouse and target positions
    headstage_base = interpolate_cubic_spline(get_bodypart(df, "headstage_base"))
    headstage_base_x, headstage_base_y = headstage_base.real, headstage_base.imag

    neck = interpolate_cubic_spline(get_bodypart(df, "spine1"))
    neck_x, neck_y = neck.real, neck.imag

    snout = interpolate_cubic_spline(get_bodypart(df, "snout"))
    snout_x, snout_y = snout.real, snout.imag

    base = interpolate_cubic_spline(get_bodypart(df, "tailbase"))
    base_x, base_y = base.real, base.imag

    # Get target positions
    # Remove target outliers by using the mag, and then for when the likelihood of the roach is greater than the mag, use the roach.
    target_x = df[('single', 'mag', 'x')].values
    target_y = df[('single', 'mag', 'y')].values

    if use_cv2_target_keypoints:
                
        # Read in pursuit target data
        cv2_target_df = pd.read_csv(cv2_csv).reset_index()
        # windowed pursuit target data
        win_cv2_target_df = cv2_target_df.iloc[win[0]:win[1]]
        # Fill in the gaps if possible with DLC data
        # Get indices of nan values
        naninds = win_cv2_target_df.isna().any(axis=1)
        cv2_target_x, cv2_target_y = cv2_target_df["x"].values, cv2_target_df["y"].values

        # For the nan values, fill in with DLC data
        cv2_target_x[naninds] = target_x[naninds]
        cv2_target_y[naninds] = target_y[naninds]

        target_x = cv2_target_x # scipy.signal.medfilt(interpolate_cubic_spline(cv2_target_x), 5)
        target_y = cv2_target_y # scipy.signal.medfilt(interpolate_cubic_spline(cv2_target_y), 5)
    
    # If the likelihood of the roach is greater than the mag, use the roach
    swap_in_roach_inds = np.where(df[('single', 'roach', 'likelihood')] > df[('single', 'mag', 'likelihood')])[0]
    target_x[swap_in_roach_inds] = df[('single', 'roach', 'x')][swap_in_roach_inds]
    target_y[swap_in_roach_inds] = df[('single', 'roach', 'y')][swap_in_roach_inds]

    # If the distance between two points is greater than 100, it's an outlier
    aberrant_inds = np.where(np.sqrt(np.diff(target_x) ** 2 + np.diff(target_y)) > 100)[0]
    target_x[aberrant_inds], target_y[aberrant_inds] = np.nan, np.nan
    target_x, target_y = interpolate_cubic_spline(target_x), interpolate_cubic_spline(target_y)
    
    # target_x, target_y = clean_target_positions(df)
    # Smooth all the data
    neck_x, neck_y = smooth(neck_x), smooth(neck_y)
    snout_x, snout_y = smooth(snout_x), smooth(snout_y)
    base_x, base_y = smooth(base_x), smooth(base_y)
    target_x, target_y = smooth(target_x), smooth(target_y)

    # suppress warnings using with statement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Get distances
        dist = np.sqrt((neck_x - target_x) ** 2 + (neck_y - target_y) ** 2) / conversion_rate

        # Get speeds
        mouse_speeds = get_speed_timeseries(neck_x, neck_y)
        target_speeds = get_speed_timeseries(target_x, target_y)

        # Get angles
        LOS = (target_x - neck_x) + (target_y - neck_y) * 1j
        head_vector = (snout_x - neck_x) + (snout_y - neck_y) * 1j
        body_vector = (neck_x - base_x) + (neck_y - base_y) * 1j

        # Egocentric head angle
        ego_angle = get_angular_difference_array(head_vector, body_vector)

        # Allocentric head angle
        head_angle = np.angle(head_vector, deg=True)

        # head-target head angle
        target_bearing = get_angular_difference_array(LOS, head_vector)
        
    return {
        'LOS':LOS, 'head_vector':head_vector, 'body_vector':body_vector, 'ego_angle':ego_angle, 'target_bearing':target_bearing,
        'head_angle':head_angle, 'dist':dist, 'mouse_speeds':mouse_speeds, 'target_speeds':target_speeds, "neck": (neck_x, neck_y),
        "snout": (snout_x, snout_y), "base": (base_x, base_y), "target":(target_x, target_y), 'df':df
    }


def relevant_measures_to_tuples(rm_d):
    """Relevant measures were originally a tuple, but now there are too many of them.
    They have to be dictionaries now, but it is useful to be able to unpack them."""
    (
        LOS, head_vector, body_vector, ego_angle, target_bearing, head_angle, dist, 
        mouse_speeds, target_speeds, (neck_x, neck_y), (snout_x, snout_y), (base_x, base_y), 
        (target_x, target_y), df
    ) = rm_d['LOS'], rm_d['head_vector'], rm_d['body_vector'], rm_d['ego_angle'], \
            rm_d['target_bearing'], rm_d['head_angle'], rm_d['dist'], rm_d['mouse_speeds'], rm_d['target_speeds'], \
            rm_d['neck'], rm_d['snout'], rm_d['base'], rm_d['target'], rm_d['df']
    
    return (
        LOS, head_vector, body_vector, ego_angle, target_bearing, head_angle, dist, 
        mouse_speeds, target_speeds, (neck_x, neck_y), (snout_x, snout_y), (base_x, base_y), 
        (target_x, target_y), df
    )

def get_boris_bout(boris_df, bout_ind, behavior="approach"):
    assert behavior in boris_df.Behavior.unique()
    behavior_df = boris_df.loc[boris_df.Behavior==behavior, :]
    bout = behavior_df.iloc[bout_ind * 2 : bout_ind * 2 + 2, :]
    return bout


def behavior_bout_generator(avi_boris_dlc, conversion_rate, behavior="approach", warnings=True):
    """"""# Now, generate videos of each behavior with the target and head vectors moving through space
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
        boris_df = boris_df.loc[boris_df["Behavior"] == behavior,:]
        
        relevant_measures = get_relevant_measures(dlc_csv, conversion_rate)

        (   LOS, head_vector, body_vector, ego_angle, target_bearing, head_angle, dist,
            mouse_speeds, target_speeds, (neck_x, neck_y), (snout_x, snout_y), (base_x, base_y),
            (target_x, target_y), df) = relevant_measures_to_tuples(relevant_measures)

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



def extract_waveforms(recording, sorting, unit_df, pre=0.001, post=0.002, n_waveforms=100, label="good", sr=30_000):
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






