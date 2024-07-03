import pandas as pd
import numpy as np
import glob
import cv2
import tqdm.auto as tqdm
import warnings
import os
import scipy.io
from .analysis import *

conversion_rate = 820 / 18 / 2.54  # pixels per cm




def smooth(angle_arr, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(angle_arr, box, mode='same')
    return y_smooth


def get_angular_difference(u, v):
    """Parameters
    ----------
    u, v: np.array, phasors or angles"""
    # First, check if it's a phasor
    # If phasor, convert to angles in degrees
    if np.iscomplex(u).any() or np.iscomplex(v).any():
        u = np.angle(u, deg=True)
        v = np.angle(v, deg=True)
    elif not np.iscomplex(u).any() and not np.iscomplex(v).any():
        # If it is an angle, modulus any angles outside [-360, 360]
        u = u % 360
        v = v % 360
    else:
        raise ValueError("One of the inputs is a phasor and the other is an angle. Please convert to the same type.")
    thing = u - v

    clip_inds = np.where(np.abs(thing) > 180)[0]
    if len(clip_inds) > 0:
        thing[clip_inds] = (thing[clip_inds] + 180) % 360 - 180
    return thing


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
    df = pd.read_csv(path, header=[0, 1, 2, 3])

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

        matched_file_l.append(
            (
                avi_file,
                 boris_tsv_file,
                 DLC_csv_file,
                 cv2_pursuit_target_file
             )
        )

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


def get_relevant_measures(dlc_csv, conversion_rate):
    """
    Parameters:
    -----------
    dlc_csv: str, path to dlc output
    conversion_rate: float, pixels per cm

    Returns
    -------
    target_vector: np.array, phasor of target position
    head_vector: np.array, phasor of head position
    body_vector: np.array, phasor of body position
    ego_theta: np.array, egocentric head angle
    target_theta: np.array, head-target angle
    head_theta: np.array, allocentric head angle
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

    headstage_base_x = df[('mouse', 'headstage_base', 'x')]
    headstage_base_y = df[('mouse', 'headstage_base', 'y')]

    neck_x = df[('mouse', 'spine1', 'x')]
    neck_y = df[('mouse', 'spine1', 'y')]

    snout_x = df[('mouse', 'snout', 'x')]
    snout_y = df[('mouse', 'snout', 'y')]

    base_x = df[('mouse', 'tailbase', 'x')]
    base_y = df[('mouse', 'tailbase', 'y')]

    target_x = df[('single', 'roach', 'x')]
    target_y = df[('single', 'roach', 'y')]

    # suppress warnings using with statement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Remove target outliers
        aberrant_inds = np.where(np.sqrt(np.diff(target_x) ** 2 + np.diff(target_y)) > 100)[0]
        target_x[aberrant_inds] = np.nan
        target_y[aberrant_inds] = np.nan

        # Get distances
        dist = smooth(np.sqrt((neck_x - target_x) ** 2 + (neck_y - target_y) ** 2) / conversion_rate, 4)

        # Get speeds
        mouse_speeds = get_speed_timeseries(neck_x, neck_y)
        target_speeds = get_speed_timeseries(target_x, target_y)

        # Get angles!
        target_vector = (target_x - neck_x) + (target_y - neck_y) * 1j
        head_vector = (snout_x - headstage_base_x) + (snout_y - headstage_base_y) * 1j
        body_vector = (neck_x - base_x) + (neck_y - base_y) * 1j

        # Egocentric head angle
        ego_theta = get_angular_difference(head_vector, body_vector)

        # Allocentric head angle
        head_theta = np.angle(head_vector, deg=True)

        # head-target head angle
        target_theta = get_angular_difference(target_vector, head_vector)

        ego_theta = smooth(ego_theta, 4)
        head_theta = smooth(head_theta, 4)
        target_theta = smooth(target_theta, 4)

    return (target_vector, head_vector, body_vector, ego_theta, target_theta, head_theta, dist,
            mouse_speeds, target_speeds, (neck_x, neck_y), (snout_x, snout_y), (base_x, base_y),
            (target_x, target_y), df)


def behavior_bout_generator(avi_boris_dlc, conversion_rate, behavior="approach"):
    # Now, generate videos of each behavior with the target and head vectors moving through space

    for avi_index, (avi_file, boris_tsv, dlc_csv, cv2_csv) in enumerate(tqdm.tqdm(avi_boris_dlc)):
        # Identify missing data
        if boris_tsv is None:
            print(f"{avi_index}th file missing boris data: {avi_file}")
            continue

        if dlc_csv is None:
            print(f"{avi_index}th file missing DLC data: {dlc_csv}")
            continue

        if cv2_csv is None:
            print(f"{avi_index}th file missing cv2 keypoint data: {cv2_csv}")
            continue

        # Open tsv
        boris_df = pd.read_csv(boris_tsv, sep='\t')

        # Clean boris
        boris_df = clean_boris(boris_df)
        
        # Remove irrelevant behaviors
        boris_df = boris_df.loc[boris_df["Behavior"] == behavior,:]
        
        relevant_measures = get_relevant_measures(dlc_csv, conversion_rate)

        (   target_vector, head_vector, body_vector, ego_theta, target_theta, head_theta, dist,
            mouse_speeds, target_speeds, (neck_x, neck_y), (snout_x, snout_y), (base_x, base_y),
            (dlc_target_x, dlc_target_y), df) = relevant_measures

        for bout_index in np.arange(0, len(boris_df), 2):
            # Get csv of target positions
            win = boris_df.iloc[bout_index:bout_index+2]["Image index"].values.astype(int)
            
            n_frames = np.diff(win)[0]

            # Windowing DLC parameters win head angle, snout position, body vector, head vector
            win_head_theta = head_theta[win[0]:win[1]]
            win_snout_x = snout_x[win[0]:win[1]]
            win_snout_y = snout_y[win[0]:win[1]]
            win_body_vector = body_vector[win[0]:win[1]]
            win_head_vector = head_vector[win[0]:win[1]]

            # Read in pursuit target data
            cv2_target_df = pd.read_csv(cv2_csv).reset_index()
            # windowed pursuit target data
            win_cv2_target_df = cv2_target_df.iloc[win[0]:win[1]]
            win_cv2_target_df = cv2_target_df.iloc[win[0]:win[1]]
            # Fill in the gaps if possible with DLC data
            # Get indices of nan values
            naninds = win_cv2_target_df.isna().any(axis=1)
            win_cv2_target_x = win_cv2_target_df["x"].values
            win_cv2_target_y = win_cv2_target_df["y"].values

            win_cv2_target_x[naninds] = dlc_target_x[win[0]:win[1]].values[naninds]
            win_cv2_target_y[naninds] = dlc_target_y[win[0]:win[1]].values[naninds]

            win_target_vector = (win_cv2_target_x - neck_x[win[0]:win[1]]) + (win_cv2_target_y - neck_y[win[0]:win[1]]) * 1j
            win_target_theta = get_angular_difference(win_target_vector, win_head_vector)

            yield (
                relevant_measures, win_head_theta, win_snout_x, win_snout_y,
                win_body_vector, win_head_vector, win_cv2_target_x, win_cv2_target_y,
                win_target_vector, win_target_theta, win, n_frames,
                avi_file, boris_tsv, dlc_csv, bout_index//2, boris_df
            )



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
