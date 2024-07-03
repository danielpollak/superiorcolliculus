import warnings

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tqdm.auto as tqdm
import seaborn as sns

from moviepy.video.io.VideoFileClip import VideoFileClip
import imageio

from .analysis import *
from .processing import *

from spikeinterface.preprocessing import bandpass_filter
import spikeinterface.widgets as sw

#https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name-boundrynorm-an
import matplotlib as mpl
import matplotlib.cm as cm
class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)


    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)



def Traj(wi,ve,vd,dt,du):
    '''
    wi = arena width in m
    ve = velocity in m/s
    vd = direction diffusion coefficient in rad^2/s
    dt = sampling interval in s
    du = duration in s
    
    Returns
    X,Y = arrays of x and y values
    '''
    # np.random.seed(0)
    # # Will use cumsum to change all of this
    # Number of samples
    N = int(du / dt)
    theta = np.cumsum(2*vd*dt*np.random.randn(N)) % (2*np.pi)
    dx = ve * np.cos(theta) * np.sign(theta)
    dy = ve * np.sin(theta) * np.sign(theta)
    dx[0] = np.random.uniform(0,wi)
    x = np.cumsum(dx)
    dy[0] = np.random.uniform(0, wi)
    y = np.cumsum(dy)

    remaining = True
    while remaining:
        x_i = np.where((x > wi) | (x < 0))[0]
        y_i = np.where((y > wi) | (y < 0))[0]
        remaining = len(x_i) > 0 or len(y_i  > 0)

        # It could, by some miracle, be empty
        if len(x_i) > 0:
            # Flip the rest of the dxs
            dx[(x_i[0]-1):] = -dx[(x_i[0]-1):]

            # Set the slice of vector to add to the remainder
            to_add = dx[(x_i[0]-1):]
            
            # Set the first element to where you were
            to_add[0] = x[(x_i[0]-1)]
            
            # Finally, compute the rest of the sequence.
            x[(x_i[0]-1):] = np.cumsum(to_add)
        
        if len(y_i) > 0:
            dy[(y_i[0]-1):] = -dy[(y_i[0]-1):]
            to_add = dy[(y_i[0]-1):]
            
            to_add[0] = y[(y_i[0]-1)]
            
            y[(y_i[0]-1):] = np.cumsum(to_add)
    
    return x, y



def get_meshgrid(field_size, n_pix):
    x_coords = np.linspace(-field_size/2, field_size/2, n_pix)
    y_coords = x_coords.copy()
    
    # note first index varies x, second index varies y
    x, y = np.meshgrid(x_coords, y_coords, indexing='ij')
    return x,y


def get_circle(field_size, n_pix,a,b):
    """
    xy: (np.array, np.array), meshgrid of coordinates in angular coordinates
    a/b: int, coords of target, azimuth and elevation, respectively

    Returns
    output: (np.array, np.array), meshgrid of distances

    You can then make a circular mask around it by doing simple boolean indexing,
    """
    x,y = get_meshgrid(field_size, n_pix)
    
    return (x - a)**2 + (y - b)**2


def get_angular_target_size(distance, target_diameter=0.00125):
    """Everything here is in cm for now"""
    # Operating on half the diameter
    return 2 * np.arctan(target_diameter/2/distance)


def get_target_elevation(distance, mouse_height=0.02):
    """1 cm"""
    return np.arctan(mouse_height/distance)


def process_binocular_region(frame, x, y, binocular_hemiregion=20):
    # Add binocular region
    frame *= 200
    frame[:, np.where((x > -np.deg2rad(binocular_hemiregion)) & (x < np.deg2rad(binocular_hemiregion)))[0]] += 20
    frame = np.clip(frame, 0, 255)

    return frame


def generate_video(path, target_azimuth, target_elevation, target_size, n_pix=208, field_size=np.pi/2*3):
    """
    Parameters
    ----------
    """
    NFRAMES = len(target_azimuth)
    stim_2D = np.zeros(shape=(n_pix, n_pix, NFRAMES), dtype="uint8")

    # 
    field_size = np.pi/2*3
    for i in tqdm.tqdm(range(NFRAMES)):
        P = get_circle(field_size, n_pix, target_azimuth[i], target_elevation[i])
        prey = P < target_size[i]
        stim_2D[:,:,i] = prey


    x, y = get_meshgrid(field_size, n_pix)

    imageio.mimsave(
        path, [255 - process_binocular_region(stim_2D[:,:,i].T, x, y) for i in tqdm.tqdm(range(stim_2D.shape[2]))],
        fps=10
    )


def generate_video_grid(
        path, target_azimuth_list, target_elevation_list,
        target_size_list, n_pix=208, field_size=np.pi/2*3, N=5, fps=30):
    """
    Generate a NxN grid video.
    
    Parameters
    ----------
    target_azimuth_list: List of azimuth arrays (each array for a smaller video)
    target_elevation_list: List of elevation arrays
    target_size_list: List of size arrays

    """
    
    NFRAMES = len(target_azimuth_list[0])
    
    # Initialize 3D array for the large video
    stim_2D_large = np.zeros((n_pix * N, n_pix * N, NFRAMES), dtype="uint8")

    # Get meshgrid for processing
    x, y = get_meshgrid(field_size, n_pix)
    for row in range(N):
        for col in range(N):
            
            if len(target_azimuth_list) > (row * N + col):
                # Query appropriate trace
                target_azimuth = target_azimuth_list[row * N + col]
                target_elevation = target_elevation_list[row * N + col]
                target_size = target_size_list[row * N + col]
                
                # Initialize 3D array for smaller video
                stim_2D_small = np.zeros((n_pix, n_pix, NFRAMES), dtype="uint8")
                
                # Get images for each frame
                for i in range(NFRAMES):
                    P = get_circle(field_size, n_pix, target_azimuth[i], target_elevation[i])
                    prey = P < target_size[i]
                    stim_2D_small[:,:,i] = prey
                    stim_2D_small[:,:,i] = process_binocular_region(stim_2D_small[:,:,i].T, x, y)

                # Place the smaller video in the larger one
                stim_2D_large[row * n_pix:(row + 1) * n_pix, col * n_pix:(col + 1) * n_pix, :] = stim_2D_small
            
    
    # Save the video
    imageio.mimsave(
        path, 
        [255 - stim_2D_large[:,:,i] for i in tqdm.tqdm(range(NFRAMES))], 
        fps=fps
    )



def generate_event_triggered_alignments(event_df, behavior, DLC_metric, win=20, ax=None):
    # Get dataframe with only the behaviors in question
    bh_df = event_df.loc[event_df["Behavior"]==behavior]

    # Check if ax is None
    if ax is None:
        _, ax = plt.subplots()
    

    # Check valid behavior parameter value
    if len(bh_df) == 0:
        print(f"no behaviors of type {behavior}")
        return

    # preallocate behavioral traces
    bh_traces_start = []
    for _, row in bh_df.iterrows():
        # Get current row    
        frame = row["Image index"]
        if np.isnan(frame):
            continue
        else:
            # Start or end of behavior
            bh_traces_start.append(DLC_metric[int(frame-win):int(frame+win)])
    
    # Plot 
    for bh_trace in bh_traces_start:
        try:
            ax.plot(np.arange(-win,win), bh_trace, color="grey", alpha=0.05) 
        except:
            pass
        
    # Make midline
    ax.vlines(0, *ax.get_ylim(), color="k", alpha=0.5)
    
    return bh_traces_start




def get_behavior_dict(event_df):
    bh_d = {bh: bh_i for bh_i, bh in enumerate(event_df["Behavior"].unique())  if bh != ""}
    return bh_d


def generate_ethogram(event_df):
    bh_d = get_behavior_dict(event_df)
    CH = MplColorHelper("tab10", 0, len(bh_d))
    _, ax = plt.subplots(figsize=(10,3))
    for row_ind in np.arange(0, len(event_df), 2):
        
        frame_start = event_df.iloc[row_ind]["Image index"]
        frame_end = event_df.iloc[row_ind + 1]["Image index"]

        behavior = event_df.iloc[row_ind]["Behavior"]

        # plot a thick line from frame to end of frame
        ax.hlines(0, frame_start, frame_end,
            colors=CH.get_rgb(bh_d[behavior]), linewidth=50)

    # Make legend
    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()
    for bh in bh_d.keys():
        ax.plot([-2, -1], [-2, -1], color=CH.get_rgb(bh_d[bh]), linewidth=5, label=bh)

    ax.set_xlim(old_xlim)
    ax.set_ylim(old_ylim);
    ax.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticklabels([])
    ax.set_yticks([])

    return ax



def generate_transition_matrix(event_df):
    """Generate transition matrix from event_df
    Parameters
    ----------
    event_df : pandas.DataFrame with columns "Behavior type" and "Behavior"

    Returns
    -------
    mat : numpy.ndarray normalized transition matrix
    """
    # Transition matrix
    bh_d = get_behavior_dict(event_df)

    # Initialize matrix
    mat = np.zeros((len(bh_d), len(bh_d)))

    # Get sequence of behaviors
    bh_seq = event_df.loc[event_df["Behavior type"]=="START", "Behavior"].values

    # For each transition, add to matrix
    for bh_i in range(len(bh_seq)-1):
        mat[bh_d[bh_seq[bh_i]], bh_d[bh_seq[bh_i+1]]] += 1

    # Normalize
    mat = mat/np.sum(mat)

    # Plot
    ax = sns.heatmap(mat, annot=True)
    ax.set_xticks(np.arange(len(bh_d))+0.5)
    ax.set_yticks(np.arange(len(bh_d))+0.5)
    ax.set_xticklabels(bh_d.keys())
    ax.set_yticklabels(bh_d.keys(), rotation=45);
    ax.set_title("Transition matrix")

    return mat, ax






def set_axis_units(ax, conversion_rate):
    ax.set_xticklabels(ax.get_xticks() // conversion_rate)
    ax.set_yticklabels(ax.get_yticks() // conversion_rate)
    ax.set_xlabel("cm")
    ax.set_ylabel("cm")


def format_behavior_ax(ax):
    ax.set_xlim((0, 900))
    ax.set_ylim((0, 800))
    ax.axis("equal")
    # remove gridlines on the axis
    # Make a 10 cm scale bar
    ax.plot([10 * conversion_rate, 20 * conversion_rate], [1 * conversion_rate, 1 * conversion_rate], c="k")
    ax.text(10 * conversion_rate, 3 * conversion_rate, "10 cm", fontsize=10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

def generate_event_triggered_maps(event_df, behavior, DLC_metric, win=10, ax=None):
    """Plotting the where of behaviors"""
    if ax is None:
        _, ax = plt.subplots(figsize=(8,3))
    
    sc = None
    traces = []
    for _, row in event_df.loc[event_df["Behavior"]==behavior].iterrows():
        
        # Start or end of behavior
        if row["Behavior type"] == "START":

            # Get current row    
            frame = row["Image index"]
            
            # Extract
            bh_trace_x, bh_trace_y = DLC_metric[0][int(frame-win):int(frame+win)], DLC_metric[1][int(frame-win):int(frame+win)]

            # Append
            traces = [(bh_trace_x, bh_trace_y)]

            # Plot
            sc = ax.scatter(
                bh_trace_x, bh_trace_y,s=1,
                color=plt.cm.seismic(np.arange(len(bh_trace_y))/len(bh_trace_x))
            )
        
    cb = plt.colorbar(sc)
    return traces





def plot_behavior_bout(generator_vars):
    (
        relevant_measures, win_head_theta, win_snout_x, win_snout_y,
        win_body_vector, win_head_vector, win_target_x, win_target_y,
        win_target_vector, win_target_theta, win, n_frames, avi_file, boris_tsv, dlc_csv, event_i, boris_df
    ) = generator_vars

    # Whole-video level variables
    (
        _, head_vector, body_vector, ego_theta, _,
        head_theta, dist, mouse_speeds, target_speeds, (neck_x, neck_y),
        (snout_x, snout_y), (base_x, base_y), (_, _), df
    ) = relevant_measures

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    axes = axes.flatten()
    cmap_name = "nipy_spectral"

    # Length of each vector is inter-frame distance.
    # For top two panels:
    for ax in axes[:2]:
        ax.quiver(
            win_target_x[:-1], win_target_y[:-1], np.diff(win_target_x), np.diff(win_target_y),
            scale=1, scale_units='xy', angles='xy', color=sns.color_palette(cmap_name, n_colors=n_frames),
            width=0.002, label="head-target")
        ax.set_title("Target position")

    # Euclidian distance between snout and target
    snout_heading_difference = get_angular_difference(win_target_theta, win_head_theta)

    # Body directions
    # Length of each vector is unit
    win_base = base_x[win[0]:win[1]] + base_y[win[0]:win[1]] * 1j
    win_neck = neck_x[win[0]:win[1]] + neck_y[win[0]:win[1]] * 1j
    axes[2].quiver(
        win_base.values.real,
        win_base.values.imag,
        win_neck.values.real-win_base.values.real,
        win_neck.values.imag-win_base.values.imag,
        scale=1, scale_units='xy', angles='xy', width=0.002, label="head angle",
        color=sns.color_palette(cmap_name, n_colors=n_frames))

    # Length of each vector is unit
    axes[3].quiver(
        win_neck.values.real,
        win_neck.values.imag,
        win_snout_x-win_neck.values.real,
        win_snout_y-win_neck.values.imag,
        scale=1, scale_units='xy', angles='xy', width=0.002, label="head angle",
        color=sns.color_palette(cmap_name, n_colors=n_frames))

    # Get heading inaccuracy from body line to target.
    body_heading_difference = get_angular_difference(win_target_theta, np.angle(win_body_vector, deg=True))
    axes[4].scatter(
        np.arange(len(body_heading_difference)), body_heading_difference, marker="|",
        c=sns.color_palette(cmap_name, n_colors=n_frames))

    # Get heading inaccuracy from neck to snout
    axes[5].scatter(np.arange(len(snout_heading_difference)), snout_heading_difference, marker="|",
                    c=sns.color_palette(cmap_name, n_colors=n_frames))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for ax_subind, ax in enumerate([axes[4], axes[5]]):
            ax.set_ylabel(r"degrees") if ax_subind == 0 else ax.set_ylabel(r"")
            tax = ax.twinx()
            tax.set_ylabel("") if ax_subind == 0 else tax.set_ylabel("m-s distance (cm)")
            tax.grid(False)
            tax.plot(
                np.arange(len(snout_heading_difference)),
                np.sqrt((win_snout_x - win_target_x) ** 2 + (win_snout_y - win_target_y) ** 2) / conversion_rate,
                c="k", alpha=0.3
            )

    for ax in axes[0:4]:
        format_behavior_ax(ax)

    [ax.grid(False) for ax in axes]

    for ax in [axes[4], axes[5]]:
        ax.set_xticklabels(np.round(ax.get_xticks() / 30, 1))
        ax.set_xlabel("time (s)")

    axes[1].set_title("Target position")
    axes[2].set_title("mouse body direction")
    axes[3].set_title("mouse head direction")

    # Could be unaligned but have a small difference in heading
    axes[4].set_title(r"bodyline target bearing ($^{\circ}$)")

    # Could be perfectly aligned but still have a large difference in heading
    axes[5].set_title(r"headline target bearing ($^{\circ}$)")

    return fig, axes


def plot_traintracks(pos, target_positions, ax, crossties=False):
    # Plot position of target
    ax.plot(target_positions.real, target_positions.imag, c="orange", alpha=0.3, label="target")
    ax.plot(target_positions.real[0], target_positions.imag[0], c="orange", alpha=0.3, marker="o")

    # Plot ground truth for pursuer
    ax.plot(pos.real, pos.imag, c="b", alpha=0.3, label="pursuer")
    ax.plot(pos.real[0], pos.imag[0], c="b", alpha=0.3, marker="o")

    if crossties:
        for xtie_ind in np.arange(0, len(pos), 10):
            ax.plot(
                [pos.real[xtie_ind], target_positions.real[xtie_ind]],
                [pos.imag[xtie_ind], target_positions.imag[xtie_ind]],
                c="k", alpha=0.3)


"""Neuropixel"""
from matplotlib import cm
from matplotlib.patches import ConnectionPatch

def plot_annot_timeseries(recording, sorting, unit_df, time_range, ax, markersize=3, N_chan=5):
    """
    Plots the raw timeseries annotated with curated traces. I should probably annotate with uncurated traces too.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_df: pd.DataFrame
        The dataframe of units
    time_range: list
        Two-element list of times (in seconds) to investigate
    ax: matplotlib.Axes
        The axis to plot on
    markersize: int
        The size of the markers for the rasters
    """
    sr = sorting.get_sampling_frequency()
    unit_df = unit_df.loc[unit_df.group == "good"]

    # Get channels
    channel_ids = unit_df.ch.unique()[:N_chan]
    sw.plot_timeseries(bandpass_filter(recording), channel_ids=channel_ids, time_range=time_range, ax=ax)

    # Get units on those channels
    channel_to_unit_d = {}
    for channel in channel_ids:
        channel_to_unit_d[channel] = unit_df.loc[unit_df.ch == channel, "cluster_id"].values

    # Get colors for units
    unit_df["color"] = [cm.brg(np.random.uniform()) for _ in range(len(unit_df))]

    # Thin lines and plot rasters
    for line, channel_id in zip(ax.get_lines(), channel_ids):
        line.set(linewidth=0.5, color="k")

        x, y = line.get_data()

        # Plot rasters
        d_entry = channel_to_unit_d[channel_id]
        for unit_id in d_entry:
            # Get event times (in samples)
            events = sorting.get_unit_spike_train(
                unit_id, start_frame=time_range[0] * sr,
                end_frame=time_range[1] * sr)

            # Convert event times to seconds
            events = events / sr

            ax.plot(events, np.repeat(np.mean(y) + 350, len(events)), "v",
                    color=unit_df.loc[unit_df["cluster_id"] == unit_id, "color"].values[0], markersize=markersize)

    return ax


def plot_zoomed_annot_timeseries(recording, sorting, unit_df, expt, tr1=[2, 3], tr2=[2.3, 2.5], markersize=8):
    """
    Plot the timeseries of the recording with the annotations of the units
    """
    fig, axes = plt.subplots(2, 1, figsize=(30, 20))

    plot_annot_timeseries(recording, sorting, unit_df, tr1, axes[0], markersize=markersize)
    plot_annot_timeseries(recording, sorting, unit_df, tr2, axes[1], markersize=markersize)

    for ax in axes:
        plt.sca(ax)
        plt.xticks(fontsize=20)
        plt.xlabel("time(s)", fontsize=20)

    # Add zoom bars
    con = ConnectionPatch(xyA=(axes[1].get_xlim()[0], axes[0].get_ylim()[0]), coordsA=axes[0].transData,
                          xyB=(axes[1].get_xlim()[0], axes[1].get_ylim()[1]), coordsB=axes[1].transData)
    axes[1].add_artist(con)
    con = ConnectionPatch(xyA=(axes[1].get_xlim()[1], axes[0].get_ylim()[0]), coordsA=axes[0].transData,
                          xyB=(axes[1].get_xlim()[1], axes[1].get_ylim()[1]), coordsB=axes[1].transData)
    axes[1].add_artist(con);

    if not os.path.isdir(r"\\datanas\family\SC\data_agg\physiology" + f"\\{expt}_diagnostic1\\"):
        os.mkdir(r"\\datanas\family\SC\data_agg\physiology" + f"\\{expt}_diagnostic1")

    fig.savefig(r"\\datanas\family\SC\data_agg\physiology" + f"\\{expt}_diagnostic1\\annot_timeseries.png")
    plt.close(fig)

