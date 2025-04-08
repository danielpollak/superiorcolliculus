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


def format_behavior_ax(ax, scalebar_color="k"):
    """Format the axis for the behavior plots
    Set x/y limits, remove ticks, and set aspect ratio
    Add a scale bar"""
    ax.set_xlim((0, 900))
    ax.set_ylim((0, 800))
    ax.axis("equal")
    # remove gridlines on the axis
    # Make a 10 cm scale bar
    ax.plot([10 * conversion_rate, 20 * conversion_rate], [1 * conversion_rate, 1 * conversion_rate], c=scalebar_color, lw=2)
    ax.text(10 * conversion_rate, 3 * conversion_rate, "10 cm", fontsize=10, color=scalebar_color)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    # To agree the DLC with the video, which is inverted by DLC.
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





def plot_behavior_bout(generator_vars, cmap_name="nipy_spectral"):
    (relevant_measures, win_head_angle, win_snout_x, win_snout_y, win_body_vector, win_head_vector, win_target_x, win_target_y,
    win_target_position, win_target_bearing, win, n_frames, avi_file, boris_tsv, dlc_csv, event_i, boris_df) = generator_vars

    # Whole-video level variables
    (_, head_vector, body_vector, ego_angle, _, head_angle, dist, mouse_speeds, target_speeds, (neck_x, neck_y),
    (snout_x, snout_y), (base_x, base_y), (_, _), df) = relevant_measures

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    axes = axes.flatten()
    # Length of each vector is inter-frame distance.
    # For top two panels:
    for ax in axes[:2]:
        ax.scatter(
            win_target_x[::2], win_target_y[::2], 
            color=sns.color_palette(cmap_name, n_colors=len(win_target_x[::2])),
            s=1, label="head-target")
        
        # ax.quiver(
        #     win_target_x[:-1], win_target_y[:-1], np.diff(win_target_x), np.diff(win_target_y),
        #     scale=1, scale_units='xy', angles='xy', color=sns.color_palette(cmap_name, n_colors=n_frames),
        #     width=0.002, label="head-target")
        ax.set_title("Target position")

    # Euclidian distance between snout and target
    snout_heading_difference = get_angular_difference(win_target_bearing, win_head_angle)

    # Body directions
    # Length of each vector is unit
    win_base = base_x[win[0]:win[1]] + base_y[win[0]:win[1]] * 1j
    win_neck = neck_x[win[0]:win[1]] + neck_y[win[0]:win[1]] * 1j
    axes[2].quiver(
        win_base.real[::2],
        win_base.imag[::2],
        win_neck.real[::2] - win_base.real[::2],
        win_neck.imag[::2] - win_base.imag[::2],
        scale=1, scale_units='xy', angles='xy', width=0.004, label="head angle",
        color=sns.color_palette(cmap_name, n_colors=len(win_base.real[::2])))

    # Length of each vector is unit
    axes[3].quiver(
        win_neck.real[::2],
        win_neck.imag[::2],
        win_snout_x[::2] - win_neck.real[::2],
        win_snout_y[::2] - win_neck.imag[::2],
        scale=1, scale_units='xy', angles='xy', width=0.004, label="head angle",
        color=sns.color_palette(cmap_name, n_colors=len(win_neck.real[::2])))

    # Get heading inaccuracy from body line to target.
    body_heading_difference = get_angular_difference(win_target_bearing, np.angle(win_body_vector, deg=True))
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
        ax.set_xticklabels(np.round(ax.get_xticks() / 30.01, 2))
        ax.set_xlabel("time (s)")

    axes[1].set_title("Target position")
    axes[2].set_title("mouse body direction")
    axes[3].set_title("mouse head direction")

    # Could be unaligned but have a small difference in heading
    axes[4].set_title(r"bodyline target bearing ($^{\circ}$)")

    # Could be perfectly aligned but still have a large difference in heading
    axes[5].set_title(r"headline target bearing ($^{\circ}$)")

    return fig, axes


def plot_traintracks(
    pos, target_positions, ax, crossties=0,
    pos_label="pursuer", target_label="target",
    xtie_kwargs={}, pos_kwargs={}, target_kwargs={}):
    """
    Parameters
    ----------
    pos: (np.ndarray) Array of complex numbers representing the position of the pursuer at each time step.
    target_positions: (np.ndarray) Array of complex numbers representing the position of the target at each time step.
    ax: (matplotlib.axes.Axes) The axis on which to plot the train tracks.
    crossties: (int) If > 0, plot dashed lines (crossties) between the pursuer and target at every crossties-th time step.
        Set to 0 to disable crossties.
    pos_label: (str) Label for the pursuer's trajectory in the legend.
    
    """
    assert np.iscomplexobj(pos)
    assert np.iscomplexobj(target_positions)
    
    # Plot position of target
    ax.plot(target_positions.real, target_positions.imag, **target_kwargs)
    ax.plot(target_positions.real[0], target_positions.imag[0], marker="o", label=target_label, **target_kwargs)

    # Plot ground truth for pursuer
    ax.plot(pos.real, pos.imag, label=pos_label, **pos_kwargs)
    ax.plot(pos.real[0], pos.imag[0], marker="o", **pos_kwargs)

    if crossties:
        for xtie_ind in np.arange(0, len(pos), crossties):
            ax.plot(
                [pos.real[xtie_ind], target_positions.real[xtie_ind]],
                [pos.imag[xtie_ind], target_positions.imag[xtie_ind]],
                c="k", alpha=0.3, linestyle="dashed", **xtie_kwargs)
            
            

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


def unit_test_pursuit_model(mode, params, tau, step_sizes=5):
    cardinal_directions = [1+0j, 1+1j, 0+1j, -1+1j, -1+0j, -1-1j, 0-1j, 1-1j]
    fig, axes = plt.subplots(1,8, figsize=(20, 2.5))
    
    for count, cardinal in enumerate(cardinal_directions):
        
        agent_init_position = -20 + 25j

        # heading depends on initial position and has to be pointing toward target
        agent_initial_heading = (0+0j) - agent_init_position
        
        target_positions = cardinal * np.arange(100) + cardinal
        agent_step_sizes = np.ones_like(target_positions, dtype=float) * step_sizes
        pos = target_positions.copy()
        
        experimental_data = (agent_initial_heading, agent_init_position, agent_step_sizes, target_positions, pos) 
        xy, agent_vectors = guidance_law(params, experimental_data[:-1], tau, tau, mode)

        plot_traintracks(np.array(xy), np.array(target_positions), axes[count], crossties=True)
    return fig, axes, xy, agent_vectors



def plot_D_r(res, Phi, axes, labelD, labelr):
    if type(res) == scipy.optimize._optimize.OptimizeResult:
        if len(res.x) == 4:
            D_sigma, D_A, r_sigma, r_A = res.x
        elif len(res.x) == 2:
            # Here's why this isn't insane. We will clear and turn off the appropriate axis.
            D_sigma, D_A = res.x
            r_sigma, r_A = res.x
        else:
            raise ValueError("OptimizeResult must have 2 or 4 elements")
    elif type(res) == np.ndarray:
        D_sigma, D_A, r_sigma, r_A = res
    axes[0].scatter(Phi, D(D_sigma, D_A, Phi),s=2, label=labelD)
    axes[1].scatter(Phi, r(r_sigma, r_A, Phi),s=2, label=labelr)

    X = np.linspace(-180, 180, 1000)
    axes[0].plot(X, D(D_sigma, D_A, X),  color="grey", zorder=-np.inf)
    axes[1].plot(X, r(r_sigma, r_A, X),  color="grey", zorder=-np.inf)
    
    axes[0].set(xlabel="$\Phi$ (deg)", ylabel="$D(\Phi)$")
    axes[1].set(xlabel="$\Phi$ (deg)", ylabel="$r(\Phi)$")
    
    for ax in axes.flatten():
        if np.sum(np.abs(ax.get_ylim())) < 1e-5:
            ax.set_ylim((-1, 1))

    if labelD is not None:
        axes[0].legend(handletextpad=0.1)
    
    if labelr is not None:
        axes[1].legend(handletextpad=0.1)


def plot_k_N(res, Phi, axes, labelk, labelN):
    if type(res) == scipy.optimize._optimize.OptimizeResult:
        if len(res.x) == 2:
            k, N = res.x
        else:
            raise ValueError("OptimizeResult must have 2 or 4 elements")
    elif type(res) == np.ndarray:
        k, N = res

    
    axes[0].scatter(Phi, k * Phi, s=2, label=labelk)
    axes[1].scatter(Phi, N * np.ones_like(Phi), s=2, label=labelN)

    X = np.linspace(-180, 180)
    axes[0].plot(X, k * X,  color="grey", zorder=-np.inf)
    axes[1].plot(X, N * np.ones_like(X),  color="grey", zorder=-np.inf)
    
    axes[0].set(xlabel="$\Phi$ (deg)", ylabel="$k \Phi$")
    axes[1].set(xlabel="$\Phi$ (deg)", ylabel="$N$")
    
    for ax in axes.flatten():
        if np.sum(np.abs(ax.get_ylim())) < 1e-5:
            ax.set_ylim((-1, 1))

    if (labelk is not None) and (labelN is not None):
        [ax.legend(handletextpad=0.1) for ax in axes]


def plot_contrib_trace(t_df, res, contrib_axes, model_name, model_sobriquet):
    """Contributions of D and r components
    Parameters
    ----------
    t_df : pd.DataFrame
        dataframe with thetadot, Phidot, Phidot, alphadot
    res : scipy.optimize._optimize.OptimizeResult
        result of the optimization
    contrib_axes : list
        list of axes to plot D, r, and prediction
    model_name : str
        model name, e.g. r"$D(\Phi) + r(\Phi)\dot{\Phi}$",
    model_sobriquet : str
        model sobriquet  e.g.  "D + r Phidot"
    """
    thetadot = t_df.thetadot.values
    t_Phi, t_Phidot, t_alphadot = t_df.Phi.values, t_df.Phidot.values, t_df.alphadot.values
    model_fun_d = {
            "D + r Phidot": lambda x, t_Phi=t_Phi, t_Phidot=t_Phidot: get_thetadot_PR(t_Phi, t_Phidot, *x),
            "D + r alphadot": (lambda x, t_Phi=t_Phi, t_alphadot=t_alphadot: get_thetadot_PR(t_Phi, t_alphadot, *x)),
            "D": lambda x, t_Phi=t_Phi: D(*x, t_Phi),
            "r Phidot": lambda x, t_Phi=t_Phi, t_Phidot=t_Phidot: r(*x, t_Phi) * t_Phidot,
            "r alphadot":lambda x, t_Phi=t_Phi, t_alphadot=t_alphadot: r(*x, t_Phi) * t_alphadot,
            "linear Phidot": lambda x, t_Phi=t_Phi, t_Phidot=t_Phidot:     x[0] * t_Phi + x[1] * t_Phidot,
            "linear alphadot": lambda x, t_Phi=t_Phi, t_alphadot=t_alphadot: x[0] * t_Phi + x[1] * t_alphadot}
    
    model_fun = model_fun_d[model_sobriquet]
    is_Phidot = r"\dot{\Phi}" in model_name

    dot_var = t_Phidot if is_Phidot else t_alphadot

    x_t = np.arange(len(t_df)) / 30.01

    contrib_axes[0].plot(x_t, t_Phi, color="blue")
    contrib_axes[1].plot(x_t, dot_var, color="c")
    
    contrib_axes[2].plot(x_t, thetadot, label=r"$\dot{\theta}$ ground truth", color="grey")
    contrib_axes[2].plot(x_t, model_fun(res.x), label=f"prediction", linewidth=1, color="r")

    # Plot D component only if D is not already the whole function, because otherwise the number of parameters would not match correctly
    if "D" in model_sobriquet and "r" in model_sobriquet:
        contrib_axes[2].plot(
            x_t, D(*res.x[:2], t_Phi), label=r"$r\dot{\Phi}$" if is_Phidot else r"$r\dot{\alpha}$",
            linewidth=1, color="b")
    
    # Plot r component only if R is not already the whole function. 
    # That means, you check if D is in the function. more complicated than the previous one.
    if "r" in model_sobriquet and "D" in model_sobriquet:
        contrib_axes[2].plot(x_t, r(*res.x[2:], t_Phi) * dot_var, label=f"D", linewidth=1, color="g")
    
    tau, tau1 = t_df.tau.values[0], t_df.tau1.values[0]
    contrib_axes[0].set(ylabel=r"$\Phi$ (deg)", title=f"{model_name} $\\tau$:{tau}, $\\tau1$:{tau1}")
    contrib_axes[1].set(ylabel=r"$\dot{\Phi}$ (deg/s)" if is_Phidot else r"$\dot{\alpha}$", xlabel="Time (s)")
    contrib_axes[2].set(ylabel=r"$\dot{\theta}$ (deg/s)")
    contrib_axes[2].legend()