#%%
import numpy as np
import pandas as pd
import cv2
from .processing import *
import tqdm.auto as tqdm
import scipy.signal
from scipy.optimize import minimize
import re
from collections import deque
from datetime import datetime
from numba import jit, njit, int32, float64
from numba.experimental import jitclass
import numpy as np
from numpy.linalg import slogdet, inv

#%%
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



def convert_to_plusminus180(angle_arr):
    """Converts angle array to be between -180 and 180"""
    angle_arr = angle_arr.copy()
    angle_arr[angle_arr > 180] = angle_arr[angle_arr > 180] - 360
    angle_arr[angle_arr < -180] = angle_arr[angle_arr < -180] + 360
    return angle_arr


def position_to_heading(positions):
    """Converts complex array of position timeseries to headings"""
    headings = np.diff(positions, prepend=positions[0])
    # The initial heading is always 0 degrees. I estimate that the difference between the first two headings
    # is 0 degrees instead, to minimize any abrupt changes
    headings[0] = headings[1]
    return headings

def get_video_duration(filename):
    # Open the video file
    cap = cv2.VideoCapture(filename)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        # ("Error: Couldn't open the video file.")
        return None
    
    # Get the frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the duration of the video in seconds
    duration = frame_count / fps
    
    # Release the VideoCapture object
    cap.release()
    
    return duration


def ReLU(x):
    return max(0, x)



def estimate_tortuosity(traj, win):
    """
    Estimate tortuosity using the arc-cord ratio, estimated continuously at each point along a 2D trajectory using a win size `win`.
    
    Parameters:
    traj (numpy.ndarray): A 2D trajectory as a numpy array of shape (n_points, 2).
    win (int): The win size to use for estimating the arc-cord ratio.
    
    Returns:
    numpy.ndarray: An array of the arc-cord ratio estimated continuously at each point along the trajectory using the given win size.
    """
    n_points = traj.shape[0]
    # Initialize as nans
    acr = np.zeros(n_points) * np.nan
    for i in range(n_points):
        
        if i > win:
            x, y = traj[i-win:i, 0], traj[i-win:i, 1]
            dx, dy = x[-1] - x[0], y[-1] - y[0]
            arc_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            cord_length = np.sqrt(dx**2 + dy**2)
            acr[i] = arc_length / cord_length
    return acr	


def fill_nans_with_interpolation(arr):
    # Identify indices of non-nans and nans
    not_nan_indices = np.where(~np.isnan(arr))[0]
    nan_indices = np.where(np.isnan(arr))[0]
    
    # Perform interpolation to fill nans
    filled_values = np.interp(nan_indices, not_nan_indices, arr[not_nan_indices])
    
    # Assign filled values to original array at nan positions
    arr[nan_indices] = filled_values
    
    return arr

# ---- CircularQueue ----
spec = [
    ('maxlen', int32),
    ('index', int32),
    ('full', int32),
    ('data', float64[:, :])
]

@jitclass(spec)
class CircularQueue:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.index = 0
        self.full = 0
        self.data = np.zeros((maxlen, 1))

    def append(self, item):
        self.data[self.index, 0] = item
        self.index = (self.index + 1) % self.maxlen
        if self.index == 0:
            self.full = 1

    def get(self):
        if self.full == 0:
            return self.data[:self.index, 0]
        else:
            out = np.empty(self.maxlen, dtype=np.float64)
            out[:self.maxlen - self.index] = self.data[self.index:, 0]
            out[self.maxlen - self.index:] = self.data[:self.index, 0]
            return out

@njit(cache=True)
def get_Phi_alpha(theta_complex, coord_complex, target_coord_complex):
    theta = np.degrees(np.angle(theta_complex)) 
    alpha = np.degrees(np.angle(target_coord_complex - coord_complex))
    
    Phi = get_angular_difference_scalar(theta_complex, np.exp(1j * np.deg2rad(alpha)))

    return Phi, alpha


@njit(cache=True)
def par_nav_dtheta_dt_alpha(Phi_queue, alpha_queue, params):
    N = params[0]
    alpha_vals = alpha_queue.get()
    if len(alpha_vals) > 1:
        dalpha_dt = alpha_vals[1] - alpha_vals[0] # simple diff
        dtheta_dt = N * dalpha_dt
    else:
        dtheta_dt = 0.0
    return dtheta_dt


@njit(cache=True)
def par_nav_dtheta_dt_Phi(Phi_queue, alpha_queue, params):
    N = params[0]
    Phi_vals = Phi_queue.get()
    if len(Phi_vals) > 1:
        dPhi_dt = Phi_vals[1] - Phi_vals[0] # simple diff
        dtheta_dt = N * dPhi_dt
    else:
        dtheta_dt = 0.0
    return dtheta_dt


@njit(cache=True)
def prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, params):
    k = params[0]
    # alpha_vals = alpha_queue.get()
    Phi_vals = Phi_queue.get()
    if len(Phi_vals) > 1:
        # dalpha_dt = (alpha_vals[1] - alpha_vals[0])
        dtheta_dt = - k * Phi_vals[0] # dalpha_dt - k * Phi_vals[0]
    else:
        dtheta_dt = 0.0
    return dtheta_dt


@njit(cache=True)
def biased_prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, params):
    k, beta = params[0], params[1]
    # alpha_vals = alpha_queue.get()
    Phi_vals = Phi_queue.get()
    if len(Phi_vals) > 1:
        # dalpha_dt = (alpha_vals[1] - alpha_vals[0])
        dtheta_dt = -k * (Phi_vals[0] + np.sign(Phi_vals[0]) * beta) # dalpha_dt - k * Phi_vals[0]
    else:
        dtheta_dt = 0.0
    return dtheta_dt

# ---- Guidance Law ----
from numba import njit
@njit(cache=True)
def sigm(x, x0, k=1):
    """Sigmoid function"""
    L = 1 # Maximum value

    return L / (1 + np.exp(-k * (x - x0)))

@njit
def guidance_law(parameters, experimental_data, tau, tau1, mode):
    
    tau = int(tau)
    tau1 = int(tau1)

    initial_heading, initial_position, step_sizes, target_positions = experimental_data

    n_steps = len(target_positions) # - tau - tau1
    vectors = np.zeros(n_steps + 1, dtype=np.complex128)
    coords = np.zeros(n_steps, dtype=np.complex128)

    vectors[0] = initial_heading
    coords[0] = initial_position

    theta = np.degrees(np.angle(initial_heading))

    initial_Phi, initial_alpha = get_Phi_alpha(initial_heading, initial_position, target_positions[0])
    Phi_queue = CircularQueue(tau + 2)
    alpha_queue = CircularQueue(tau1 + 2)
    [Phi_queue.append(initial_Phi) for _ in range(tau + 2)]
    [alpha_queue.append(initial_alpha) for _ in range(tau1 + 2)]

    for i in range(n_steps):
        target_position = target_positions[i]
        step_size = step_sizes[i]

        if mode == 0: # Pure proportional pursuit
            dtheta_dt = prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, parameters)

        elif mode == 1: # Biased proportional pursuit
            dtheta_dt = biased_prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, parameters)

        elif mode == 2: # Parallel navigation
            dtheta_dt = par_nav_dtheta_dt_alpha(Phi_queue, alpha_queue, parameters)

        elif mode == 3: # mixed
            dtheta_dt = prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, np.array([parameters[0]])) + par_nav_dtheta_dt_alpha(Phi_queue, alpha_queue, np.array([parameters[1]]))
        
        elif mode == 4:
            Phi = Phi_queue.get()
            dPhi_dt = (Phi[1] - Phi[0])  # simple diff

            D_sigma =  parameters[0]
            D_A =      parameters[1]
            r_sigma =  parameters[2]
            r_A =      parameters[3]
            
            dtheta_dt = get_thetadot_PR(Phi[0], dPhi_dt, D_sigma, D_A,  r_sigma, r_A)
        
        elif mode == 5:
            alpha = alpha_queue.get()
            dalpha_dt = (alpha[1] - alpha[0])  # simple diff

            D_sigma =  parameters[0]
            D_A =      parameters[1]
            r_sigma =  parameters[2]
            r_A =      parameters[3]
            
            dtheta_dt = get_thetadot_PR(alpha[0], dalpha_dt, D_sigma, D_A,  r_sigma, r_A)
        elif mode == 6:
            # CATD
            dtheta_dt = par_nav_dtheta_dt_Phi(Phi_queue, alpha_queue, parameters)
        elif mode == 7:
            distance = np.abs(target_position - coords[i]) # distance to target
            sig = sigm(distance, parameters[-1]) 

            # When Distance is low, we use PN pursuit, when high, we use PP
            PP_contrib = sig * prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, np.array([parameters[0]]))
            PN_contrib = (1 - sig) * par_nav_dtheta_dt_alpha(Phi_queue, alpha_queue, np.array([parameters[1]]))
            dtheta_dt = PP_contrib + PN_contrib

        elif mode == 8:
            distance = np.abs(target_position - coords[i]) # distance to target
            sig = sigm(distance, parameters[-1]) 

            # When Distance is low, we use PP, when high, we use PN
            PP_contrib = (1-sig) * prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, np.array([parameters[0]]))
            PN_contrib = (sig) * par_nav_dtheta_dt_alpha(Phi_queue, alpha_queue, np.array([parameters[1]]))
            dtheta_dt = PP_contrib + PN_contrib
        

        theta = (theta + dtheta_dt) % 360.0

        vector = step_size * np.exp(1j * np.deg2rad(theta))
        vectors[i + 1] = vector
        coords[i] = coords[i - 1] + vector if i > 0 else initial_position + vector

        Phi, alpha = get_Phi_alpha(vector, coords[i], target_position)
        # print("phi", int(Phi), "alpha",int(alpha)%360, "dthetadt",int(dtheta_dt), "theta", int(theta), "vector",int(np.angle(vector, deg=True)%360))
        Phi_queue.append(Phi)
        alpha_queue.append(alpha)

    return coords, vectors




@njit(cache=True)
def D(sigma, A, Phi):
    """First term, direction sensitivity term
    
    Parameters
    ----------
    sigma: (float) standard deviation of the Gaussian derivative
    A: (float) amplitude of the Gaussian
    Phi: (float) angle in degrees
    """
    Phi = 2 * np.pi / 360 * Phi
    return A * Phi / (np.sqrt(2 * np.pi) * sigma**3) * np.exp(-Phi ** 2 / (2 * sigma**2))


@njit(cache=True)
def r(sigma, A, Phi):
    """Second term, angular velocity gain term
    Parameters
    ----------
    sigma: (float) standard deviation of the Gaussian
    A: (float) amplitude of the Gaussian"""
    Phi = 2 * np.pi / 360 * Phi
    return A / (np.sqrt(2 * np.pi ) * sigma) * np.exp(-Phi**2/(2*sigma**2))


@njit(cache=True)
def get_thetadot_PR(Phi, Phidot, D_sigma, D_A,  r_sigma, r_A):
    return D(D_sigma, D_A,  Phi) + r(r_sigma, r_A, Phi) * Phidot


# k, b
def constant_bearing_pursuit(parameters, experimental_data, tau:int, tau1:int):
    """Parameters is beta, k is held to 1
    Parameters
    ----------
    parameters: (int, ) beta;
    experimental_data: (tuple) Initial heading (complex), initial coordinates (complex),
        step_sizes (np.array), target_coordinates (complex np.array)"""
    return prop_pursuit((1, *parameters), experimental_data, tau, tau1)


def classic_pursuit(parameters, experimental_data, tau, tau1):
    """Parameter is tau, beta is held to 0
    Parameters
    ----------
    parameters: (,) Empty, there are none
    experimental_data: (tuple) Initial heading (complex), initial coordinates (complex),
        step_sizes (np.array), target_coordinates (complex np.array)"""
    
    return prop_pursuit((1, 0, *parameters), experimental_data, tau, tau1)

def get_experimental_data(event_df):
    """Assumes data is formatted by PoggioReichart get data function
    Extracts the relevant experimental data from the event_df that it can be run through the guidance law function"""

    pos = event_df.pos.values
    agent_initial_heading = event_df.theta.values[0] # complex
    agent_init_position = pos[0]
    agent_step_sizes = np.abs(np.diff(pos, prepend=agent_init_position))

    target_positions = event_df.target.values
    return (agent_initial_heading, agent_init_position, agent_step_sizes, target_positions, pos)
'''Optimization'''
def get_datetime_from_avi(avi_file):

    # Regular expression to match the desired string
    regex = r'-(\d{14})-\d{4}.avi'

    # Find the match
    match = re.search(regex, avi_file)

    if match:
        dtstr = match.group(1)  # Extracting the specific capture group
        dt = np.datetime64(datetime.strptime(dtstr, '%m%d%Y%H%M%S'))
        return dt
    else:
        return None
    

def residual_variance(prediction, ground_truth):
    return np.var(prediction - ground_truth)


def get_VAF(prediction, ground_truth):
    GT_var = np.var(ground_truth)
    if GT_var == 0:
        return np.nan
    else:
        return 1 - ((residual_variance(prediction, ground_truth) / GT_var))


def MAE(prediction, ground_truth):
    """Mean absolute error"""
    # Calculate the mean absolute error (MAE)
    mae = np.mean(np.abs(prediction - ground_truth))
    return mae


def MSE(prediction, ground_truth):
    """Mean squared error"""
    # Calculate the mean squared error (MSE)
    mse = np.mean((prediction - ground_truth)**2)
    return mse


from concurrent.futures import ProcessPoolExecutor
def compute_cost(args):
    """Helper function to make cost picklable."""
    return cost(*args)


# def aggregate_cost(params, experimental_data_l, tau, tau1, mode, max_workers=8):
#     '''Parallelized version of aggregate_cost with constrained CPU usage.'''
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         errors = executor.map(compute_cost, [(params, data, tau, tau1, mode) for data in experimental_data_l])
#     error_l = list(errors)
#     total_avg_error = np.mean(error_l)
#     log_cost(params, total_avg_error, tau, tau1, mode)

#     return np.mean(error_l)


def aggregate_cost(params, experimental_data_l, tau, tau1, mode):
    '''Example useage:
    result = minimize(aggregate_cost, initial_guess, args=(experimental_data_l, constant_bearing_interception_pursuit), method='Nelder-Mead')
    prop_pursuit_k_optimized, prop_pursuit_tau_optimized = result.x
    '''
    error_l = []
    for experimental_data in experimental_data_l:
        error_l.append(cost(params, experimental_data, tau, tau1, mode))
    err = np.mean(error_l)
    log_cost(params, err, tau, tau1, mode)
    return err


def cost(params, experimental_data, tau:int, tau1:int, mode:int):
    """Now trying with VAF instead of error"""
    predicted_data, _ = guidance_law(params, experimental_data[:-1], tau, tau1, mode)
    error = MAE(predicted_data, experimental_data[-1])
    # error = 1 - residual_variance(predicted_data, experimental_data[-1])
    return error



def aggregate_cost_MSE(params, experimental_data_l, tau, tau1, mode):
    '''Example useage:
    result = minimize(aggregate_cost, initial_guess, args=(experimental_data_l, constant_bearing_interception_pursuit), method='Nelder-Mead')
    prop_pursuit_k_optimized, prop_pursuit_tau_optimized = result.x
    '''
    error_l = []
    for experimental_data in experimental_data_l:
        error_l.append(cost_MSE(params, experimental_data, tau, tau1, mode))
    err = np.mean(error_l)
    log_cost(params, err, tau, tau1, mode)
    return err


def cost_MSE(params, experimental_data, tau:int, tau1:int, mode:int):
    """Now trying with VAF instead of error"""
    predicted_data, _ = guidance_law(params, experimental_data[:-1], tau, tau1, mode)
    error = MSE(predicted_data, experimental_data[-1])
    # error = 1 - residual_variance(predicted_data, experimental_data[-1])
    return error


import csv
import os

iteration_counter = 0
LOGFILE = None

def init_log(label, tau, tau1,mode):
    global LOGFILE
    LOGFILE = r"C:\Users\dan\Documents\SuperiorColliculus\data\log" + f'\\SC{label}_mode{mode}_tau{tau}_tau1{tau1}_log.csv'

    with open(LOGFILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', "params", "err", "tau", "tau1", "mode"])

def log_cost(params, err, tau, tau1, mode):
    global iteration_counter
    global LOGFILE
    with open(LOGFILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([iteration_counter] + list(params) + [err, tau, tau1, mode])
    iteration_counter += 1



def shorten_experimental_data(experimental_data, start):
    """Shortens experimental data."""    
    # Shorten experimental data
    (initial_heading, init_position, step_sizes, target_positions, pos) = experimental_data
    
    # If there's anything left after start time, shorten it and return. Otherwise return None.
    # This will not run into an indexing error, I believe.
    if len(target_positions[start:]) > 1:
        shortened_experimental_data = (np.diff(pos[start:])[0], pos[start], step_sizes[start:], fill_nans_with_interpolation(target_positions[start:]), pos[start:])
        return shortened_experimental_data
    else:
        return None


def incremental_guidance_law(experimental_data, guidance_law_params):
    """Runs get_guidance_law in second-long increments"""
    # For each increment
    parameter_df_l = []
    pos = experimental_data[-1]
    # Iterate through each start period
    starts = np.arange(0, min(len(pos), 15), 3)
    for start_ind, start in enumerate(starts):
        short_expt_data_l = [shorten_experimental_data(elem, start) for elem in experimental_data if (shorten_experimental_data(elem, start) is not None)]

        parameter_df = get_guidance_law_df(short_expt_data_l, *guidance_law_params)
        
        parameter_df["handicap"] = start
        parameter_df_l.append(parameter_df)

    return pd.concat(parameter_df_l)


def get_guidance_law_df(experimental_data_l:list, mode,  initial_guess:tuple, tau:int, tau1:int, subject):
    """Selects guidance law and appropriate constraints to optimize

    Parameters
    ----------
    experimental_data: (list of tuples) list of: initial_heading, init_position, step_sizes, target_positions, pos
    mode: (int) 0: prop pursuit, 1: par nav, 2: Poggio-Reichart, 3: PR alphadot
    initial_guess: (tuple) Initial guess for parameters
    tau(1): (int) tau for the model. tau is the time delay for the model.
        tau1 is the time delay for the model (for the second parameter).

    Returns
    -------
    parameter_df: (pd.DataFrame) Dataframe containing the optimized parameters"""
    # Optimize
    init_log(subject, tau, tau1, mode)

    resultPowell = minimize(aggregate_cost, initial_guess, args=(experimental_data_l, tau, tau1, mode), method='Powell') 
    result = minimize(aggregate_cost, resultPowell.x, args=(experimental_data_l, tau, tau1, mode), method='L-BFGS-B', bounds=[(-17,17)] * len(initial_guess), options={"maxls":50}) 
    # print(result.x, result.success, result.nit, result.message)
    # Add datetime metadata.
    parameter_df = pd.DataFrame(
        {"params":[np.array(result.x)],"converged": result.success, "nit":result.nit,
        "error":result.fun}, index=[0])
    return parameter_df


        



def get_directional_variables(pos, target_positions, fps, body_angle_vectors=None):
    """
    Returns variables relevant for evaluating navigation models.
    Takes either trajectory-tangent headings or body angle vectors.
    
    Parameters
    ----------
    pos: (np.array, complex) Agent positions
    target_positions: (np.array, complex) Target positions
    allocentric_head_directions: (np.array, complex) head/body headings from DLC

    Returns
    -------
    thetadot: (np.array) head direction angle velocity
    phi: (np.array) bearing
    alphadot: (np.array) target angle
    phidot: (np.array) target angle velocity
    """

    assert (body_angle_vectors is None) or np.iscomplexobj(body_angle_vectors), "Headings must be complex"
    assert np.iscomplexobj(pos), "Positions must be complex"
    assert np.iscomplexobj(target_positions), "Target positions must be complex"

    IFI = 1/fps # s / frame

    # Headings tangent to trajectory
    if body_angle_vectors is not None:
        headings = body_angle_vectors
    else:
        diff_pos = np.diff(pos) # complex
        # The first heading is repeated so that the length
        # of headings is the same as the length of pos
        headings = np.concatenate(([diff_pos[0]], diff_pos)) # complex
    
    # Complex
    LOS = target_positions-pos # complex

    # head angle, omega_f, in degrees
    thetadot = dangle_dt(headings) # deg/frame
    thetadot = thetadot / IFI # deg/s

    # alpha: allocentric LOS angle (deg); alphadot: its angular velocity (deg/s)
    alpha    = np.angle(LOS, deg=True)
    alphadot = dangle_dt(LOS) / IFI # deg/s

    # bearing, theta_e, in phasor
    phi = get_angular_difference_array(LOS, headings) # deg
    phidot = dangle_dt(phasor(phi)) / IFI # deg/s

    assert not np.iscomplexobj(thetadot)
    assert not np.iscomplexobj(phi)
    assert not np.iscomplexobj(alpha)
    assert not np.iscomplexobj(alphadot)
    assert not np.iscomplexobj(phidot)

    return thetadot, phi, alpha, alphadot, phidot


def get_lagged_design_matrix(predictor, d_past, d_future=0):
    """
    Create time-lag design matrix from stimulus intensity vector.
    AKA an FIR
    Args:
        stim (np.ndarray): M datapoints x 1 dimensional array
            Stimulus intensity at each time point.
        d (number): Number of time lags to use.
    Returns
        X (2D array): GLM design matrix with shape T, d
    """
    assert (d_past > 0) or (d_future > 0)

    past_padded_stim, future_padded_stim = predictor, predictor


    # You subtract one to make sure that the first line has only one non-padded element.
    past_padded_stim = np.concatenate([np.zeros(max(0,d_past - 1)), predictor.T])

    # Same playbook as above, but with future. So flip it around
    future_padded_stim = np.concatenate([predictor.T, np.zeros(d_future)])
    
    
    # Construct a matrix where each row has the d frames of
    # the stimulus preceding and including timepoint t
    T = len(predictor)  # Total number of timepoints
    X = np.zeros((T, d_past + d_future))

    for t in range(T):

        sl_past = past_padded_stim[t:t+d_past]
        # Flip back around so it is in the right order
        sl_future = future_padded_stim[t+1:t+d_future+1]

        X[t,:] = np.concatenate([sl_past, sl_future])
    
    return X


def crosscorr(x,y):
    """Scale inputs to remain within range of correlation coefficients,

    Let's explain how to read this.
    Relative to `x`, at time `t`, `y` is `r` frames in the future (or past, if t is negative).
    Therefore, if there is a peak at +5, you would need to shift `y` 5 frames 
    to the left (to the past) to align it with `x`. That means `y` lags `x`.
    If there is a peak to the left, then y leads x.
    """
    return np.correlate(
        (x-np.mean(x))/np.std(x)/len(x),
        (y-np.mean(y))/np.std(y),
        "same")


def add_two_delays(df, Phi_tau, alphadot_tau):
    """If taus are different, then arrays will not be same lengths.
    To deal with this, we shift the whole dataframe by the maximum of the two taus.
    Then, phi and phi dot are cut from the end by Phi_tau and alphadot_tau respectively.
    If taus are different, then we cut the difference in taus from the beginning of the longer one.
    This preserves the temporal lags of all of them.
    
    Parameters
    -----------
    df : pd.DataFrame
    Phi_tau : int time delay for Phi and Phidot (frames)
    alphadot_tau : int time delay for alphadot and alpha (frames)
    """
    assert (Phi_tau >= 0) & (alphadot_tau >= 0), "tau must be non-negative"

    # Phidot = dPhi/dt, so it uses the same delay as Phi (Phi_tau), not alphadot_tau.
    # alphadot and alpha use alphadot_tau.
    Phi      = df.Phi.values[:-Phi_tau if Phi_tau > 0 else None]
    Phidot   = df.Phidot.values[:-Phi_tau if Phi_tau > 0 else None]
    alphadot = df.alphadot.values[:-alphadot_tau if alphadot_tau > 0 else None]
    alpha    = df.alpha.values[:-alphadot_tau if alphadot_tau > 0 else None]

    if Phi_tau != alphadot_tau:
        # Cut the difference in taus from the front of the longer array so all
        # arrays align with the same thetadot segment (offset by max tau).
        front = np.abs(Phi_tau - alphadot_tau)

        if Phi_tau > alphadot_tau:
            alphadot = alphadot[front:]
            alpha    = alpha[front:]
        else:
            Phi    = Phi[front:]
            Phidot = Phidot[front:]

    # frontshift all and replace the phi and phidot separately
    tau = max(Phi_tau, alphadot_tau)
    output_df = df.copy().iloc[tau:]
    output_df["Phi"]      = Phi
    output_df["Phidot"]   = Phidot
    output_df["alphadot"] = alphadot
    output_df["alpha"]    = alpha

    return output_df
    

def get_regex(input, pattern):
    """general regex convenience function"""
    match = re.search(pattern, input)
    return match.group(1) if match else None


def get_feedrate(avi_str):
    """Gets feedrate for open loop. In this case, the feedrate is given in the form of 'feedrateXXX' where XXX is the feedrate."""
    if "feedrate" in avi_str:
        return get_regex(avi_str,  r'feedrate(\d+)')
    elif "speed" in avi_str:
        return get_regex(avi_str, r'speed(\d+)')


def get_contingency(avi_str):
    """Gets the experimental contingency for closed loop. In this case, the contingency is given in the form of 'contingencyXXX' where XXX is the contingency."""
    if "lissajous" in avi_str:
        return "lissajous"
    elif "filleted" in avi_str:
        return "rectangle"
    elif "classic" in avi_str:
        return "classic"
    elif "probescape" in avi_str:
        return "probabilistic"
    elif "hide" in avi_str:
        return "hide"
    else:   
        return None


def radius_of_curvature(x, y, t):
    """
    Parameter
    ---------
    x, y: 
    t: time
    """
    # Calculate first and second derivatives
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    
    # Calculate curvature radius
    numerator = (dx**2 + dy**2)**(3/2)
    denominator = np.abs(dx * ddy - ddx * dy)
    denominator[denominator == 0] = np.nan  # Avoid division by zero
    R = (numerator / denominator)
    return R

def get_bh_data(subject_l, behavior="all"):
    """Data guidance law for each subject 
    example inputs;
    cortical: ["SC5",  "SC8", "SC16", "SC44","SC7",]
    acortical: ["SC9", 'SC17',"SC51"]

    Parameters
    ----------
    subject_l : list
        list of subjects
    function : function
        function to get thetadot, Phi, alphadot
    tau : int
    """
    bout_df_l = []
    for subject in subject_l:
        # approach vs nonapproach
        avi_boris_HD = match_avi_boris_dlc(r'\\datanas\family\SC\data_agg\behavior' + f'/{subject}\\*')
        for gv_d in behavior_bout_generator(avi_boris_HD, conversion_rate, behavior=behavior, warnings=False):
            
            # Event-level variables
            rm_d, avi, event_ind, boris_df = gv_d["relevant_measures"], gv_d["avi"], gv_d["event_ind"], gv_d["boris_df"]
            
            nframe = int(cv2.VideoCapture(avi).get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Inter-frame interval (s)
            fps = cv2.VideoCapture(avi).get(cv2.CAP_PROP_FPS)
            
            bh = boris_to_array(boris_df, nframe)  # boris_df
            # Whole-video level variables
            head_vector =  rm_d["head_vector"]
            body_vector = rm_d["body_vector"]
            (neck_x, neck_y), (snout_x, snout_y), (target_x, target_y) = rm_d["neck"], rm_d["snout"], rm_d["target"]
            
            dist = rm_d["dist"]
            

            pos, target = (neck_x + neck_y * 1j), (target_x + target_y * 1j)
            
            # Filter tangent variables due to their high frequency noise
            MEDFILT_KERNEL_SIZE = 5
            if len(pos) <= MEDFILT_KERNEL_SIZE: # medfilt kernel size
                continue
            
            dist = rm_d['dist'] # cm/s
            speed = np.concatenate([np.array([0]), get_speed_timeseries(pos.real, pos.imag, interval=1, fps=fps)])

            pos_RC = radius_of_curvature(pos.real, pos.imag, np.arange(len(pos)) / fps)
            tar_RC = radius_of_curvature(target.real, target.imag, np.arange(len(target)) / fps)

            tan_thetadot, tan_Phi, alpha, alphadot, tan_Phidot = get_directional_variables(pos, target, fps)

            common_args = {
                "alpha":alpha, "alphadot":alphadot, "pos":pos, "target":target, "behavior":bh,
                "dist":dist, "ddist":np.diff(dist, prepend=dist[0]), "subject":subject, "avi":avi, "event ind":event_ind,
                "speed":speed, "pos_RC":pos_RC, "tar_RC":tar_RC, "t":np.arange(len(pos))}

            tan_theta = np.concatenate([[np.diff(pos)[0]], np.diff(pos)])
            bout_df = pd.DataFrame({
                "theta":tan_theta, "thetadot":tan_thetadot,
                "Phi":tan_Phi, "Phidot":tan_Phidot, 
                "mode":"tangent","ego":np.nan,**common_args})

            bout_df_l.append(add_two_delays(bout_df, 0,0))


            HD_thetadot, HD_Phi, _, _, HD_Phidot = get_directional_variables(pos, target, fps, head_vector)
            bout_df = pd.DataFrame({
                "theta":head_vector, "thetadot":HD_thetadot,
                "Phi":HD_Phi, "Phidot":HD_Phidot,
                "ego": rm_d["ego_angle"],"mode":"HD", **common_args})

            bout_df_l.append(add_two_delays(bout_df, 0,0))

            # alpha and alphadot are the same regardless of whether you're getting the bearing for HD or BD
            BD_thetadot, BD_Phi, _, _, BD_Phidot = get_directional_variables(pos, target, fps, body_vector)
            bout_df = pd.DataFrame({
                "theta":body_vector, "thetadot":BD_thetadot,
                "Phi":BD_Phi, "Phidot":BD_Phidot,
                "ego":np.nan, "mode":"BD",**common_args})

            bout_df_l.append(add_two_delays(bout_df, 0,0))

            # Only do it for the first bout
            break

    bout_df = pd.concat(bout_df_l)

    return pd.concat(bout_df_l)


def get_pursuit_data(subject_l, tau_phi=0, tau_alphadot=0, smoothing_sigma=0):
    """Data guidance law for each subject
    example inputs;
    cortical: ["SC5",  "SC8", "SC16", "SC44","SC7",]
    acortical: ["SC9", 'SC17',"SC51"]

    Parameters
    ----------
    subject_l : list
        list of subjects
    function : function
        function to get thetadot, Phi, alphadot
    tau : int
    smoothing_sigma : float
        Gaussian smoothing sigma (frames) applied to pos and target position
        timeseries before computing kinematic variables. 0 means no smoothing.
    """
    bout_df_l = []
    for subject in subject_l: #: # :,
        # approach vs nonapproach
        avi_boris_HD = match_avi_boris_dlc(r'\\datanas\family\SC\data_agg\behavior' + f'/{subject}\\*')
        for gv_d in behavior_bout_generator(avi_boris_HD, conversion_rate, behavior="approach", warnings=False):

            # Event-level variables
            rm_d, win, avi, event_ind = gv_d["relevant_measures"], gv_d["win"], gv_d["avi"], gv_d["event_ind"]

            # Whole-video level variables
            head_vector =  rm_d["head_vector"]
            body_vector = rm_d["body_vector"]
            (neck_x, neck_y), (snout_x, snout_y), (target_x, target_y) = rm_d["neck"], rm_d["snout"], rm_d["target"]

            dist = rm_d["dist"]
            # Inter-frame interval (s)
            fps = cv2.VideoCapture(avi).get(cv2.CAP_PROP_FPS)


            full_pos, full_target_positions = (neck_x + neck_y * 1j), (target_x + target_y * 1j)
            pos, target = full_pos[slice(*win)], full_target_positions[slice(*win)]

            if smoothing_sigma > 0:
                pos    = smooth(pos.real,    smoothing_sigma) + 1j * smooth(pos.imag,    smoothing_sigma)
                target = smooth(target.real, smoothing_sigma) + 1j * smooth(target.imag, smoothing_sigma)
            
            # Filter tangent variables due to their high frequency noise
            MEDFILT_KERNEL_SIZE = 5
            if len(pos) <= MEDFILT_KERNEL_SIZE: # medfilt kernel size
                continue
            
            dist = rm_d['dist'][slice(*win)] # cm/s
            speed = np.concatenate([np.array([0]), get_speed_timeseries(pos.real, pos.imag, interval=1, fps=fps)])

            body_unit_win = body_vector[slice(*win)] / np.abs(body_vector[slice(*win)])
            v_complex = np.diff(pos, prepend=pos[0]) * fps / conversion_rate
            v_proj = v_complex * np.conj(body_unit_win)
            v_forward = np.real(v_proj)  # cm/s; positive = forward (tailbase→neck), negative = backward
            v_lateral = np.imag(v_proj)  # cm/s; positive = rightward from mouse's perspective

            pos_RC = radius_of_curvature(pos.real, pos.imag, np.arange(len(pos)) / fps)
            tar_RC = radius_of_curvature(target.real, target.imag, np.arange(len(target)) / fps)

            tan_thetadot, tan_Phi, alpha, alphadot, tan_Phidot = get_directional_variables(pos, target, fps)

            tan_theta = np.concatenate([[np.diff(pos)[0]], np.diff(pos)])
            bout_df = pd.DataFrame({"theta":tan_theta,
                 "thetadot":tan_thetadot, "Phi":tan_Phi, "Phidot":tan_Phidot,
                 "alpha":alpha, "alphadot":alphadot,
                 "pos":pos, "target":target, "behavior":"approach", "dist":dist, "subject":subject, "tau":tau_phi, "tau1":tau_alphadot, "ego":np.nan,
                 "mode":"tangent", "avi":avi, "event ind":event_ind, "speed":speed, "pos_RC":pos_RC, "tar_RC":tar_RC,
                 "v_forward":v_forward, "v_lateral":v_lateral})

            bout_df_l.append(add_two_delays(bout_df, tau_phi, tau_alphadot))

            # alpha and alphadot are the same regardless of whether you're getting the bearing for HD or BD
            HD_thetadot, HD_Phi, _, _, HD_Phidot = get_directional_variables(pos, target, fps, head_vector[slice(*win)])
            bout_df = pd.DataFrame({"theta":head_vector[slice(*win)], "thetadot":HD_thetadot, "Phi":HD_Phi, "Phidot":HD_Phidot,
                "alpha":alpha, "alphadot":alphadot, "pos":pos, "dist":dist, "target":target, "behavior":"approach", "subject":subject,
                "tau":tau_phi, "tau1":tau_alphadot,
                "ego": rm_d["ego_angle"][slice(*win)],"mode":"HD",
                "avi":avi, "event ind":event_ind, "speed":speed, "pos_RC":pos_RC, "tar_RC":tar_RC,
                "v_forward":v_forward, "v_lateral":v_lateral})

            bout_df_l.append(add_two_delays(bout_df, tau_phi, tau_alphadot))

            BD_thetadot, BD_Phi, _, _, BD_Phidot = get_directional_variables(pos, target, fps, body_vector[slice(*win)])
            bout_df = pd.DataFrame({"theta":body_vector[slice(*win)],
                 "thetadot":BD_thetadot, "Phi":BD_Phi, "Phidot":BD_Phidot,
                 "alpha":alpha, "alphadot":alphadot, "pos":pos, "target":target, "dist":dist,
                 "behavior":"approach", "subject":subject, "tau":tau_phi, "tau1":tau_alphadot,
                 "ego":np.nan, "mode":"BD", "avi":avi, "event ind":event_ind,
                 "speed":speed, "pos_RC":pos_RC, "tar_RC":tar_RC,
                 "v_forward":v_forward, "v_lateral":v_lateral})

            bout_df_l.append(add_two_delays(bout_df, tau_phi, tau_alphadot))
    
    bout_df = pd.concat(bout_df_l)

    return pd.concat(bout_df_l)


def compute_aic(residuals, num_params):
    """Uses an assumption residuals are Gaussian
    Computes a pseudo-likelihood and plugs it into AIC
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a model fit, should be complex numbers
    num_params : int
        Number of parameters in the model fit
    Returns
    -------
    float
        AIC value
    """

    T = len(residuals)
    resid_array = np.column_stack([residuals.real, residuals.imag])
    Sigma = np.cov(residuals.real, residuals.imag)
    sign, logdet = slogdet(Sigma)
    quad_form = np.sum(resid_array @ inv(Sigma) * resid_array)
    log_likelihood = -0.5 * T * logdet - 0.5 * quad_form
    return 2 * num_params - 2 * log_likelihood



def sigmoid(x, A, m, b):
    """
    A: amplitude (limit)
    m: slope
    b: x offset"""
    return A / (1 + np.exp(-m*(x-b)))


def get_spiketrain_histogram(st, iup1, bins):
    """
    Limits histogram to be within the bounds of the video, hence range
    Step is the inter frame interval or bins

    
    """
    counts, time = np.histogram(st, range=(iup1[0], iup1[-1]), bins=bins)
    return counts, time


def clean_features(arr):
    """
    Clean feature array: Turn nans into last real value
    """
    curr = 0
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            arr[i] = curr
        else:
            curr = arr[i]
    return arr



@njit(cache=True)
def get_theta_mse(X, spike_counts):
    """
    Commputes mean mean squared error for STA 
    I think
    Parameters
    ----------
    X : array
        Design matrix
    spike_counts : array
        Spike counts    """
    return np.linalg.inv(X.T @ X) @ X.T @ spike_counts


@njit(cache=True)
def neg_log_lik_LNP(theta, X, y, beta):
    """From https://github.com/NeuromatchAcademy/course-content/blob/main/tutorials/W1D3_GeneralizedLinearModels/solutions/W1D3_Tutorial1_Solution_90f0230d.py"""
    """Return -loglike for the Poisson GLM model.
    Args:
        theta (1D array): Parameter vector.
        X (2D array): Full design matrix.
        y (1D array): Data values.
    Returns:
        number: Negative log likelihood.
    """
    # Compute the Poisson log likelihood
    rate = np.exp(X @ theta)
    
    log_lik = y @ np.log(rate) - rate.sum() - beta*np.sum( theta**2)
    # could just use mean squared error
    return -log_lik


def fit_LNP(spikes, X, x0, beta=0):
    """https://github.com/NeuromatchAcademy/course-content/blob/main/tutorials/W1D3_GeneralizedLinearModels/solutions/W1D3_Tutorial1_Solution_90f0230d.py"""
    """Obtain MLE parameters for the Poisson GLM.
    Args:
        stim (1D array): Stimulus values at each timepoint
        spikes (1D array): Spike counts measured at each timepoint
        d (number): Number of time lags to use.
    Returns:
        1D array: MLE parameters
    """
    #
    y = spikes

    # Find parameters that minmize the negative log likelihood function
    res = minimize(neg_log_lik_LNP, x0, args=(X, y, beta))

    return res["x"], res


def mean_sq_err(Ambtheta, X, y, beta, A):
    """ """
    m = Ambtheta[0]
    b = Ambtheta[1]
    theta = Ambtheta[2:]
    return np.mean((y-sigmoid(X @ theta / len(theta), A, m, b))**2) + beta*np.sum(theta**2)


def fit_LNS(spikes, X, x0, A, beta=0, maxiter=1000):
    """https://github.com/NeuromatchAcademy/course-content/blob/main/tutorials/W1D3_GeneralizedLinearModels/solutions/W1D3_Tutorial1_Solution_90f0230d.py"""
    """Obtain MLE parameters for the Sigmoid Linear nonlinear model.
    Not sure why the tutorial says this is about GLMs; this is not a GLM.
    Args:
        spikes (1D array): Spike counts measured at each timepoint
        X (2D array): Full design matrix.
        x0 (1D array): Initial guess for MLE parameters: m, b.
        beta (float): Regularization parameter.
        maxiter (int): Maximum number of iterations for optimization.
    Returns:
        1D array: MLE parameters
    """
    y = spikes

    # Find parameters that minimize the mean square error function    
    res = minimize(mean_sq_err, x0, args=(X, y, beta, A), method="Nelder-Mead", options={"maxiter":maxiter})
    m, b = res["x"][0], res["x"][1]
    theta = res["x"][2:]
    
    return m, b, theta, res



def plot_GLM(theta_MSE, theta, A, m, b, counts, time, predictor, d, LNP=False, expt="", label="", save=True):
    """generate diagnostic plots
    Parameters
    ----------
    theta_MSE : (array) Raw STA linear filter (initial guess)
    theta : (array) linear filter after optimizing alongside nonlinearity
    A : (float) Amplitude
    m : (float) Slope
    b : (float) Offset

    counts : (array) Binned spike counts
    time : (array) Time bins
    predictor : (array) Stimulus intensity
    d : (int, optional) Number of time lags to use. The default is 32.
    beta : (float, optional) Regularization parameter. The default is 0.
    """

    """Visualize kernels"""
    fig, ax = plt.subplots(1,2, figsize=(9,3))
    ax[0].plot(theta_MSE, label="STA") 
    ax[0].plot(theta, label="linear filter estimate")
    ax[0].set_title("linear filters")
    ax[0].legend()
    
    if not LNP:
        x = np.arange(-5,15,0.1)
        ax[1].plot(x, sigmoid(x, A, m, b), label=f"A: {np.round(A,3)}, m: {np.round(m,3)}, b:{np.round(b,3)}")
        ax[1].legend()
    # else: 
    #     ax[1].plot(x, sigmoid(x, A, m, b), label=f"A: {np.round(A,3)}, m: {np.round(m,3)}, b:{np.round(b,3)}")
    #     ax[1].legend()
    
    if save:
        fig.savefig(r"C:\Users\dan\Documents\SuperiorColliculus\figs\LN" + f"\\{expt}_kernels_{label}.png")
        plt.close(fig)

    """Visualize fits"""
    fig, ax = plt.subplots(2,1,figsize=(10,6))
    ax[0].plot(time, predictor)
    X = get_lagged_design_matrix(predictor, d, d)
    ax_twin = ax[0]
    ax_twin.plot(time, X @ theta_MSE, label="activation", color="b")
    if LNP:
        ax[1].plot(time, np.exp(X @ theta), label="LNP", color="c")
    else:
        ax[1].plot(time, sigmoid(X @ theta, A, m, b), label="LNS", color="c")
    ax[1].plot(time, counts, label="spike counts", color="k")
    
    ax[0].set_ylabel("predictor")
    ax[1].set_ylabel("spike counts")
    ax[1].set_xlabel("time (s)")
    ax[1].legend()
    if save:
        fig.savefig(r"C:\Users\dan\Documents\SuperiorColliculus\figs\LN" + f"\\{expt}_predictions_{label}.png")
        plt.close(fig)


def get_sta_X(X, r):
    """Gets spike triggered average using the design matrix
    Parameters
    ----------
    X : (np.ndarray) Design matrix, should be 2D
    r : (np.ndarray) Spike counts, should be 1D
    
    Returns
    -------
    sta : (np.ndarray) Spike triggered average, same length as number of columns in X

    """
    assert r.ndim==1
    assert X.ndim==2, "X should be 2D, got {}".format(X.ndim)
    assert X.shape[0] == r.shape[0], "X and r should have the same number of rows, got {} and {}".format(X.shape[0], r.shape[0])
    assert X.shape[1] > 0, "X should have at least one column, got {}".format(X.shape[1])
    n_spikes = np.sum(r)
    sta = X.T @ r / n_spikes  # Scales to the actual units of what we are looking at # * X.shape[1]

    return sta



from concurrent.futures import ThreadPoolExecutor
def bootstrap_sta_X_l(X, r, mask=None, N_SHUFFLES=1000, max_workers=None, shuffle_X=False):
    assert type(N_SHUFFLES) is int, "N_SHUFFLES should be an integer, got {}".format(type(N_SHUFFLES))
    
    if mask is None:
        mask = np.ones(len(r), dtype=bool)
    
    r_masked = r[mask]
    X_masked = X[mask]
    
    r_shuffles = np.random.uniform(0, len(r_masked), N_SHUFFLES)

    def _single_shuffle(s):
        r_rolled = np.roll(r_masked, int(s))

        return get_sta_X(X_masked, r_rolled)

    def _single_shuffle_X(s):        
        r_rolled = np.roll(r_masked, int(s))

        X_shift = int(np.random.uniform(0, X.shape[1]))
        X_rolled = np.roll(X_masked, X_shift)

        return get_sta_X(X_rolled, r_rolled)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_single_shuffle_X if shuffle_X else _single_shuffle, r_shuffles))


def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    
    '''
    assert len(signal) == len(ker), "signal and ker must have same shape"
    # Circular convolution using FFT
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))


def get_sta(predictor, r, window=None):
    '''
    Computes the spike-triggered average (STA) of a predictor and a response 
    via circular convolution.
    Parameters
    ----------
    predictor: real 1D array
    r: real 1D array
    window: int, half-size of STA to return

    Returns
    -------
    rolled_sta: real 1D array
    '''
    assert predictor.ndim == 1, "predictor must be 1D"
    assert r.ndim == 1, "r must be 1D"
    # assert len(predictor) == len(r), "predictor and r must have same shape"
    
    scaled_sta = conv_circ(predictor, r [::-1]) / np.sum(r)
    mp = len(scaled_sta) // 2

    # Roll the STA to center it
    rolled_sta = np.roll(scaled_sta, mp)

    # 
    if window is not None:
        # For some reason the STA is shifted by half the window size
        rolled_sta = rolled_sta[mp - window:mp + window]
    return rolled_sta


def circconv_sta_test(sta, comparisons=1000, test_statistic=np.var):
    """Generates null distribution of test_statistic and 
    of the middle of the STA. 

    Parameters
    ----------
    sta: (np.array) Spike-triggered average, full circular convolution. No windowing.
    comparisons: (int) Number of comparisons to make
    test_statistic: (function) Function to compute the test statistic, e.g. np.var, np.mean, etc.

    Returns
    -------
    statistic_vals: (np.array) Null distribution of test_statistic
    mp_sta: (float) Test statistic of the middle of the STA
    """
    # first, split up sta into comparisions sections centered around the middle
    winlen = len(sta) // comparisons
    mp = len(sta) // 2

    # First, get STA middle
    mp_sta = test_statistic(sta[mp - winlen // 2:mp + winlen // 2])

    # Then, get the middleless STA
    middleless_sta = np.concatenate(
        [sta[:mp - winlen // 2],
         sta[mp + winlen // 2:]]
    )

    # Compute statistics for each section
    statistic_vals = [
        test_statistic(
            middleless_sta[i * winlen:(i + 1) * winlen]
        ) for i in range(comparisons - 1)]
    

    return np.array(statistic_vals), mp_sta


def get_pval(arr, val):
    '''
        arr: 1D array
        val: value to compare against
    '''
    return np.sum(arr >= val) / len(arr)


def sorted_eig(mat):
    eig = np.linalg.eig(mat)
    eigenvalues = eig.eigenvalues
    eigenvectors = eig.eigenvectors

    sort_inds = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_inds]
    eigenvectors = eigenvectors[:, sort_inds]
    return eigenvalues, eigenvectors


'''
# pytorch 
import torch
from torch import nn, optim
import torch.nn.functional as F

class SingleLayerPerceptron(nn.Module):
  """ network with a single hidden layer h with a RELU """

  def __init__(self, n_inputs):
    super().__init__()  # needed to invoke the properties of the parent class nn.Module
    self.in_layer = nn.Linear(n_inputs, 1) # neural activity --> hidden units

  def forward(self, r):

    y = self.in_layer(r)


    return y


class DeepNet(nn.Module):
  """ network with a single hidden layer h with a RELU """

  def __init__(self, n_inputs, n_hidden=16):
    super().__init__()  # needed to invoke the properties of the parent class nn.Module
    self.in_layer = nn.Linear(n_inputs, n_hidden) # neural activity --> hidden units
    self.out_layer = nn.Linear(n_hidden, 1) # hidden units --> output

  def forward(self, r):
   
    h = self.in_layer(r) # h is size (n_inputs, n_hidden) 
    y = self.out_layer(h)


    return y


class DeepNetReLU(nn.Module):
  """ network with a single hidden layer h with a RELU """

  def __init__(self, n_inputs, n_hidden=16):
    super().__init__()  # needed to invoke the properties of the parent class nn.Module
    self.in_layer = nn.Linear(n_inputs, n_hidden) # neural activity --> hidden units
    self.out_layer = nn.Linear(n_hidden, 1) # hidden units --> output

  def forward(self, r):
   
    h = torch.relu(self.in_layer(r)) # h is size (n_inputs, n_hidden) 
    y = self.out_layer(h)


    return y


class DeepNetReLU(nn.Module):
    """Flexible MLP with configurable depth and ReLU nonlinearity."""

    def __init__(self, n_inputs, n_hidden=16, n_layers=1):
        """
        n_inputs: number of input features
        n_hidden: number of hidden units per layer
        n_layers: number of hidden layers
        """
        super().__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(n_inputs, n_hidden))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(n_hidden, 1))

        # Combine all layers into a Sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



class Conv1DTemporalNet(nn.Module):
    """Multiple layers. not interpretable"""
    def __init__(self, n_inputs, n_hidden=1, kernel_size=9, n_layers=2):
        super().__init__()

        # symmetrical padding for forward+backward context
        padding = kernel_size // 2

        layers = []
        in_channels = n_inputs
        for i in range(n_layers):
            layers.append(nn.Conv1d(in_channels, n_hidden, kernel_size, padding=padding))
            layers.append(nn.ReLU())
            # Maxpool & batchnorm?
            # Add validation loss and if it goes up add dropout for regularization
            in_channels = n_hidden

        self.conv_stack = nn.Sequential(*layers)
        self.output_layer = nn.Conv1d(n_hidden, 1, kernel_size=1)

    def forward(self, r):
        """
        r: (batch, n_inputs, time)
        Returns: (batch, 1, time)
        """
        h = self.conv_stack(r)
        y = self.output_layer(h)
        return y
    

class DepthwiseTemporalConv(nn.Module):
    def __init__(self, n_inputs, kernel_size, relu=False):
        super().__init__()
        # depthwise convolution: one kernel per input channel
        self.relu = relu
        self.depthwise = nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_inputs,
            kernel_size=kernel_size,
            groups=n_inputs,   # <--- key: makes it depthwise!
            padding=kernel_size // 2
        )

        # linear combination step
        self.combine = nn.Conv1d(
            in_channels=n_inputs,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, x):
        """
        x: (batch, n_inputs, time)
        returns: (batch, 1, time)
        """
        if self.relu:
            h = torch.relu(self.depthwise(x))
        else:
            h = self.depthwise(x)
        y = self.combine(h)
        return y


def format_data(X, y, net_class):
    """
    Wrangles data for training and testing in convolutinal
    and non-convolutinal neural nets.

    Assumes train-test split has already happened

    Parameters
    ----------
    X: np.array (n_neurons x timepoints)
    y: np.array (timepoints)
    
    Returns
    -------
    X_, y_: torch.tensor """
    assert type(net_class) == type
    
    X_ = torch.tensor(X, dtype=torch.float32) 
    y_ = torch.tensor(y, dtype=torch.float32)

    if (net_class == Conv1DTemporalNet) or (net_class == DepthwiseTemporalConv):
        
        X_ = X_.unsqueeze(0)
        y_ = y_.unsqueeze(0).unsqueeze(0)
    else:
        X_ = X_.T
        y_ = y_.unsqueeze(1)
    
    return X_, y_'''
# %%
