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
from numba import njit, int32, float64
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
def sigmoid(x, x0, k=1):
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
            dtheta_dt = par_nav_dtheta_dt_Phi(Phi_queue, alpha_queue, parameters)
        elif mode == 7:
            distance = np.abs(target_position - coords[i]) # distance to target
            sig = sigmoid(distance, parameters[-1]) 

            # When Distance is low, we use PN pursuit, when high, we use PP
            PP_contrib = sig * prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, np.array([parameters[0]]))
            PN_contrib = (1 - sig) * par_nav_dtheta_dt_alpha(Phi_queue, alpha_queue, np.array([parameters[1]]))
            dtheta_dt = PP_contrib + PN_contrib

        elif mode == 8:
            distance = np.abs(target_position - coords[i]) # distance to target
            sig = sigmoid(distance, parameters[-1]) 

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
    return 1-((residual_variance(prediction, ground_truth) / np.var(ground_truth)))


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
    LOGFILE = r"C:\Users\dan\Documents\SuperiorColliculus\data\log" + f'\SC{label}_mode{mode}_tau{tau}_tau1{tau1}_log.csv'

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

    # alphadot

    alphadot = dangle_dt(LOS) / IFI # deg/s

    # bearing, theta_e, in phasor
    phi = get_angular_difference_array(LOS, headings) # deg
    phidot = dangle_dt(phasor(phi)) / IFI # deg/s

    assert not np.iscomplexobj(thetadot)
    assert not np.iscomplexobj(phi)
    assert not np.iscomplexobj(alphadot)
    assert not np.iscomplexobj(phidot)

    return thetadot, phi, alphadot, phidot


def get_lagged_design_matrix(predictor, d):
    """
    Create time-lag design matrix from stimulus intensity vector.
    Args:
        stim (1D array): Stimulus intensity at each time point.
        d (number): Number of time lags to use.
    Returns
        X (2D array): GLM design matrix with shape T, d
    """
    # Create version of stimulus vector with zeros before onset
    padded_stim = np.concatenate([np.zeros(d - 1), predictor.T])

    # Construct a matrix where each row has the d frames of
    # the stimulus preceding and including timepoint t
    T = len(predictor)  # Total number of timepoints
    X = np.zeros((T, d))
    for t in range(T):
        X[t] = padded_stim[t:t + d]
    # X = np.hstack((np.ones((X.shape[0],1)), X))
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


def add_two_delays(df, Phi_tau, Phidot_tau):
    """If taus are different, then arrays will not be same lengths.
    To deal with this, we shift the whole dataframe by the maximum of the two taus.
    Then, phi and phi dot are cut from the end by Phi_tau and Phidot_tau respectively.
    If taus are different, then we cut the difference in taus from the beginning of the longer one.
    This preserves the temporal lags of all of them.
    
    I am confusingly interested in both phidot and alphadot. Unfortunately I do need to shift both. Therefore, I will
    shift alphadot with phidot_tau for now.

    Parameters
    -----------
    df : pd.DataFrame
    A_phi : int time delay for Phi
    B_phidot : int time delay for Phidot
    """
    assert (Phi_tau >= 0) & (Phidot_tau >= 0), "tau must be non-negative"
    
    # Shift phi and phidot
    Phi    = df.Phi.values[:-Phi_tau if Phi_tau > 0 else None]
    Phidot = df.Phidot.values[:-Phidot_tau if Phidot_tau > 0 else None]
    alphadot=df.alphadot.values[:-Phidot_tau if Phidot_tau > 0 else None]
    
    if Phi_tau != Phidot_tau:
        # If different, then cut the difference in taus from the longer one. That's equivalent to cutting the theta two different ways.
        front = np.abs(Phi_tau - Phidot_tau)

        # Make them the same length
        if Phi_tau > Phidot_tau:
            Phidot = Phidot[front:]
            alphadot=alphadot[front:]
        else:
            Phi = Phi[front:]
    
    # frontshift all and replace the phi and phidot separately
    tau = max(Phi_tau, Phidot_tau)
    output_df = df.copy().iloc[tau:]
    output_df["Phi"] = Phi
    output_df["Phidot"] = Phidot
    output_df["alphadot"] = alphadot

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


def get_data_PoggioReichart(subject_l, tau_phi=0, tau_phidot=0):
    """Data for estimating r and D for each subject 
    based on ReichaDrt and Poggio (1976) model
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
            
            # Inter-frame interval (s)
            fps = cv2.VideoCapture(avi).get(cv2.CAP_PROP_FPS)

            full_pos, full_target_positions = (neck_x + neck_y * 1j), (target_x + target_y * 1j)
            pos, target = full_pos[slice(*win)], full_target_positions[slice(*win)]

            dist = np.sqrt((snout_x[slice(*win)] - target_x[slice(*win)])**2 + (snout_y[slice(*win)] - target_y[slice(*win)])**2) / conversion_rate

            tan_thetadot, tan_Phi, alphadot, tan_Phidot = get_directional_variables(pos, target, fps)
            # Filter tangent variables due to their high frequency noise
            order = 2
            W = .2
            '''tan_thetadot = scipy.signal.filtfilt(
                *scipy.signal.butter(order, W, btype='lowpass', analog=False, output='ba'),
                tan_thetadot) if len(tan_thetadot) > 10 else tan_thetadot # because of issues with short bouts
            tan_Phi = scipy.signal.filtfilt(
                *scipy.signal.butter(order, W, btype='lowpass', analog=False, output='ba'),
                tan_Phi) if len(tan_Phi) > 10 else tan_Phi
            # alphadot = scipy.signal.filtfilt(*scipy.signal.butter(order, W, btype='lowpass', analog=False, output='ba'),alphadot) if len(alphadot) > 10 else alphadot
            tan_Phidot = scipy.signal.filtfilt(
                *scipy.signal.butter(order, W, btype='lowpass', analog=False, output='ba'),
                tan_Phidot) if len(tan_Phidot) > 10 else tan_Phidot'''
            tan_thetadot = scipy.signal.medfilt(tan_thetadot, 5)# because of issues with short bouts
            tan_Phi = scipy.signal.medfilt(tan_Phi, 5)
            alphadot = scipy.signal.medfilt(alphadot, 5)
            tan_Phidot = scipy.signal.medfilt(tan_Phidot)
            

            bout_df = pd.DataFrame(
                {"theta":np.concatenate([[np.diff(pos)[0]], np.diff(pos)]),
                 "thetadot":tan_thetadot, "Phi":tan_Phi, "Phidot":tan_Phidot, "alphadot":alphadot,
                 "pos":pos, "target":target, "dist":dist, "subject":subject, "tau":tau_phi, "tau1":tau_phidot, "ego":np.nan,
                 "mode":"tangent", "avi":avi, "event ind":event_ind})

            bout_df_l.append(add_two_delays(bout_df, tau_phi, tau_phidot))


            HD_thetadot, HD_Phi, _, HD_Phidot = get_directional_variables(pos, target, fps, head_vector[slice(*win)])
            HD_thetadot = scipy.signal.medfilt(HD_thetadot, 5)
            HD_Phi = scipy.signal.medfilt(HD_Phi, 5)
            HD_Phidot = scipy.signal.medfilt(HD_Phidot, 5)
            bout_df = pd.DataFrame(
                {"theta":head_vector[slice(*win)], "thetadot":HD_thetadot, "Phi":HD_Phi, "Phidot":HD_Phidot,
                "alphadot":alphadot,"pos":pos, "dist":dist, "target":target, "behavior":"approach", "subject":subject,
                "tau":tau_phi, "tau1":tau_phidot, "ego": rm_d["ego_angle"][slice(*win)],"mode":"HD", "avi":avi, "event ind":event_ind})

            bout_df_l.append(add_two_delays(bout_df, tau_phi, tau_phidot))

            # Alphadot is the same regaDrless of whether you're getting the bearing for HD or BD
            BD_thetadot, BD_Phi, _, BD_Phidot = get_directional_variables(pos, target, fps, body_vector[slice(*win)])
            BD_thetadot = scipy.signal.medfilt(BD_thetadot, 5)
            BD_Phi = scipy.signal.medfilt(BD_Phi, 5)
            BD_Phidot = scipy.signal.medfilt(BD_Phidot, 5)
            bout_df = pd.DataFrame(
                {"theta":body_vector[slice(*win)],"thetadot":BD_thetadot, "Phi":BD_Phi, "Phidot":BD_Phidot, "alphadot":alphadot,
                 "pos":pos, "target":target, "dist":dist, "behavior":"approach", "subject":subject, "tau":tau_phi, "tau1":tau_phidot,"ego":np.nan,
                 "mode":"BD", "avi":avi, "event ind":event_ind})

            bout_df_l.append(add_two_delays(bout_df, tau_phi, tau_phidot))
    
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


# This didn't quite behave. Ignore for now.
# @njit(cache=True)
# def linear_regression(x, y):
#     """
#     Perform linear regression using Numba for speed.
#     """
#     x_mean = np.mean(x)
#     y_mean = np.mean(y)
#     m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
#     b = y_mean - m * x_mean
#     return m, b

# import numpy.linalg as LA
# @njit(cache=True)
# def linear_regression_LA(x, y):
#     return LA.pinv(x) @ y

# @njit(cache=True)
# def bootstrap_CI_jit(x, y, n_boot=1000, alpha=0.05):
#     """
#     Perform bootstrap regression to calculate confidence intervals.
#     """
#     n = len(x)
#     m_values = np.zeros((n_boot, x.shape[1]))

#     for i in range(n_boot):
#         indices = np.random.choice(n, n, replace=True)
#         x_boot = x[indices]
#         y_boot = y[indices]
#         m_values[i] = linear_regression_LA(x_boot, y_boot).T
#     return m_values

# def bootstrap_CI(x, y, n_boot=1000, alpha=0.05):
#     m_values = bootstrap_CI_jit(x, y, n_boot=n_boot, alpha=alpha)
#     return np.percentile(m_values, [2.5,97.5], axis=0).T