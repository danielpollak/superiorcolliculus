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

'''
@njit(cache=True)
def get_Phi_alpha(theta:np.complex128, coord:np.complex128, target_coord:np.complex128):
    """
    Parameters
    ----------
    theta: (complex) heading
    coord: (complex) position
    target_coord: (complex) target position

    Returns
    -------
    Phi: bearing
    alpha: LOS
    """
    # Update bearing
    alpha = target_coord - coord # deg
    Phi = get_angular_difference(np.complex128(theta), np.complex128(alpha)) # deg
    return Phi, alpha
'''

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
    Phi = get_angular_difference(theta, alpha)
    return Phi, alpha


@njit(cache=True)
def par_nav_dtheta_dt(Phi_queue, alpha_queue, params):
    N, = params
    alpha_vals = alpha_queue.get()
    if len(alpha_vals) > 1:
        dalpha_dt = (alpha_vals[1] - alpha_vals[0])  # simple diff
        dtheta_dt = N * dalpha_dt
    else:
        dtheta_dt = 0.0
    return dtheta_dt

@njit(cache=True)
def prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, params):
    k, = params
    alpha_vals = alpha_queue.get()
    Phi_vals = Phi_queue.get()
    if len(alpha_vals) > 1:
        dalpha_dt = (alpha_vals[1] - alpha_vals[0])
        dtheta_dt = dalpha_dt - k * Phi_vals[0]
    else:
        dtheta_dt = 0.0
    return dtheta_dt

# ---- Guidance Law ----

@njit
def guidance_law(parameters, experimental_data, tau, tau1, mode):
    tau = int(tau)
    tau1 = int(tau1)

    initial_heading, initial_position, step_sizes, target_positions = experimental_data

    n_steps = len(target_positions) - tau - tau1
    vectors = np.zeros(n_steps + 1, dtype=np.complex128)
    coords = np.zeros(n_steps, dtype=np.complex128)

    vectors[0] = initial_heading
    coords[0] = initial_position

    theta = np.degrees(np.angle(initial_heading))

    initial_Phi, initial_alpha = get_Phi_alpha(initial_heading, initial_position, target_positions[0])

    Phi_queue = CircularQueue(tau + 2)
    alpha_queue = CircularQueue(tau1 + 2)
    Phi_queue.append(initial_Phi)
    alpha_queue.append(initial_alpha)

    for i in range(n_steps):
        target_position = target_positions[i]
        step_size = step_sizes[i]

        if mode == 0:
            dtheta_dt = par_nav_dtheta_dt(Phi_queue, alpha_queue, parameters)
        else:
            dtheta_dt = prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, parameters)

        theta = (theta + dtheta_dt) % 360.0

        vector = step_size * np.exp(1j * np.deg2rad(theta))
        vectors[i + 1] = vector
        coords[i] = coords[i - 1] + vector if i > 0 else initial_position + vector

        Phi, alpha = get_Phi_alpha(vector, coords[i], target_position)
        Phi_queue.append(Phi)
        alpha_queue.append(alpha)

    return coords, vectors

'''
@njit
def guidance_law(parameters, experimental_data, tau, tau1, mode):
    """
    Note: because the deque is not filled initially, the first two dalpha_dt values will be the same.
    May want to address that by adding more data to the initial values possibly from BEFORE the bout starts. 
    Yeah, the solution is to add two initial values to the deque from before the bout starts I think.
    That way, even if tau is 0, the deque will have two values to calculate the derivative from.

    Parameters
    ----------
    experimental_data : Initial heading (complex), initial coordinates (complex), step_sizes (np.array), target_coordinates (complex np.array)
    parameters: (float, float, int) k, beta tau; negative gain parameter, constant bearing, and time delay"""
    tau = int(tau)
    tau1 = int(tau1)
    
    initial_heading, initial_position, step_sizes, target_positions = experimental_data
    # NEW
    n_steps = len(target_positions) - tau - tau1
    vectors, coords =  np.zeros((n_steps + 1, 1), dtype=np.complex128), np.zeros((n_steps, 1), dtype=np.complex128)
    vectors[0], coords[0] = initial_heading, initial_position
    
    # taking 1st instead of 0th because 0th is just used for computing derivative.
    # We do this because sometimes we input the first `tau` values

    theta = np.degrees(np.angle(initial_heading[1] if hasattr(initial_heading, '__len__') else initial_heading) )

    initial_Phi, initial_alpha = get_Phi_alpha(initial_heading, initial_position, 
        target_positions[:len(initial_position) if hasattr(initial_position, '__len__') else 1])

    Phi_queue = CircularQueue(tau+2)
    alpha_queue = CircularQueue(tau1+2)
    Phi_queue.append(initial_Phi)
    alpha_queue.append(initial_alpha)
    for i in range(n_steps):
        target_position = target_positions[i] # +tau+au1? I don't think so.
        step_size = step_sizes[i] #  + tau + tau1? I don't think so

        # Update heading based on previous direction
        # Taking the first Phi value because two values are needed to calculate the derivative
        # But only one value is needed to calculate the bearing. Thus the first value is ignored when not taking the derivative.
        # This is because the second value is the last value from the previous timestep, with tau delay.
        if mode == 0:
            dtheta_dt = par_nav_dtheta_dt(Phi_queue, alpha_queue, parameters)
        elif mode == 1:
            dtheta_dt = prop_pursuit_dtheta_dt(Phi_queue, alpha_queue, parameters)
        theta = (theta + dtheta_dt) % 360 # deg
        
        # Complex, replaced phasor() with the hard code.
        vector = step_size * np.exp(1j * np.deg2rad(theta))
        vectors[i] = vector
        coords[i+1] = coords[i] + vector

        # Appending pops the beginning of the array, so append after evaluting thetadot for the first time.
        # Only update the queue after the initial run so that there is not a fictive delay.
        Phi, alpha = get_Phi_alpha(vector, coords[i+1], target_position)
        Phi_queue.append(Phi)
        alpha_queue.append(alpha)

    # all but the last coordinate so that it's the 
    return coords[:-1-tau], vectors[:-1-tau]
'''
def D(sigma, A, Phi):
    """First term, direction sensitivity term"""
    Phi = np.deg2rad(Phi)
    return A * Phi / (np.sqrt(2 * np.pi) * sigma**3)*np.exp(-Phi ** 2 / (2 * sigma**2))


def r(sigma, A, Phi):
    """Second term, angular velocity gain term"""
    Phi = np.deg2rad(Phi)
    return A / (np.sqrt(2 * np.pi ) * sigma) * np.exp(-Phi**2/(2*sigma**2))


def get_thetadot_PR(Phi, Phidot, D_sigma, D_A,  r_sigma, r_A):
    return D(D_sigma, D_A,  Phi) + r(r_sigma, r_A, Phi) * Phidot


def PoggioReichart(parameters, experimental_data, tau):
    """Parameters
    ----------
    parameters: (OptimizationResult) res;
    experimental_data : Initial heading (complex), initial coordinates (complex), step_sizes (np.array), target_coordinates (complex np.array)
    """
    def PR_dt(Phi_queue, alpha_queue, res): 
        dPhi_dt = dangle_dt(np.array(Phi_queue)) # deg/frame
        dtheta_dt = get_thetadot_PR(Phi_queue[1], dPhi_dt[0], *res.x)
        return dtheta_dt

    coords, vectors = guidance_law(parameters, experimental_data, tau, PR_dt)

    return coords, vectors


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
    

def get_VAF(prediction, ground_truth):
    return 1-((np.var(prediction - ground_truth) / np.var(ground_truth)))


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


"""def aggregate_cost(params, experimental_data_l, tau, tau1, mode, max_workers=4):
    '''Parallelized version of aggregate_cost with constrained CPU usage.'''
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        errors = executor.map(compute_cost, [(params, data, tau, tau1, mode) for data in experimental_data_l])
    error_l = list(errors)
    total_avg_error = np.mean(error_l)
    return np.mean(error_l)
"""


def aggregate_cost(params, experimental_data_l, tau, tau1, mode):
    '''Example useage:
    result = minimize(aggregate_cost, initial_guess, args=(experimental_data_l, constant_bearing_interception_pursuit), method='Nelder-Mead')
    prop_pursuit_k_optimized, prop_pursuit_tau_optimized = result.x
    '''
    error_l = []
    for experimental_data in experimental_data_l:
        error_l.append(cost(params, experimental_data, tau, tau1, mode))
    # print(int(np.mean(error_l)), np.std(error_l).astype(int), params, tau, tau1, mode)

    return np.mean(error_l)


def cost(params, experimental_data, tau:int, tau1:int, mode:int):
    '''Example useage:'''
    # Get predicted data
    # print('params type:', type(params))
    # print('experimental_data types:', [type(e) for e in experimental_data])
    # print('experimental_data dtypes:', [getattr(e, 'dtype', None) for e in experimental_data])
    # print('tau type:', type(tau))
    # print('tau1 type:', type(tau1))
    # print('mode type:', type(mode))
    predicted_data, _ = guidance_law(params, experimental_data[:-1], tau, tau1, mode)
    # MAE
    # print(tau, tau1, predicted_data.shape, experimental_data[-1][:-tau if tau > 0 else None].shape)
    error = MAE(predicted_data, experimental_data[-1])
    return error


def get_guidance_law_df(experimental_data_l:list, pursuit_model:str,  initial_guess:tuple, tau:int, tau1:int):
    """Selects guidance law and appropriate constraints to optimize
    Parameters
    ----------
    experimental_data: (list of tuples) list of: initial_heading, init_position, step_sizes, target_positions, pos
    pursuit_model: (str) Pursuit model to fit:
        'classic_pursuit':
        'constant bearing':
        'proportional pursuit':
        'proportional navigation':
        'mixed_pursuit':
    initial_guess: (tuple) Initial guess for parameters
    tau(1): (int) tau for the model. tau is the time delay for the model.
        tau1 is the time delay for the model (for the second parameter).

    Returns
    -------
    parameter_df: (pd.DataFrame) Dataframe containing the optimized parameters
    """

    """Parameters for each model
    prop_pursuit: (k, beta)
    par_nav: (N)"""

    if pursuit_model == "proportional pursuit":
        mode=0
    elif pursuit_model == "proportional navigation":
        mode=1

    # Optimize
    result = minimize(aggregate_cost, initial_guess, args=(experimental_data_l, tau, tau1, mode), method='Powell') 
    # print(result.x, result.success, result.nit, result.message)
    # Add datetime metadata.
    parameter_df = pd.DataFrame(
        {"param":result.x, "result":result, 
        "converged": result.success, "nit":result.nit,
        "error":result.fun}, index=[0])
    
    return parameter_df


def shorten_experimental_data(experimental_data, start, interval=30):
    """Shortens experimental data to a second-long interval"""    
    # Shorten experimental data
    (initial_heading, init_position, step_sizes, target_positions, pos) = experimental_data
    
    # If there's anything left after start time, shorten it and return. Otherwise return None.
    # This will not run into an indexing error, I believe.
    if len(target_positions[start:]) > 1:
        shortened_experimental_data = (
            np.diff(pos[start:start+interval])[0],
            pos[start],
            step_sizes[start:start+interval],
            fill_nans_with_interpolation(target_positions[start:start+interval]),
            pos[start:start+interval])
    
        return shortened_experimental_data
    else:
        return None
        

def incremental_guidance_law(experimental_data, guidance_law_params, fps):
    """Runs get_guidance_law in second-long increments"""
    
    # For each increment
    parameter_df_l = []

    pos = experimental_data[-1]
    # Iterate through each start period
    starts = np.arange(0, len(pos), 30)
    for start_ind, start in enumerate(starts):
        short_expt_data = shorten_experimental_data(experimental_data, start)
        
        if (start_ind > 0) and (short_expt_data is None):
            break

        parameter_df = get_guidance_law_df(short_expt_data, *guidance_law_params)
        
        parameter_df["start (s)"] = np.round(start / fps, 2)
        parameter_df_l.append(parameter_df)

    return pd.concat(parameter_df_l)




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
        diff_pos = np.diff(pos)
        # The first heading is repeated so that the length
        # of headings is the same as the length of pos
        headings = np.concatenate(([diff_pos[0]], diff_pos))
    
    headings = np.angle(headings, deg=True)
    LOS = np.angle(target_positions-pos, deg=True)

    # head angle, omega_f, in degrees
    thetadot = dangle_dt(headings) # deg/frame
    thetadot = thetadot / IFI # deg/s

    # alphadot
    alphadot = dangle_dt(LOS) / IFI # deg/s

    # bearing, theta_e, in degrees
    phi = get_angular_difference(LOS, headings)
    phidot = dangle_dt(phi) / IFI # deg/s
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
            tan_thetadot = scipy.signal.medfilt(tan_thetadot, 5)
            tan_Phi = scipy.signal.medfilt(tan_Phi, 5)
            alphadot = scipy.signal.medfilt(alphadot, 5)
            tan_Phidot = scipy.signal.medfilt(tan_Phidot, 5)
            
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

