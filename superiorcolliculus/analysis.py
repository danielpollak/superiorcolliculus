import numpy as np
import pandas as pd
import cv2
from .processing import *


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
        print("Error: Couldn't open the video file.")
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
