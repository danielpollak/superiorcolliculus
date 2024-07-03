import numpy as np
from superiorcolliculus.processing import *

def test_vectorized_schmidt_trigger():
    input_signal = np.array([1, 2, 7, 7, 7, 2, 1, 7, 1])
    lower_threshold = 2
    upper_threshold = 6
    expected_output = np.array([
        False, False,  True,  True,  True,  True, False,  True, False])
    input = vectorized_schmidt_trigger(
        input_signal, lower_threshold, upper_threshold
    )
    assert np.array_equal(input, expected_output)



def test_extract_datetime():
    input_string = "06272023115155"
    assert get_date_time(input_string) == np.datetime64("2023-06-27 11:51:55")
