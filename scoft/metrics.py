import numpy as np


def mad(data) -> np.dtype:

    mean = np.mean(data)
    deviation = data - mean
    abs_deviation = np.abs(deviation)
    mad = np.mean(abs_deviation)

    return mad