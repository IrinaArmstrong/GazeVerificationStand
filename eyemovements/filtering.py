import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter
from typing import List

import logging_handler
logger = logging_handler.get_logger(__name__)


def sgolay2d(z: np.ndarray, window_size: int, order: int, derivative=None):
    """
    Savitsky-Golay filter to smooth two dimensional data.
    1. for each point of the two dimensional matrix extract a sub-matrix,
        centered at that point and with a size equal to an odd number "_window_size".
    2. for this sub-matrix compute a least-square fit of a polynomial surface,
        defined as p(x,y) = a0 + a1*x + a2*y + a3*x\^2 + a4*y\^2 + a5*x*y + ... .
        Note that x and y are equal to zero at the central point.
    3. replace the initial central point with the value computed with the fit.
    """
    # Number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0
    logger.debug(f"Number of terms in the polynomial expression: {n_terms}")

    if window_size % 2 == 0:
        raise ValueError('_window_size must be odd')

    if window_size ** 2 < n_terms:
        logger.error(f"{order} order is too high for the window size = {window_size}")
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # Exponents of the polynomial:
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # Coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size ** 2, )

    # Build matrix of system of equation
    A = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # Pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros((new_shape))
    # Top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(np.flipud(z[1:half_size + 1, :]) - band)
    # Bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size - 1:-1, :]) - band)
    # Left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1:half_size + 1]) - band)
    # Right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size - 1:-1]) - band)
    # Central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # Top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1:half_size + 1, 1:half_size + 1])) - band)
    # Bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(np.flipud(np.fliplr(z[-half_size - 1:-1, -half_size - 1:-1])) - band)

    # Top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(np.flipud(Z[half_size + 1:2 * half_size + 1, -half_size:]) - band)
    # Bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(np.fliplr(Z[-half_size:, half_size + 1:2 * half_size + 1]) - band)

    # Solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')

    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')

    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')

    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')


def sgolay_filter_dataset(dataset: List[pd.DataFrame], window_size: int = 15,
                          order: int = 6, derivative=None):
    """
    Run filtering on full dataset.
    :param dataset: in form of list of sessions
    :param window_size: odd number > order
    :param order: < _window_size
    :param derivative: 'col' or 'row', default None - so no deriviate taken
    :return: dataset with velocity, acceleration and filtered raw data.
    """
    for data in tqdm(dataset):
        filtered_gaze = sgolay2d(data[['gaze_X','gaze_Y']].values.reshape(-1, 2), window_size, order)
        velocity = sgolay2d(filtered_gaze, window_size, order, derivative)
        acceleration = sgolay2d(velocity, window_size, order, derivative)
        stimulus_velocity = sgolay2d(data[['stim_X','stim_Y']].values.reshape(-1, 2), window_size, order, derivative)

        data['filtered_X'] = filtered_gaze[:, 0]
        data['filtered_Y'] = filtered_gaze[:, 1]
        data['velocity_X'] = velocity[:, 0]
        data['velocity_Y'] = velocity[:, 1]
        data['velocity_sqrt'] = np.sqrt(np.power(velocity[:, 1], 2)
                                        + np.power(velocity[:, 0], 2))

        data['stimulus_velocity'] = np.sqrt(np.power(stimulus_velocity[:, 1], 2)
                                            + np.power(stimulus_velocity[:, 0], 2))

        data['acceleration_X'] = acceleration[:, 0]
        data['acceleration_X'] = acceleration[:, 1]
        data['acceleration_sqrt'] = np.sqrt(np.power(acceleration[:, 1], 2) + np.power(acceleration[:, 0], 2))
    return dataset