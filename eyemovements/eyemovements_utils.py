# Basic
import traceback
import numpy as np
import pandas as pd
from itertools import chain
from scipy.spatial.distance import euclidean
from typing import (List, Dict, Any, Tuple, TypeVar, Union, Iterable)

import warnings
warnings.filterwarnings('ignore')

import logging_handler
logger = logging_handler.get_logger(__name__)

GazeAnalyzerType = TypeVar("GazeAnalyzerType", bound='GazeAnalyzer')
GazeStateType = TypeVar("GazeStateType", bound="GazeState")


class GazeState:
    unknown = 0
    fixation = 1
    saccade = 2
    sp = 3
    error = 4

    @classmethod
    def get(cls, attr_name: str):
        return cls.__dict__.get(attr_name)

    @classmethod
    def decode(cls, attr_num: int):
        attr_name = [key for key, val in dict(GazeState.__dict__).items() if val == attr_num]
        return attr_name[0] if len(attr_name) > 0 else "unknown"


def get_movement_indexes(movements: np.ndarray,
                         movement_type: int) -> List[List[int]]:
    """
    Get lists of consecutive points marked as eye movements.
    :param movements: list or array-like
    :param movement_type: type to look for in data list
    :return: list of lists of indexes
    """
    indexes = []
    list_i = []
    for i, m in enumerate(movements):
        # skip another movements
        if m != movement_type:
            if len(list_i) > 0:
                indexes.append(list_i)
                list_i = []
            continue
        list_i.append(i)
    if len(list_i) > 0:
        indexes.append(list_i)
    if len(indexes) == 0:
        logger.warning(f"No {GazeState.decode(movement_type)} type detected.")
        return []
    return indexes


def get_sp_moves_dataset(data: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Select from given data only SP moves and forms dataset.
    :param data: classified data
    :return: dataframe with only SP moves.
    """
    sps = []
    for df in data:
        moves_sp = get_movement_indexes(df['movements'], GazeState.sp)
        if len(moves_sp) > 0:
            for sp_ids in moves_sp:
                sps.append(df.iloc[sp_ids])
    for i, sp_df in enumerate(sps):
        sp_df.loc[:, 'move_id'] = i

    logger.info(f"In classified sessions there are {len(sps)} with total length: {np.sum([len(s) for s in sps])} SP.")
    sps = pd.concat(sps, ignore_index=True)
    return sps


def clean_short_movements(movements: np.ndarray,
                          timestamps: np.ndarray,
                          movements_type: int,
                          threshold_clean: float) -> np.ndarray:
    """
    Cleaning small inserts of some movement in gaze data.
    For example:
        if in gaze row is small saccade (< 12 ms.) between two fixations/sp, then
        mark all poits of this saccade as fixation/sp.
    """
    moves = get_movement_indexes(movements, movements_type)

    for move_idxs in moves:
        try:
            if (timestamps[move_idxs[-1]] - timestamps[move_idxs[0]]) < threshold_clean:
                logger.info(f"Found too small {GazeState.decode(movements_type)}: {move_idxs}")
                prev_type = movements[move_idxs[0] - 1] if (move_idxs[0] - 1) > 0 else move_idxs[0]
                post_type = movements[move_idxs[-1] + 1] if (move_idxs[-1] + 1) < len(movements) else move_idxs[-1]
                if prev_type == post_type:
                    movements[move_idxs] = prev_type
                else:
                    movements[move_idxs] = GazeState.unknown
        except Exception as e:
            print(f"Error occurred: {traceback.print_tb(e.__traceback__)}")

    return movements


def merge_consecutive_movements(movements: np.ndarray,
                                movement_type: int) -> List[List[int]]:
    """
    Merge consecutive eye movements in one (fix/sp/saccade).
    """
    moves = get_movement_indexes(movements, movement_type)
    all_moves_ids = chain.from_iterable(moves)

    all_res_moves = []
    prev_id = 0
    move = [prev_id]
    for curr_id in all_moves_ids:
        # if they are consecutive
        if curr_id == prev_id + 1:
            move.append(curr_id)
        else:
            all_res_moves.append(move)
            move = [curr_id]
    return all_res_moves


def filter_errors(movements: np.ndarray,
                  valid_flgs: np.ndarray) -> Union[Iterable, tuple]:
    """
    Filter erroneous data samples.
    :param movements: list or array-like
    :param valid_flgs: errors list (1 - ok, 0 - error)
    :return: list of indexes of filtered points (except erroneous samples)
    """
    def inverse_valid_flg(valid_flg: int):
        return 0 if valid_flg == 1 else 1

    mask = np.ones((len(movements),), dtype=np.int32)  # all eye movements detected
    unique_moves = [m for m in np.unique(movements) if m not in [GazeState.error, GazeState.unknown]]
    moves_types = chain.from_iterable([get_movement_indexes(movements, m) for m in unique_moves])
    inverse = np.vectorize(inverse_valid_flg)
    errors = inverse(valid_flgs)

    for i, move_idxs in enumerate(moves_types):

        # discard movements with more than 50% erroneous samples
        percentage_error = np.sum(errors[move_idxs]) / float(len(move_idxs))
        if percentage_error >= 0.5:
            print(f"[INFO-FILTER]: Errors in indexes: {move_idxs} more then 50% of samples.")
            mask[move_idxs] = 0

    return np.nonzero(mask)


def filter_arrays(data: Dict[str, Any], mask: List[int]) -> Dict[str, Any]:
    """
    Filter all data arrays with given mask.
    """
    _ = [data.update({key: val[mask]}) for key, val in data.items() if type(val) == np.ndarray]
    return data


def get_previous_saccade(movements: np.ndarray,
                         fix_start_index: int) -> List[int]:
    """
    Select previous saccade movement (before current fixation)
    :return: previous saccade indexes in given eye movements.
    """
    i = fix_start_index - 1
    saccade = []
    while i >= 0:
        # yet another point to saccade
        if movements[i] == GazeState.saccade:
            saccade.append(i)
            i -= 1
        # found saccade ended, return
        elif (movements[i] != GazeState.saccade) and (len(saccade) > 0):
            return saccade
        else:
            i -= 1
    logger.info("No saccades were found in eye movements")
    return list(reversed(saccade))


def get_movement_for_index(index: int,
                           movements: np.ndarray) -> List[int]:
    """
    Get full movement indexes for given single point of eye movement (as index).
    """
    state = movements[index]
    prev_state = state
    start_index, end_index = index, index

    while (prev_state == state) and (start_index >= 0):
        start_index -= 1
        prev_state = movements[start_index]

    prev_state = movements[index]
    while (prev_state == state) and (end_index < (len(movements) - 1)):
        end_index += 1
        prev_state = movements[end_index]

    return list(range(start_index + 1, end_index, 1))


def x_axis_angle(point_start: Tuple[float, float],
                 point_end: Tuple[float, float]) -> float:
    """
    Calculate angles created by the pairs of points and horizontal axis
    """
    vec = np.subtract(point_end, point_start)
    norm = np.linalg.norm(vec)
    vec = np.divide(vec, norm)

    if (vec[0] == 0) or (norm == 0):
        radians = 0
    else:
        radians = np.arctan(np.true_divide(vec[1], vec[0]))

    if vec[0] > 0:
        if vec[1] < 0:
            radians += (2 * np.pi)
        else:
            radians = np.pi + radians
    return radians


def angle_unit_coords(radians: float) -> Tuple[float, float]:
    """
    Represent computed angles as points on circumference of a unit circle
    and return their xp and yp coordinates on x and y axes.
    (degrees = radians * (np.pi / 180))
    """
    # as xp = l*cos(alpha), l=1 => xp=cos(alpha)
    xp = np.cos(radians)
    # as yp = l*cos(pi/2 - alpha), l=1 => yp=sin(alpha)
    yp = np.sin(radians)
    return (xp, yp)


def get_amplitude_and_angle(gaze: np.ndarray) -> List[float]:
    """
    Calculate amplitude and angle of saccade (in radians).
    """
    dx = gaze[-1, 0] - gaze[0, 0]
    dy = gaze[-1, 1] - gaze[0, 1]

    amplitude = np.linalg.norm([dx, dy])

    if dx == 0:
        radians = 0
    else:
        radians = np.arctan(np.true_divide(dy, dx))

    if dx > 0:
        if dy < 0:
            radians += (2 * np.pi)
        else:
            radians = np.pi + radians
    return [amplitude, radians]


def get_path_and_centroid(gaze: np.ndarray) -> List[float]:
    """
    Calculate centroid and path length.
    """
    last_x, last_y = 0, 0
    sumx, sumy = 0, 0
    distance = 0
    for i, (gaze_x, gaze_y) in enumerate(gaze):
        # first sample
        if i == 0:
            center_x = gaze_x
            center_y = gaze_y
            last_x = gaze_x
            last_y = gaze_y
        else:
            center_x = np.true_divide(sumx, i)
            center_y = np.true_divide(sumy, i)

        # update distance
        distance += np.sqrt((gaze_x - last_x) * (gaze_x - last_x)
                            + (gaze_y - last_y) * (gaze_y - last_y))
        sumx += gaze_x
        sumy += gaze_y
    return [distance, center_x, center_y]


def get_closest_centroid(centroid: List[float],
                         centroids_list: List[List[float]]):
    """
    ???
    """
    dists = [euclidean(centroid, cc) for cc in centroids_list]
    return np.argmax(dists), np.max(dists)
