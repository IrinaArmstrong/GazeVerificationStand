import sys
import pandas as pd
import numpy as np
from itertools import chain
from abc import ABC, abstractmethod
from scipy.spatial.distance import euclidean
from typing import List, Any, Union, NewType

from eyemovements.eyemovements_utils import (get_movement_indexes, GazeState,
                                             get_amplitude_and_angle, get_path_and_centroid,
                                             get_previous_saccade, get_movement_for_index)

import logging_handler
logger = logging_handler.get_logger(__name__)


class Metric(ABC):

    def __init__(self, metric_name: str):
        self._name = metric_name

    @abstractmethod
    def estimate(self, data: Union[List[np.ndarray], np.ndarray], **kwargs) -> Any:
        pass


Metric = NewType("Metric", Metric)


class AverageSaccadesNumber(Metric):
    """
    Average number of saccades.
    """

    def __init__(self):
        super().__init__("Average Saccades Number")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        """
        return np.mean([len(get_movement_indexes(session, GazeState.saccade)) for session in data])


class AverageSaccadesAmplitude(Metric):
    """
    Average amplitude of saccades.
    By default Frobenius norm: np.sum(np.abs(x)**2)**(1./2)
    """

    def __init__(self):
        super().__init__("Average Saccades Amplitude")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        kwargs: 'gaze_coordinates' - gaze x and y coordinates (as 2 dim. array)
        """
        return np.mean([get_amplitude_and_angle(gaze[sac])[0]
                        for movements, gaze in zip(data, kwargs.get('gaze_coordinates'))
                        for sac in get_movement_indexes(movements, GazeState.saccade)])


class AverageSaccadesMagnitude(Metric):
    """
    Average magnitude of saccades.
    As arctan of angle, in radians.
    """

    def __init__(self):
        super().__init__("Average Saccades Magnitude")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        kwargs: 'gaze_coordinates' - gaze x and y coordinates (as 2 dim. array)
        """
        return np.mean([get_amplitude_and_angle(gaze[sac])[1]
                        for movements, gaze in zip(data, kwargs.get('gaze_coordinates'))
                        for sac in get_movement_indexes(movements, GazeState.saccade)])


class SQnS(Metric):
    """
    The Saccade Quantitative Score (SQnS) represents
    the amount of classified saccadic behavior given
    the amount of saccadic behavior encoded in the stimuli.

    ! Mark up (expert assessment or the most probable eye movements on given stimuli path)
    of stimulus data should be given !
    """

    def __init__(self):
        super().__init__("Saccade Quantitative Score (SQnS)")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        kwargs: 'gaze_coordinates' - gaze x and y coordinates (as 2 dim. array)
                'stimulus_eyemovements' - stimulus eye movements - expert assessment
                                        or the most probable eye movements on given stimuli path (as 1 dim. array)
                'stimulus_coordinates' - stimulus x and y coordinates (as 2 dim. array),
                                        optional - if not given 'gaze_coordinates' would be used.
        """
        gaze_coordinates = kwargs.get("gaze_coordinates", None)
        if gaze_coordinates is None:
            logger.error(f"Necessary parameter `gaze_coordinates` was not provided in **kwargs.")
            raise AttributeError

        stimulus_eyemovements = kwargs.get("stimulus_eyemovements", None)
        if stimulus_eyemovements is None:
            logger.error(f"Necessary parameter `stimulus_eyemovements` was not provided in **kwargs.")
            raise AttributeError

        stimulus_coordinates = kwargs.get("stimulus_coordinates", gaze_coordinates)

        # For each session
        sqns_estims = []
        for gaze_move, gaze_coord, stimulus_move, stimulus_coord in zip(data, gaze_coordinates,
                                                                        stimulus_eyemovements, stimulus_coordinates):
            # Stimulus
            stimulus_sac = get_movement_indexes(stimulus_coord, GazeState.saccade)
            stimulus_ampl_sum = (np.sum([get_amplitude_and_angle(stimulus_coord[sac])[0]
                                         for sac in stimulus_sac])
                                 + sys.float_info.epsilon)  # to prevent zero division

            # Gaze
            gaze_sac = get_movement_indexes(gaze_coord, GazeState.saccade)
            gaze_ampl_sum = (np.sum([get_amplitude_and_angle(gaze_coord[sac])[0]
                                     for sac in gaze_sac])
                             + sys.float_info.epsilon)  # to prevent zero division
            # Count SQnS
            sqns = 100 * (gaze_ampl_sum / stimulus_ampl_sum)
            sqns_estims.append(sqns)

        return sqns_estims

# --------------- Fixations metrics ------------------------------


class AverageFixationsNumber(Metric):
    """
    Count average number of fixations.
    """

    def __init__(self):
        super().__init__("Average Fixations Number")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        """
        return np.mean([len(get_movement_indexes(session, GazeState.fixation)) for session in data])


class AverageFixationsDuration(Metric):
    """
    Average duration (date-time units) of fixations.
    """

    def __init__(self):
        super().__init__("Average Fixations Duration")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        kwargs: 'timestamps' - gaze x and y coordinates (as 2 dim. array)
        """
        timestamps = kwargs.get('timestamps', None)
        if timestamps is None:
            logger.error(f"Necessary parameter `timestamps` was not provided in **kwargs.")
            raise AttributeError

        fix_durations = list(chain.from_iterable([[ts[fix[-1]] - ts[fix[0]]
                                                   for fix in get_movement_indexes(movements, GazeState.fixation)]
                                                  for movements, ts in zip(data, timestamps)]))
        if len(fix_durations) == 0:
            return 0.0
        return np.mean(fix_durations)


class FQnS(Metric):
    """
    Fixation Quantitative Score (FQnS) compares the amount of
    detected fixational behavior to the amount of
    fixational behavior encoded in the stimuli.

    ! Mark up (expert assessment or the most probable eye movements on given stimuli path)
    of stimulus data should be given !
    """

    def __init__(self):
        super().__init__("Fixation Quantitative Score (FQnS)")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        kwargs: 'gaze_coordinates' - gaze x and y coordinates (as 2 dim. array)
                'stimulus_eyemovements' - stimulus eye movements - expert assessment
                                        or the most probable eye movements on given stimuli path (as 1 dim. array)
                'stimulus_coordinates' - stimulus x and y coordinates (as 2 dim. array),
                                        optional - if not given 'gaze_coordinates' would be used.
        """
        gaze_coordinates = kwargs.get("gaze_coordinates", None)
        if gaze_coordinates is None:
            logger.error(f"Necessary parameter `gaze_coordinates` was not provided in **kwargs.")
            raise AttributeError

        stimulus_eyemovements = kwargs.get("stimulus_eyemovements", None)
        if stimulus_eyemovements is None:
            logger.error(f"Necessary parameter `stimulus_eyemovements` was not provided in **kwargs.")
            raise AttributeError

        stimulus_coordinates = kwargs.get("stimulus_coordinates", gaze_coordinates)
        amplitude_coefficient = kwargs.get("amplitude_coefficient", 1/3)

        # For each session
        fqns_estims = []
        for gaze_move, gaze_coord, stimulus_move, stimulus_coord in zip(data, gaze_coordinates,
                                                                        stimulus_eyemovements, stimulus_coordinates):
            # Stimulus fixations
            stimulus_fixations = get_movement_indexes(stimulus_move, GazeState.fixation)
            stimulus_fix_points_num = len(stimulus_fixations == GazeState.fixation)
            # Centroids as (x, y) coordinates
            stimulus_fix_centroids = [get_path_and_centroid(stimulus_coord[fix])[1:] for fix in stimulus_fixations]
            stimulus_prev_saccades = [get_previous_saccade(stimulus_move, fix[0]) for fix in stimulus_fixations]
            default_sac_amplitude = np.mean([get_amplitude_and_angle(stimulus_coord[prev_sac])[0]
                                             for prev_sac in stimulus_prev_saccades
                                             if len(prev_sac) > 0])

            fixations_detected_cnt = 0
            for stim_fix_idxs, centroid, prev_sac in zip(stimulus_fixations,
                                                         stimulus_fix_centroids,
                                                         stimulus_prev_saccades):
                if len(prev_sac) > 0:
                    sac_amplitude = get_amplitude_and_angle(stimulus_coord[prev_sac])[0]
                else:
                    sac_amplitude = default_sac_amplitude

                for idx in stim_fix_idxs:
                    # If on the same state in gaze eye movements there is a fixation
                    if gaze_move[idx] == GazeState.fixation:
                        # get full movement indexes
                        detected_fix = get_movement_for_index(idx, gaze_move)
                        if len(detected_fix) > 0:
                            # get gaze movement centroid
                            (xc, yc) = get_path_and_centroid(gaze_coord[detected_fix])[1:]
                            # stimulus movement centroid
                            (xs, ys) = stimulus_coord[idx]
                            # compare
                            if euclidean((xs, ys), (xc, yc)) <= amplitude_coefficient * sac_amplitude:
                                fixations_detected_cnt += 1

            fqns = 100 * (fixations_detected_cnt / stimulus_fix_points_num)
            fqns_estims.append(fqns)
        return fqns_estims
