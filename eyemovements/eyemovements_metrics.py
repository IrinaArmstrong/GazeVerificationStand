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

    def __str__(self):
        return self._name


MetricType = NewType("MetricType", Metric)

# ----------------------- Saccades metrics -----------------------


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

        return np.mean([len(get_movement_indexes(session, GazeState.saccade))
                        for session in data])


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
            stimulus_sac = get_movement_indexes(stimulus_move, GazeState.saccade)
            stimulus_ampl_sum = (np.sum([get_amplitude_and_angle(stimulus_coord[sac])[0]
                                         for sac in stimulus_sac])
                                 + sys.float_info.epsilon)  # to prevent zero division

            # Gaze
            gaze_sac = get_movement_indexes(gaze_move, GazeState.saccade)
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
            # Gaze fixations
            gaze_fixations = get_movement_indexes(gaze_move, GazeState.fixation)

            # if in both: stimulus and gaze there are no fixations at all -> good, 100 grad.
            if (len(stimulus_fixations) == 0) and (len(gaze_fixations) == 0):
                logger.warning("During computation FQnS score no stimulus and gaze fixations detected. Score -> 100 grad.")
                fqns_estims.append(100.0)
                continue
            # if in stimulus there are no fixations at all, but in gaze are -> bad, 0 grad.
            elif (len(stimulus_fixations) == 0) and (len(gaze_fixations) > 0):
                logger.warning("During computation FQlS score no stimulus fixations detected, but in gaze found few.")
                logger.warning("Score -> 0 grad.")
                fqns_estims.append(0.0)
                continue
            elif (len(stimulus_fixations) > 0) and (len(gaze_fixations) == 0):
                logger.warning("During computation FQlS score no gaze fixations detected, but in stimulus found few.")
                logger.warning("Score -> 0 grad.")
                fqns_estims.append(0.0)
                continue
            else:
                stimulus_fix_points_num = len((stimulus_fixations == GazeState.fixation).nonzero())
                stimulus_prev_saccades = [get_previous_saccade(stimulus_move, fix[0]) for fix in stimulus_fixations]
                default_sac_amplitude = np.mean([get_amplitude_and_angle(stimulus_coord[prev_sac])[0]
                                                 for prev_sac in stimulus_prev_saccades
                                                 if len(prev_sac) > 0])

            fixations_detected_cnt = 0
            for stim_fix_idxs, prev_sac in zip(stimulus_fixations, stimulus_prev_saccades):
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
                            # get gaze movement centroid as (x, y) coordinates
                            (xc, yc) = get_path_and_centroid(gaze_coord[detected_fix])[1:]
                            # stimulus movement as centroid
                            (xs, ys) = stimulus_coord[idx]
                            # compare
                            if euclidean((xs, ys), (xc, yc)) <= amplitude_coefficient * sac_amplitude:
                                fixations_detected_cnt += 1

            fqns = 100 * (fixations_detected_cnt / stimulus_fix_points_num)
            fqns_estims.append(fqns)
        return fqns_estims


class FQlS(Metric):
    """
    The Fixation Qualitative Score (FQlS) compares the spatial proximity
    of the classified eye fixation signal to the presented stimulus signal,
    therefore indicating the positional accuracy or error of the classified fixations.

    ! Mark up (expert assessment or the most probable eye movements on given stimuli path)
    of stimulus data should be given !
    """

    def __init__(self):
        super().__init__("Fixation Qualitative Score (FQlS)")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        kwargs: 'gaze_coordinates' - gaze x and y coordinates (as 2 dim. array)
                'stimulus_eyemovements' - stimulus eye movements - expert assessment
                                        or the most probable eye movements on given stimuli path (as 1 dim. array)
                'stimulus_coordinates' - stimulus x and y coordinates (as 2 dim. array),
                                        optional - if not given 'gaze_coordinates' would be used.
                `averaging_strategy` - way to average estimates over all sessions: `micro` or `macro`.
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

        averaging_strategy = kwargs.get("averaging_strategy", 'macro')
        if averaging_strategy not in ['micro', 'macro']:
            logger.error(f"Averaging parameter `averaging_strategy` is not one from available: ['micro', 'macro']")
            logger.warning(f"Setting to `macro`")
            averaging_strategy = 'macro'

        # For each session
        fqls_estims = []

        for gaze_move, gaze_coord, stimulus_move, stimulus_coord in zip(data, gaze_coordinates,
                                                                        stimulus_eyemovements, stimulus_coordinates):
            # Stimulus fixations
            stimulus_fixations = get_movement_indexes(stimulus_move, GazeState.fixation)
            # Gaze fixations
            gaze_fixations = get_movement_indexes(gaze_move, GazeState.fixation)

            # if in both: stimulus and gaze there are no fixations at all -> good, 0 grad.
            if (len(stimulus_fixations) == 0) and (len(gaze_fixations) == 0):
                logger.warning("During computation FQlS score no stimulus and gaze fixations detected. Score -> 0 grad.")
                fqls_estims.append(0.0)
                continue
            # if in stimulus there are no fixations at all, but in gaze are -> bad, inf. grad.
            elif (len(stimulus_fixations) == 0) and (len(gaze_fixations) > 0):
                logger.warning("During computation FQlS score no stimulus fixations detected, but in gaze found few.")
                logger.warning("Score -> inf. grad.")
                fqls_estims.append(np.inf)
                continue
            elif (len(stimulus_fixations) > 0) and (len(gaze_fixations) == 0):
                logger.warning("During computation FQlS score no gaze fixations detected, but in stimulus found few.")
                logger.warning("Score -> inf. grad.")
                fqls_estims.append(np.inf)
                continue
            else:
                stimulus_fix_points_num = len((stimulus_fixations == GazeState.fixation).nonzero())
                stimulus_prev_saccades = [get_previous_saccade(stimulus_move, fix[0]) for fix in stimulus_fixations]
                default_sac_amplitude = np.mean([get_amplitude_and_angle(stimulus_coord[prev_sac])[0]
                                                 for prev_sac in stimulus_prev_saccades
                                                 if len(prev_sac) > 0])
            # get gaze movement centroid as (x, y) coordinates
            gaze_fix_centroids = [get_path_and_centroid(gaze_coord[fix])[1:] for fix in gaze_fixations]

            fixations_dists_list = []
            # Iterate over all found fixations
            for fixations_idxes in stimulus_fixations:
                for fix_idx in fixations_idxes:
                    # If on the same state in gaze eye movements there is a fixation
                    if gaze_move[fix_idx] == GazeState.fixation:
                        # get full movement indexes
                        detected_fix = get_movement_for_index(fix_idx, gaze_move)
                        if len(detected_fix) > 0:
                            # get gaze movement centroid as (x, y) coordinates
                            (xc, yc) = get_path_and_centroid(gaze_coord[detected_fix])[1:]
                            # stimulus movement as centroid
                            (xs, ys) = stimulus_coord[fix_idx]
                            fixations_dist = euclidean((xs, ys), (xc, yc))
                            # Compare if found fixation point is close
                            if fixations_dist <= amplitude_coefficient * default_sac_amplitude:
                                # Computes the Euclidean distance
                                fixations_dists_list.append(fixations_dist)
            if averaging_strategy == 'macro':
                fqls_estims.append(np.mean(fixations_dists_list))
            else:
                fqls_estims.extend(fixations_dists_list)

        return np.mean(fqls_estims)

# -------------------- Smooth Persuite Metrics ----------------


class AverageSPNumber(Metric):
    """
    Count average number of Smooth Persuites.
    """

    def __init__(self):
        super().__init__("Average Smooth Persuite Number")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        """
        return np.mean([len(get_movement_indexes(session, GazeState.sp)) for session in data])


class PQlS(Metric):
    """
    The intuitive idea behind the smooth pursuit qualitative scores (PQlS)
    is to compare the proximity of the detected SP signal with the signal
    presented in the stimuli.
    Two scores are indicative of positional (PQlS_P) and velocity (PQlS_V) accuracy.

    ! Mark up (expert assessment or the most probable eye movements on given stimuli path)
    of stimulus data should be given !
    """

    def __init__(self):
        super().__init__("Smooth Pursuit Qualitative Scores (PQlS)")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        - `data` - gaze eye movements
        kwargs: 'gaze_coordinates' - gaze x and y coordinates (as 2 dim. array)
                'stimulus_eyemovements' - stimulus eye movements - expert assessment
                                        or the most probable eye movements on given stimuli path (as 1 dim. array)
                'stimulus_coordinates' - stimulus x and y coordinates (as 2 dim. array),
                                        optional - if not given 'gaze_coordinates' would be used.
                `gaze_velocity` - gaze velocity as sqrt(vel_x^2 + vel_y^2)
                `stimulus_velocity` - stimulus velocity as sqrt(vel_x^2 + vel_y^2),
                                    optional - if not given 'gaze_velocity' would be used.
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

        gaze_velocity = kwargs.get("gaze_velocity", None)
        if stimulus_eyemovements is None:
            logger.error(f"Necessary parameter `gaze_velocity` was not provided in **kwargs.")
            raise AttributeError

        stimulus_velocity = kwargs.get("stimulus_velocity", gaze_velocity)

        # For each session
        pqls_p_estims = []
        pqls_v_estims = []

        for pack in zip(data, gaze_coordinates, gaze_velocity,
                        stimulus_eyemovements, stimulus_coordinates, stimulus_velocity):
            # Unpack tuple
            gaze_move, gaze_coord, gaze_vel, stimulus_move, stimulus_coord, stim_vel = pack

            # Stimulus sp
            stimulus_sp = get_movement_indexes(stimulus_move, GazeState.sp)
            # Gaze
            gaze_sp = get_movement_indexes(gaze_move, GazeState.sp)

            if (len(stimulus_sp) == 0) and (len(gaze_sp) == 0):
                logger.warning("During computation PQlS score no stimulus and gaze SP detected. Scores -> 0 grad.")
                pqls_p_estims.append(0.0)
                pqls_v_estims.append(0.0)
                continue
            # If there are no SP in stimulus and there are detected some in gaze -> bad, inf. score.
            elif (len(stimulus_sp) == 0) and (len(gaze_sp) > 0):
                logger.warning("During computation PQlS score no stimulus SP, but some detected in gaze .")
                logger.warning("Scores -> inf. grad.")
                pqls_p_estims.append(np.inf)
                pqls_v_estims.append(np.inf)
                continue
            # If there are SP in stimulus and there are nothing detected in gaze -> bad, inf. score.
            elif (len(stimulus_sp) > 0) and (len(gaze_sp) == 0):
                logger.warning("During computation PQlS score no gaze SP detected, but there some in stimulus.")
                logger.warning("Scores -> inf. grad.")
                pqls_p_estims.append(np.inf)
                pqls_v_estims.append(np.inf)
                continue

            sp_detected_cnt = 0
            vel_diff = 0
            dist_diff = 0

            # Iterate over found SPs
            for sp_idxes in stimulus_sp:
                for sp_idx in sp_idxes:
                    # If on the same state in gaze eye movements there is a SP
                    if gaze_move[sp_idx] == GazeState.sp:
                        sp_detected_cnt += 1
                        vel_diff += np.abs(stim_vel[sp_idx] - gaze_vel[sp_idx])
                        dist_diff += euclidean(stimulus_coord[sp_idx], gaze_coord[sp_idx])
            if sp_detected_cnt > 0:
                pqls_p = dist_diff / sp_detected_cnt
                pqls_v = vel_diff / sp_detected_cnt
            else:
                pqls_p = np.inf
                pqls_v = np.inf

            pqls_p_estims.append(pqls_p)
            pqls_v_estims.append(pqls_v)

        return np.mean(pqls_p_estims), np.mean(pqls_v_estims)


class PQnS(Metric):
    """
    The smooth pursuit quantitative score (PQnS) measures the amount of detected SP behavior given
    the SP behavior encoded in the stimuli.

    ! Mark up (expert assessment or the most probable eye movements on given stimuli path)
    of stimulus data should be given !
    """

    def __init__(self):
        super().__init__("Smooth Pursuit Quantitative Score (PQnS)")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        - `data` - gaze eye movements
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
        pqns_estims = []

        for gaze_move, gaze_coord, stimulus_move, stimulus_coord in zip(data, gaze_coordinates,
                                                                        stimulus_eyemovements, stimulus_coordinates):
            # Stimulus SP
            stimulus_sp = get_movement_indexes(stimulus_move, GazeState.sp)
            # Gaze SP
            gaze_sp = get_movement_indexes(gaze_move, GazeState.sp)

            if len(stimulus_sp) == 0:
                logger.warning("During computation PQnS score no stimulus SP detected")
                stimulus_paths = 0
            else:
                stimulus_paths = [get_path_and_centroid(stimulus_coord[sp])[0] for sp in stimulus_sp]

            # If there are no SP in stimulus detected
            if len(gaze_sp) == 0:
                logger.warning("During computation PQlS score no gaze SP detected")
                gaze_paths = 0
            else:
                gaze_paths = [get_path_and_centroid(gaze_coord[sp])[0] for sp in gaze_sp]

            pqns = 100 * (np.sum(gaze_paths) / (np.sum(stimulus_paths)
                                                + sys.float_info.epsilon))  # to prevent zero division
            pqns_estims.append(pqns)

        return np.mean(pqns_estims)


class MisFix(Metric):
    """
    Misclassification error of the SP can be determined during a fixation stimulus,
    when correct classification is most challenging.

    ! Mark up (expert assessment or the most probable eye movements on given stimuli path)
    of stimulus data should be given !
    """

    def __init__(self):
        super().__init__("Misclassification Error SP/Fixation (MisFix)")

    def estimate(self, data: List[np.ndarray], **kwargs) -> Any:
        """
        Estimates on Dataset - list of sessions.
        - `data` - gaze eye movements
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
        misfix_estims = []
        for gaze_move, gaze_coord, stimulus_move, stimulus_coord in zip(data, gaze_coordinates,
                                                                        stimulus_eyemovements, stimulus_coordinates):
            # Stimulus fixations
            stimulus_fixations = list(chain.from_iterable(get_movement_indexes(stimulus_move, GazeState.fixation)))
            if len(stimulus_fixations) == 0:
                stimulus_fix_points_num = 0
            else:
                stimulus_fix_points_num = len((stimulus_fixations == GazeState.fixation).nonzero())
            # Gaze
            gaze_sp = list(chain.from_iterable(get_movement_indexes(gaze_move, GazeState.sp)))

            # No SP detected -> no mistakes
            if len(gaze_sp) == 0:
                misfix_estims.append(0.0)
                continue

            # Calculate error
            mis_class = len([stim_fix_ind for stim_fix_ind in stimulus_fixations
                             if stim_fix_ind in gaze_sp])
            misfix = 100 * (mis_class / (stimulus_fix_points_num + sys.float_info.epsilon))  # to prevent zero division
            misfix_estims.append(misfix)

        return np.mean(misfix_estims)

# --------------------------- All Metrics ---------------------------

all_metrics_list = [AverageSaccadesNumber, AverageSaccadesAmplitude, AverageSaccadesMagnitude, SQnS,  # saccedes
                    AverageFixationsNumber, AverageFixationsDuration, FQlS, FQnS,  # fixations
                    AverageSPNumber, PQnS, PQlS, MisFix]  # sp