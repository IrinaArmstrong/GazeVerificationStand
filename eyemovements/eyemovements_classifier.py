# Basic
import numpy as np
from itertools import chain
from typing import List, Tuple

import warnings
warnings.filterwarnings('ignore')

from eyemovements.eyemovements_utils import (GazeState, GazeAnalyzer)


class IVDT(GazeAnalyzer):

    def __init__(self, saccade_min_velocity: float,
                 saccade_min_duration: float,
                 saccade_max_duration: float, window_size: int,
                 dispersion_threshold: float):
        """
        Set up parameters for fixation/saccade/sp detection
        :param saccade_min_velocity: velocity threshold for saccade detection
        :param saccade_min_duration: if less, then it is more likely tobe microsaccade
        :param saccade_max_duration: if more, then end up saccade
        :param window_size: size of sliding window
        :param dispersion_threshold: threshold for sp and fixations separation
        """
        super().__init__()
        self._saccade_min_duration = saccade_min_duration
        self._saccade_min_velocity = saccade_min_velocity
        self._saccade_max_duration = saccade_max_duration
        self._window_size = window_size
        self._dispersion_threshold = dispersion_threshold


    def classify_eyemovements(self, gaze: np.ndarray,
                              timestamps: np.ndarray,
                              velocity: np.ndarray,
                              **kwargs) -> Tuple[np.ndarray, List[float]]:
        """
        Get list of eye movements as fixations or saccades.
        """
        # if [2, n_samples] -> change to [n_samples, 2]
        if gaze.shape[0] < gaze.shape[1]:
            gaze = gaze.reshape(gaze.shape[1], gaze.shape[0])
        n, m = gaze.shape

        movements = np.zeros((n,), dtype=np.int32)  # all eye movements detected
        stats = []

        # detect saccades
        detected_saccades = self.detect_saccades(timestamps, velocity)
        cleaned_saccades = self.clean_short_saccades(detected_saccades, timestamps)
        for i, m in enumerate(movements):
            if i in list(chain.from_iterable(cleaned_saccades)):
                movements[i] = GazeState.saccade
            else:
                movements[i] = GazeState.unknown

        start = 0  # current window start position
        end = 0  # current window end position
        fix_marked = False  # fixation found flag
        window_size = self._window_size  # instantiate local variable (it can be changed!)

        # fixations and sp identification
        while (end < len(gaze) - 1) and (start < len(gaze) - 1):

            # Calculation window
            if (start == 0) or (start == end):  # first point
                g = gaze[start: start + window_size]
                end = start + window_size

            elif fix_marked:  # last time found a fix, then take full new window
                if (start + window_size) >= len(gaze):  # tail
                    g = gaze[start:]
                    window_size = len(gaze) - start - 1
                    end = len(gaze)
                else:
                    g = gaze[start: start + window_size]
                    end = start + window_size
                # reset fix flg
                fix_marked = False

            else:  # otherwize add one element from left side od array
                if movements[start] != GazeState.saccade:
                    g = np.append(g, gaze[start].reshape(1, 2), axis=0)
                    end += 1

            # Re-calc dispersion
            dispersion = self.count_dispersion(g)
            stats.append(dispersion)
            # fixation
            if dispersion < self._dispersion_threshold:
                while (dispersion < self._dispersion_threshold) and (end + 1 < len(gaze)):
                    end += 1
                    g = np.append(g, gaze[end].reshape(1, 2), axis=0)
                    dispersion = self.count_dispersion(g)
                fix_marked = True
                # mark as fixation
                for i in range(start, end, 1):
                    if (i < len(movements)) and (movements[i] != GazeState.saccade):
                        movements[i] = GazeState.fixation
                start = end
            # sp
            else:
                # mark as sp
                if movements[start] != GazeState.saccade:
                    movements[start] = GazeState.sp
                start += 1

        return (movements, stats)



    def count_dispersion(self, gaze_points: np.ndarray):
        """
        Get gaze_points as 2D arraty of x and y coordinates [n_samples, 2]
        and return dispersion.
        """
        return (max(gaze_points[:, 0]) - min(gaze_points[:, 0])
                + max(gaze_points[:, 1]) - min(gaze_points[:, 1]))


    def clean_short_saccades(self, saccades_list: List[List[int]],
                             timestamps: np.ndarray):
        """
        If in gaze row is small saccade (< 12 ms.), then
        all poits of this saccade is discarded from future analysis.
        """
        return [move_idxs for move_idxs in saccades_list
                if (timestamps[move_idxs[-1]] - timestamps[move_idxs[0]]) > self._saccade_min_duration]


    def detect_saccades(self, timestamps: np.ndarray, velocity: np.ndarray):
        """
        Returns list of saccades.
        """
        all_saccades = []  # all detected saccades as movements
        sac_start = 0  # number of points in the saccade
        saccade = []  # single saccade indexes
        curr_state = None
        last_state = None

        for i, (ts, v) in enumerate(zip(timestamps, velocity)):

            # point mark as saccade
            if v > self._saccade_min_velocity:
                curr_state = "saccade"
                # new saccade
                if last_state != curr_state:
                    sac_start = i
                    saccade.append(sac_start)
                    last_state = curr_state
                else:
                    # duration of saccade
                    duration = ts - timestamps[sac_start]
                    if duration > self._saccade_max_duration:
                        # end up saccade
                        saccade.append(i)
                        all_saccades.append(saccade)
                        saccade = []
                        last_state = None
                    else:
                        saccade.append(i)
                        last_state = curr_state
            else:
                if last_state == "saccade":
                    # end up saccade
                    all_saccades.append(saccade)
                    saccade = []
                    last_state = None

        return all_saccades

    # --------------- ACCESSORS --------------------

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, ws: int):
        self._window_size = ws

    @property
    def dispersion_threshold(self):
        return self._dispersion_threshold

    @dispersion_threshold.setter
    def dispersion_threshold(self, ds: float):
        self._dispersion_threshold = ds