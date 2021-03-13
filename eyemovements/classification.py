# Basic
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from typing import (List)


from config import config
from helpers import read_json
from eyemovements.eyemovements_utils import get_movement_indexes, GazeState
from eyemovements.filtering import sgolay_filter_dataset
from eyemovements.eyemovements_classifier import IVDT, classify_eyemovements_dataset
from eyemovements.eyemovements_metrics import estimate_quality
from data_utilities import horizontal_align_data, groupby_session

import warnings
warnings.filterwarnings('ignore')


def get_sp_moves_dataset(data: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Select from given data only SP moves and forms dataset.
    :param data: classified data
    :return: dataframe with only SP moves.
    """
    sps = []
    for df in tqdm(data):
        moves_sp = get_movement_indexes(df['movements'], GazeState.sp)
        if len(moves_sp) > 0:
            for sp_ids in moves_sp:
                sps.append(df.iloc[sp_ids])
    for i, sp_df in enumerate(sps):
        sp_df.loc[:, 'move_id'] = i

    print(f"In classified sessions there are {len(sps)} with total length: {np.sum([len(s) for s in sps])} SP.")
    sps = pd.concat(sps, ignore_index=True)
    return sps


def classify_eyemovements_wrapper(data: pd.DataFrame) -> pd.DataFrame:
    """
    Make eye movements classification.
    :param data: dataframe with gaze data
    :return: dataframe with added column with classified movements.
    """
    data = groupby_session(data)
    data = sgolay_filter_dataset(data, **dict(read_json(config.get("EyemovementClassification",
                                                 "filtering_params"))))
    model_params = dict(read_json(config.get('EyemovementClassification', 'model_params')))

    ivdt = IVDT(saccade_min_velocity=model_params.get('saccade_min_velocity'),
                saccade_min_duration=model_params.get('min_saccade_duration_threshold'),
                saccade_max_duration=model_params.get('max_saccade_duration_threshold'),
                window_size=model_params.get('window_size'),
                dispersion_threshold=model_params.get('dispersion_threshold'))

    # Filtering
    thresholds_dict = {'min_saccade_duration_threshold': model_params.get('min_saccade_duration_threshold'),
                       'max_saccade_duration_threshold': model_params.get('max_saccade_duration_threshold'),
                       'min_fixation_duration_threshold': model_params.get('min_fixation_duration_threshold'),
                       'min_sp_duration_threshold': model_params.get('min_sp_duration_threshold')}

    return classify_eyemovements_dataset(ivdt, data, gaze_col=['filtered_X', 'filtered_Y'],
                                         time_col='timestamps', velocity_col='velocity_sqrt',
                                         duration_thresholds=thresholds_dict)


def run_eyemovements_classification(data: pd.DataFrame, is_train: bool,
                                    do_estimate_quality: bool) -> pd.DataFrame:
    """
    Make eye movements classification in training or running mode.
    :param data: dataframe with gaze data
    :param is_train: whether is training mode (user_id is known) ot running (unknown)
    :param do_estimate_quality: whether to print evaluation metrics of classification
    :param filtering_kwargs: parameters for Savitsky-Golay filter
    :return: dataframe with SP moves only.
    """
    data = classify_eyemovements_wrapper(data)

    if do_estimate_quality:
        metrics = estimate_quality(data)
        print("Eye movements classification metrics:")
        pprint(metrics)

    sp_data = get_sp_moves_dataset(data)
    if is_train:
        # with known user_id
        sp_data = horizontal_align_data(sp_data,
                                        grouping_cols=['user_id', 'session_id', 'stimulus_type', 'move_id'],
                                        aligning_cols=['x_diff', 'y_diff']).reset_index().rename({"index": "sp_id"},
                                                                                                 axis=1)
    else:
        # no user_id
        sp_data = horizontal_align_data(sp_data,
                                        grouping_cols=['session_id', 'stimulus_type', 'move_id'],
                                        aligning_cols=['x_diff', 'y_diff']).reset_index().rename({"index": "sp_id"},
                                                                                                 axis=1)
    return sp_data


