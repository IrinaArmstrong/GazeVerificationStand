import sys
import unittest
import random
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from pprint import pprint

from helpers import read_json
from config import config, init_config
from datasets import TrainDataset
from eyemovements.filtering import sgolay_filter_dataset
from data_utilities import groupby_session, horizontal_align_data, interpolate_sessions
from eyemovements.classification import (get_sp_moves_dataset, IVDT, GazeState)
from eyemovements.eyemovements_metrics import estimate_quality
from eyemovements.classification import classify_eyemovements_wrapper
from visualization import visualize_eyemovements

import warnings
warnings.filterwarnings('ignore')


class TestEyemovementsModule(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        init_config("../set_locations.ini")
        self.train_dataset = TrainDataset(config.get("DataPaths", "run_data"), ).create_dataset()
        print(f"Shape of loaded data: {self.train_dataset.shape}")
        print(f"Unique users: {self.train_dataset['user_id'].nunique()}")
        print(f"Unique sessions: {self.train_dataset['session_id'].nunique()}")


    def test_groupby_session(self):
        sess = groupby_session(self.train_dataset)
        self.assertEqual(self.train_dataset.session_id.nunique(), len(sess))

    def test_sgolay_filter_dataset(self):
        sess = groupby_session(self.train_dataset)
        sess = sgolay_filter_dataset(sess, **dict(read_json(config.get("EyemovementClassification",
                                                                       "filtering_params"))))

        self.assertEqual(len(sess), np.sum(["velocity_sqrt" in s.columns for s in sess]))
        self.assertEqual(len(sess), np.sum(["stimulus_velocity" in s.columns for s in sess]))
        self.assertEqual(len(sess), np.sum(["acceleration_sqrt" in s.columns for s in sess]))

    def test_ivdt_quality(self):
        sess = groupby_session(self.train_dataset)
        sess = sgolay_filter_dataset(sess, **dict(read_json(config.get("EyemovementClassification",
                                                                       "filtering_params"))))

        model_params = dict(read_json(config.get('EyemovementClassification', 'model_params')))
        ivdt = IVDT(saccade_min_velocity=model_params.get('saccade_min_velocity'),
                    saccade_min_duration=model_params.get('min_saccade_duration_threshold'),
                    saccade_max_duration=model_params.get('max_saccade_duration_threshold'),
                    window_size=model_params.get('window_size'),
                    dispersion_threshold=model_params.get('dispersion_threshold'))

        sess_num = 0
        movements, stats = ivdt.classify_eyemovements(sess[sess_num][['filtered_X', 'filtered_Y']].values.reshape(-1, 2),
                                                      sess[sess_num]['timestamps'].values,
                                                      sess[sess_num]['velocity_sqrt'].values)

        sess[sess_num]["movements"] = movements
        sess[sess_num]["movements_type"] = [GazeState.decode(x) for x in sess[sess_num]["movements"]]
        print("Statistics of dispersion:\n", pd.Series(stats).describe())
        visualize_eyemovements(sess[sess_num], fn="eyemovements",
                               y_col="gaze_Y", x_col='gaze_X', time_col="timestamps", color="movements_type")
        metrics = estimate_quality([sess[sess_num]])
        print("Eye movements classification metrics:")
        pprint(metrics)

    def test_full_classification(self):
        data = classify_eyemovements_wrapper(self.train_dataset)
        self.assertEqual(self.train_dataset['session_id'].nunique(), len(data))


    def test_get_sp_moves_dataset(self):
        testing_df = [pd.DataFrame({'movements': 10*[1] + 4*[3] + 6*[2] + 10*[3]})]
        sp_data = get_sp_moves_dataset(testing_df)
        self.assertEqual(2, sp_data['move_id'].nunique())
        self.assertEqual(4, sp_data.loc[sp_data['move_id'] == 0].shape[0])
        self.assertEqual(10, sp_data.loc[sp_data['move_id'] == 1].shape[0])





if __name__ == '__main__':
    unittest.main()
