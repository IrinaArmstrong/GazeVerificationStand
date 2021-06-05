import sys
import unittest
import random
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from pprint import pprint

from helpers import read_json
from config import config, init_config
from create_training_dataset import TrainDataset
from eyemovements.filtering import sgolay_filter_dataset
from data_utilities import groupby_session, horizontal_align_data, interpolate_sessions
from eyemovements.classification import (get_sp_moves_dataset, IVDT, GazeState)
from eyemovements.eyemovements_metrics import estimate_quality
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

    def test_interpolation(self):
        initial_num_sess = self.train_dataset.session_id.nunique()
        beat_sess_id = random.choice(self.train_dataset.session_id.unique())
        sess_len = self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id].shape[0]
        if sess_len > 500:
            self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id, "gaze_X"] = -100
            self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id, "gaze_Y"] = -100

        self.train_dataset = interpolate_sessions(self.train_dataset, "gaze_X", "gaze_Y")
        self.assertEqual(initial_num_sess - 1, self.train_dataset.session_id.nunique())

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
        metrics = estimate_quality(sess[sess_num])
        print("Eye movements classification metrics:")
        pprint(metrics)



    def test_get_sp_moves_dataset(self):
        testing_df = [pd.DataFrame({'movements': 10*[1] + 4*[3] + 6*[2] + 10*[3]})]
        sp_data = get_sp_moves_dataset(testing_df)
        self.assertEqual(2, sp_data['move_id'].nunique())
        self.assertEqual(4, sp_data.loc[sp_data['move_id'] == 0].shape[0])
        self.assertEqual(10, sp_data.loc[sp_data['move_id'] == 1].shape[0])


    def test_horizontal_align_data(self):
        testing_df = pd.DataFrame({'movements': 20*[3],
                                    'session_id': 10*[0] + 10*[1],
                                    'x': np.random.random_sample(20),
                                    'y': np.random.random_sample(20),
                                    'move_id': 5*[0] + 5*[1] + 5*[2] + 5*[3]})
        testing_hdf = horizontal_align_data(testing_df,
                                            grouping_cols=['session_id', 'move_id'],
                                            aligning_cols=['x', 'y'])
        self.assertEqual(4, testing_hdf.shape[0])
        self.assertEqual(12, testing_hdf.shape[1])


if __name__ == '__main__':
    unittest.main()
