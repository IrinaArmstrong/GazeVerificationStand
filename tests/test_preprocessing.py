import os
import sys
import random
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, '..')

from create_training_dataset import TrainDataset
from config import config, init_config
from data_utilities import (split_dataset, truncate_dataset, pad_dataset,
                            vertical_align_data, horizontal_align_data,
                            interpolate_sessions, normalize_gaze)


class TestFeatureGeneration(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        init_config("../set_locations.ini")
        self.train_dataset = TrainDataset(config.get("DataPaths", "run_data"), ).create_dataset().reset_index().rename({
            "index": "sp_id"}, axis=1)
        self.train_dataset["ts_id"] = self.train_dataset.apply(lambda row:
                                                               (str(row['session_id']) +
                                                                "_" + str(row['sp_id']) +
                                                                "_" + str(row['stimulus_type'])), axis=1)


    def test_interpolation(self):
        initial_num_sess = self.train_dataset.session_id.nunique()
        beat_sess_id = random.choice(self.train_dataset.session_id.unique())
        sess_len = self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id].shape[0]
        if sess_len > 500:
            self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id, "gaze_X"] = -100
            self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id, "gaze_Y"] = -100

        self.train_dataset = interpolate_sessions(self.train_dataset, "gaze_X", "gaze_Y")
        self.assertEqual(initial_num_sess - 1, self.train_dataset.session_id.nunique())


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


    def test_vertical_align_data(self):
        data = pd.DataFrame({'user_id': 10 * [0] + 10 * [1],
                                   'session_id': 10 * [0] + 10 * [1],
                                   'gaze_X': np.random.random_sample(20),
                                   'gaze_Y': np.random.random_sample(20),
                                   'ts_id': np.arange(0, 20)})
        vdata = vertical_align_data(data, data_col=["gaze_X", "gaze_Y"],
                                    target_col='user_id', guid_col='ts_id')
        self.assertEqual(len(data), len(vdata))
        self.assertListEqual(['i', 'x', 'y', 'label', 'guid'], list(vdata.columns))


    def test_split_dataset(self):
        data = pd.DataFrame.from_dict({str(row): np.random.uniform(0, 1, size=50) + 1 for row in range(10)},
                                      orient="index")
        data['label_col'] = "user_0"
        # Case #1
        splitted_data = split_dataset(data, label_col_name='label_col', max_seq_len=25)
        self.assertEqual(len(data) * 2, len(splitted_data))
        # Case #2
        splitted_data = split_dataset(data, label_col_name='label_col', max_seq_len=35)
        self.assertEqual(len(data), len(splitted_data))
        # Case #3
        splitted_data = split_dataset(data, label_col_name='label_col', max_seq_len=85)
        self.assertEqual(0, len(splitted_data))


    def test_pad_dataset(self):
        data = [{'guid': row,
                 "data": np.random.uniform(0, 1, size=row),
                 'label': 0} for row in range(10)]
        padded_data = pad_dataset(data, max_seq_len=10, pad_symbol=0.0)
        self.assertEqual([10] * 10, [len(d.get('data')) for d in padded_data])

    def test_truncate_dataset(self):
        data = [{'guid': row,
                 "data": np.random.uniform(0, 1, size=row+11),
                 'label': 0} for row in range(10)]
        trunc_data = truncate_dataset(data, max_seq_len=10)
        self.assertEqual([10] * 10, [len(d.get('data')) for d in trunc_data])

    def test_normalize_dataset(self):
        data = pd.DataFrame({'data': [list(np.random.random_integers(1, 20, size=10)) for _ in range(50)]})
        # Case #1
        norm_data = normalize_gaze(data, to_restore=False, to_save=True,
                                   checkpoint_dir=config.get('GazeVerification', 'pretrained_model_location'))
        self.assertListEqual(["data", "data_scaled"], list(norm_data.columns))
        self.assertTrue(os.path.isfile(os.path.join(config.get('GazeVerification', 'pretrained_model_location'),
                                                    "scaler.pkl")))
        # Case #2
        norm_data = normalize_gaze(data, to_restore=True, to_save=False,
                                   checkpoint_dir=config.get('GazeVerification', 'pretrained_model_location'))
        self.assertListEqual(["data", "data_scaled"], list(norm_data.columns))
        self.assertTrue(os.path.isfile(os.path.join(config.get('GazeVerification', 'pretrained_model_location'),
                                                    "scaler.pkl")))
        try:
            os.remove(os.path.join(config.get('GazeVerification', 'pretrained_model_location'), "scaler.pkl"))
        except OSError:
            pass
        self.assertFalse(os.path.isfile(os.path.join(config.get('GazeVerification', 'pretrained_model_location'),
                                                    "scaler.pkl")))


if __name__ == '__main__':
    unittest.main()
