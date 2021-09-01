import unittest
import random

import numpy as np
import pandas as pd
from pprint import pprint

from helpers import read_json
from config import config, init_config
from eyemovements.eyemovements_classifier import EyemovementsClassifier
from create_training_dataset import TrainDataset
from eyemovements.filtering import sgolay_filter_dataset
from eyemovements.eyemovements_utils import get_sp_moves_dataset
from data_utilities import groupby_session, horizontal_align_data, interpolate_sessions
from eyemovements.eyemovements_metrics_old import estimate_quality
from visualizations.visualization import visualize_eyemovements

import warnings
warnings.filterwarnings('ignore')

import logging_handler
logger = logging_handler.get_logger(__name__)


class TestEyemovementsModule(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        logger.info(f"Testing Eye movements Module started")
        super().__init__(method_name)
        init_config("../set_locations.ini")

        self.train_dataset = TrainDataset(config.get("DataPaths", "run_data"), ).create_dataset()
        logger.info(f"Shape of loaded data: {self.train_dataset.shape}")
        logger.info(f"Unique users: {self.train_dataset['user_id'].nunique()}")
        logger.info(f"Unique sessions: {self.train_dataset['session_id'].nunique()}")

    def test_interpolation(self):
        """
        Test cleaning beaten coordinates and interpolating.
        """
        initial_num_sess = self.train_dataset.session_id.nunique()
        beat_sess_id = random.choice(self.train_dataset.session_id.unique())
        sess_len = self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id].shape[0]
        if sess_len > 500:
            self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id, "gaze_X"] = -100
            self.train_dataset.loc[self.train_dataset.session_id == beat_sess_id, "gaze_Y"] = -100

        self.train_dataset = interpolate_sessions(self.train_dataset, "gaze_X", "gaze_Y")
        self.assertEqual(initial_num_sess - 1, self.train_dataset.session_id.nunique())

    def test_groupby_session(self):
        """
        Test grouping long-formed dataset by sessions ids.
        """
        sess = groupby_session(self.train_dataset)
        self.assertEqual(self.train_dataset.session_id.nunique(), len(sess))

    def test_sgolay_filter_dataset(self):
        """
        Test getting derivatives with Savitsky-Golay filter.
        (Here we do not check correctness of filtration and derivatives calculation)
        """
        sess = groupby_session(self.train_dataset)
        sess = sgolay_filter_dataset(sess, **dict(read_json(config.get("EyemovementClassification",
                                                                       "filtering_params"))))

        self.assertEqual(len(sess), np.sum(["velocity_sqrt" in s.columns for s in sess]))
        self.assertEqual(len(sess), np.sum(["stimulus_velocity" in s.columns for s in sess]))
        self.assertEqual(len(sess), np.sum(["acceleration_sqrt" in s.columns for s in sess]))

    def test_get_sp_moves_dataset(self):
        testing_df = [pd.DataFrame({'movements': 10 * [1] + 4 * [3] + 6 * [2] + 10 * [3]})]
        sp_data = get_sp_moves_dataset(testing_df)
        self.assertEqual(2, sp_data['move_id'].nunique())
        self.assertEqual(4, sp_data.loc[sp_data['move_id'] == 0].shape[0])
        self.assertEqual(10, sp_data.loc[sp_data['move_id'] == 1].shape[0])

    def test_horizontal_align_data(self):
        testing_df = pd.DataFrame({'movements': 20 * [3],
                                   'session_id': 10 * [0] + 10 * [1],
                                   'x': np.random.random_sample(20),
                                   'y': np.random.random_sample(20),
                                   'move_id': 5 * [0] + 5 * [1] + 5 * [2] + 5 * [3]})
        testing_hdf = horizontal_align_data(testing_df,
                                            grouping_cols=['session_id', 'move_id'],
                                            aligning_cols=['x', 'y'])
        self.assertEqual(4, testing_hdf.shape[0])
        self.assertEqual(12, testing_hdf.shape[1])

    def test_classifier_constrains(self):
        """
        Test correctness of eye movements classifier exceptions raising.
        """
        self.assertRaises(NotImplementedError, EyemovementsClassifier, {'mode': 'any', "algorithm": 'ivdt'})
        self.assertRaises(NotImplementedError, EyemovementsClassifier, {"mode": 'run', "algorithm": 'ivvt'})


    def test_full_pipeline_no_options(self):
        """
        Test full pipeline of eye movements classification running without exceptions.
        Here we do not test estimation & visualizations.
        """
        cls = EyemovementsClassifier(mode='calibrate', algorithm='ivdt')
        cls.classify_eyemovements(self.train_dataset, sp_only=True, visualize=False, estimate=False)

    def test_full_pipeline_visualize(self):
        """
        Test full pipeline of eye movements classification running without exceptions.
        Here we test visualization part.
        """
        cls = EyemovementsClassifier(mode='calibrate', algorithm='ivdt')
        cls.classify_eyemovements(self.train_dataset, sp_only=True, visualize=True, estimate=False)


    def test_full_pipeline_estimate(self):
        """
        Test full pipeline of eye movements classification running without exceptions.
        Here we test estimation part.
        """
        cls = EyemovementsClassifier(mode='calibrate', algorithm='ivdt')
        cls.classify_eyemovements(self.train_dataset, sp_only=True, visualize=False, estimate=True)




if __name__ == '__main__':
    unittest.main()
