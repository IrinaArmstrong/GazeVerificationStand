import os
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, '..')

from helpers import read_json
from config import config, init_config
from verification.train_dataloaders import create_training_dataloaders
from verification.model import EmbeddingNet
from verification.train_utils import save_model, init_model, clear_logs_dir

import logging_handler
logger = logging_handler.get_logger(__name__)

class TestTraining(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        init_config("../set_locations.ini")


    def test_create_dataloaders(self):
        n_rows = 600

        data = pd.DataFrame({
            'data_scaled': [list(np.random.uniform(0, 1, size=(120,))) for _ in range(n_rows)],
            'user_id': [sss for ss in [[s]*60 for s in np.arange(0, n_rows // 60)] for sss in ss],
            'unique_session_id': [sss for ss in [[s] * 10 for s in np.arange(0, n_rows // 10)] for sss in ss]
        })
        dataloaders = create_training_dataloaders(data,
                                                  splitting_params_fn=config.get('Preprocessing',
                                                                                 'processing_params'),
                                                  batching_params_fn=config.get('GazeVerification', 'model_params'))
        self.assertEqual(2, len(dataloaders))
        self.assertIn("train", list(dataloaders.keys()))
        self.assertIn("val", list(dataloaders.keys()))

    def test_save_load_model(self):
        parameters = dict(read_json(config.get("GazeVerification", "model_params")))
        model = EmbeddingNet(**parameters.get("model_params"))
        # Saving final version
        save_model(model,
                   dir=parameters.get("training_options", {}).get("checkpoints_dir", "checkpoints_dir"),
                   filename=parameters.get("training_options",
                                           {}).get("model_name", "model") + "_test.pt")
        fname = os.path.join(sys.path[0],
                             parameters.get("training_options", {}).get("checkpoints_dir", "checkpoints_dir"),
                             parameters.get("training_options",
                                            {}).get("model_name", "model") + "_test.pt")
        self.assertTrue(os.path.isfile(fname))
        loaded_model = init_model(EmbeddingNet, parameters=parameters,
                                  dir=parameters.get("training_options", {}).get("checkpoints_dir", "checkpoints_dir"),
                                  filename=parameters.get("training_options",
                                                          {}).get("model_name", "model") + "_test.pt")
        logger.info(loaded_model)

    def test_clear_logs_dir(self):
        parameters = dict(read_json(config.get("GazeVerification", "model_params")))
        open(os.path.join(sys.path[0], parameters.get("training_options",
                                                      {}).get("tensorboard_log_dir", "tblogs"), "newfile.txt"), 'a').close()
        clear_logs_dir(parameters.get("training_options",
                                      {}).get("tensorboard_log_dir", "tblogs"), ignore_errors=False)
        self.assertTrue(os.path.exists(os.path.join(sys.path[0], parameters.get("training_options",
                                                      {}).get("tensorboard_log_dir", "tblogs"))))
        existing_files = len(os.listdir(os.path.join(sys.path[0], parameters.get("training_options",
                                                                                 {}).get("tensorboard_log_dir", "tblogs"))))
        self.assertEqual(0, existing_files)


if __name__ == '__main__':
    unittest.main()
