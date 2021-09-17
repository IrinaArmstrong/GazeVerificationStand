import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from helpers import read_json
from config import config, init_config
from verification.train_dataloaders import create_training_dataloaders

import logging_handler
logger = logging_handler.get_logger(__name__)

class TestTraining(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_base_path = Path(__file__).parent.parent.resolve()
        init_config(str(self._current_base_path / "set_locations.ini"))

    def test_create_dataloaders(self):

        splitting_params = dict(read_json(config.get('Preprocessing', 'processing_params'))).get("splitting_params", {})
        logger.debug(f"Splitting parameters: {splitting_params}")

        batching_params = dict(read_json(config.get('GazeVerification', 'model_params'))).get("batching_options", {})
        logger.debug(f"Batching parameters: {batching_params}")

        n_rows = 600
        data_col = splitting_params.get("data_col")  # 'data_scaled'
        target_col = splitting_params.get("target_col")  # 'user_id'
        session_col = splitting_params.get("session_id_col")  # 'unique_session_id'

        data = pd.DataFrame({
            data_col: [list(np.random.uniform(0, 1, size=(120,))) for _ in range(n_rows)],
            target_col: [sss for ss in [[s]*60 for s in np.arange(0, n_rows // 60)] for sss in ss],
            session_col: [sss for ss in [[s] * 10 for s in np.arange(0, n_rows // 10)] for sss in ss]
        })
        dataloaders = create_training_dataloaders(data,
                                                  splitting_params_fn=config.get('Preprocessing',
                                                                                 'processing_params'),
                                                  batching_params_fn=config.get('GazeVerification', 'model_params'))
        self.assertEqual(2, len(dataloaders))
        self.assertIn("train", list(dataloaders.keys()))
        self.assertIn("validation", list(dataloaders.keys()))

    # def test_save_load_model(self):
    #     parameters = dict(read_json(config.get("GazeVerification", "model_params")))
    #     model = EmbeddingNet(**parameters.get("model_params"))
    #     # Saving final version
    #     save_model(model,
    #                dir=parameters.get("training_options", {}).get("checkpoints_dir", "checkpoints_dir"),
    #                filename=parameters.get("training_options",
    #                                        {}).get("model_name", "model") + "_test.pt")
    #     fname = os.path.join(sys.path[0],
    #                          parameters.get("training_options", {}).get("checkpoints_dir", "checkpoints_dir"),
    #                          parameters.get("training_options",
    #                                         {}).get("model_name", "model") + "_test.pt")
    #     self.assertTrue(os.path.isfile(fname))
    #     loaded_model = init_model(EmbeddingNet, parameters=parameters,
    #                               dir=parameters.get("training_options", {}).get("checkpoints_dir", "checkpoints_dir"),
    #                               filename=parameters.get("training_options",
    #                                                       {}).get("model_name", "model") + "_test.pt")
    #     logger.info(loaded_model)
    #
    # def test_clear_logs_dir(self):
    #     parameters = dict(read_json(config.get("GazeVerification", "model_params")))
    #     open(os.path.join(sys.path[0], parameters.get("training_options",
    #                                                   {}).get("tensorboard_log_dir", "tblogs"), "newfile.txt"), 'a').close()
    #     clear_logs_dir(parameters.get("training_options",
    #                                   {}).get("tensorboard_log_dir", "tblogs"), ignore_errors=False)
    #     self.assertTrue(os.path.exists(os.path.join(sys.path[0], parameters.get("training_options",
    #                                                   {}).get("tensorboard_log_dir", "tblogs"))))
    #     existing_files = len(os.listdir(os.path.join(sys.path[0], parameters.get("training_options",
    #                                                                              {}).get("tensorboard_log_dir", "tblogs"))))
    #     self.assertEqual(0, existing_files)


if __name__ == '__main__':
    unittest.main()
