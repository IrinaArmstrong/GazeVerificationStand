import unittest
from pathlib import Path
from config import config, init_config

class TestStandConfiguration(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        init_config("../set_locations.ini")

    def test_basic_paths(self):
        self.assertTrue(Path(config.get("Basic", "root_dir")).exists())
        self.assertTrue(Path(config.get("Basic", "settings_dir")).exists())
        self.assertTrue(Path(config.get("Basic", "output_dir")).exists())

    def test_data_paths(self):
        self.assertTrue(Path(config.get("DataPaths", "train_data")).exists())
        self.assertTrue(Path(config.get("DataPaths", "train_data")).is_dir())

        self.assertTrue(Path(config.get("DataPaths", "owner_data")).exists())
        self.assertTrue(Path(config.get("DataPaths", "owner_data")).is_dir())

        self.assertTrue(Path(config.get("DataPaths", "run_data")).exists())
        self.assertTrue(Path(config.get("DataPaths", "run_data")).is_dir())

        self.assertTrue(Path(config.get("DataPaths", "selected_columns")).exists())

    def test_eyemovements_clf_paths(self):
        self.assertTrue(Path(config.get("EyemovementClassification", "filtering_params")).exists())
        self.assertTrue(Path(config.get("EyemovementClassification", "model_params")).exists())

    def test_preprocessing_paths(self):
        self.assertTrue(Path(config.get("Preprocessing", "processing_params")).exists())

    def test_verification_paths(self):
        self.assertTrue(Path(config.get("GazeVerification", "verification_params")).exists())
        self.assertTrue(Path(config.get("GazeVerification", "model_params")).exists())


if __name__ == '__main__':
    unittest.main()
