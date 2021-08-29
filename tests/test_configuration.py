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


if __name__ == '__main__':
    unittest.main()
