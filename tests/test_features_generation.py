import sys
import unittest
sys.path.insert(0, '..')

from config import config, init_config
from data_utilities import (split_dataset, truncate_dataset, pad_dataset,
                            vertical_align_data, horizontal_align_data)
from generate_features import FeatureGenerator


class TestFeatureGeneration(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        init_config("../set_locations.ini")

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
