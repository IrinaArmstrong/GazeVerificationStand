import sys
import unittest
sys.path.insert(0, '..')


from config import config, init_config
from create_training_dataset import TrainDataset, RunDataset


class TestDatasets(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        init_config("../set_locations.ini")


    def test_run_dataset_creation_estim(self):
        ds = RunDataset(owner_path=config.get('DataPaths', 'owner_data'),
                        others_path=config.get('DataPaths', 'run_data'),
                        estimate_quality=True)
        for user in ds._others:
            print(user)
        print(ds._owner)
        owner_data = ds.get_owner_data()
        others_data = ds.get_others_data()
        print(f"Owner data: {owner_data.shape}")
        print(f"Others data: {others_data.shape}")
        self.assertEqual(True, True)


    def test_run_dataset_creation_unknown(self):
        ds = RunDataset(owner_path=config.get('DataPaths', 'owner_data'),
                        others_path=config.get('DataPaths', 'run_data'),
                        estimate_quality=False)
        print(ds._others)
        print(ds._owner)
        self.assertEqual(True, True)

    def test_train_dataset_creation_estim(self):
        ds = TrainDataset(ds_path=config.get('DataPaths', 'train_data'))
        for user in ds._users:
            print(user)
        gaze_data = ds.create_dataset()
        print(f"Dataset shape: {gaze_data.shape}")
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
