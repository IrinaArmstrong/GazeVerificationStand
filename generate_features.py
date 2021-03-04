import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features
from sklearn.preprocessing import StandardScaler

from config import config
from helpers import read_json
from data_utilities import (split_dataset, pad_dataset, truncate_dataset,
                            vertical_align_data)

class FeatureGenerator:

    def __init__(self):
        self._feature_params = dict(read_json(config.get('FeatureGeneration', 'features_params')))
        self._feature_columns = []
        self._target_column = []


    def preprocess_data(self, data: pd.DataFrame, is_train: bool,
                        max_seq_length: int,
                        padding_symbol: float=0.0) -> pd.DataFrame:
        """
        Split, pad and truncate time series data.
        :param data: dataframe with samples
        :param is_train: mode
        :param max_seq_length: selected maximum length of sample
        :param padding_symbol: symbol for padding (default is 0.0)
        :return: dataframe with processed samples.
        """

        # If stimulus name contains '_' change it to '-'
        data.stimulus_type = data.stimulus_type.str.replace('_', '-', regex=True)
        if is_train:
            data["ts_id"] = data.apply(lambda row: (str(row['user_id']) +
                                                    "_" + str(row['session_id']) +
                                                    "_" + str(row['sp_id']) +
                                                    "_" + str(row['stimulus_type'])), axis=1)
        else:
            data["ts_id"] = data.apply(lambda row: (str(row['session_id']) +
                                                    "_" + str(row['sp_id']) +
                                                    "_" + str(row['stimulus_type'])), axis=1)
        # Split, pad and truncate
        data = split_dataset(data, label_col_name='ts_id', max_seq_len=max_seq_length)
        data = pad_dataset(data, max_seq_len=max_seq_length, pad_symbol=padding_symbol)
        data = pd.DataFrame(list(truncate_dataset(data, max_seq_len=max_seq_length)))
        if is_train:
            data['user_id'] = data.label.apply(lambda x: x[0])
        else:
            data['user_id'] = 0
        data['guid'] = data.label + "_" + data.guid.map(str)
        data = vertical_align_data(data, data_col='data',
                                   target_col='user_id', guid_col='guid')
        return data


    def extract_features(self, data: pd.DataFrame, is_train: bool,
                         rescale: bool) -> pd.DataFrame:
        """
        Extract selected features from data.
        :param data: dataframe with samples (converted horizontally)
        :param is_train: mode
        :param rescale: whether to normalize features to mean=0, std=1
        :param process_params: truncation and padding parameters
        :return: dataframe with features
        """
        data = self.preprocess_data(data, is_train=is_train, **dict(read_json(config.get("FeatureGeneration",
                                                                      "processing_params"))))
        data = extract_features(data[['guid', 'i', 'x', 'y']],
                                        column_id='guid', column_sort='i',
                                        default_fc_parameters=self._feature_params,
                                        impute_function=impute, n_jobs=1)

        # guid = user_id + session_id + sp_id + stimulus_type +  splitted SP id
        data = data.reset_index()
        split_i = 0
        if is_train:
            data["user_id"] = data["index"].apply(lambda x: x.split("_")[split_i]).astype(int)
            split_i += 1

        data["session_id"] = data["index"].apply(lambda x: x.split("_")[split_i]).astype(int)
        data["sp_id"] = data["index"].apply(lambda x: x.split("_")[split_i+1]).astype(int)
        data["stimulus_type"] = data["index"].apply(lambda x: x.split("_")[split_i+2])
        data["splitted_sp_id"] = data["index"].apply(lambda x: x.split("_")[split_i+3]).astype(int)

        self._feature_columns = [col for col in data.columns if
                            col not in ['sp_guid', 'user_id', 'stimulus_type',
                                           'session_id', 'sp_id', 'splitted_sp_id', 'index']]
        self._target_column = 'user_id'

        if rescale:
            scaler = StandardScaler()
            data[self._feature_columns] = scaler.fit_transform(data[self._feature_columns])

        return data



