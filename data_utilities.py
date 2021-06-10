import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from typing import (List, Union, Dict, Any)
from sklearn.preprocessing import StandardScaler

from helpers import read_json
#---------------------------- PREPROCESSING UTILITIES ----------------------------

def groupby_session(data: pd.DataFrame,
                    filter_threshold: int=50) -> List[pd.DataFrame]:
    """
    Group data by sessions.
    :param data: single DataFrame with all recorded sessions
    :param filter_threshold: minimum length of session recording to select
    :return: list of sessions (as DataFrames).
    """
    sess_data = []
    for group_name, group in data.groupby(by=['user_id', 'session_id']):
        group['user_id'] = group_name[0]
        group['session_id'] = group_name[1]
        if group.shape[0] > filter_threshold:
            sess_data.append(group)
    print(f"Resulted list of data length: {len(sess_data)}")
    del data
    return sess_data


def horizontal_align_data(df: pd.DataFrame,
                          grouping_cols: Union[str, List[str]],
                          aligning_cols: List[str]) -> pd.DataFrame:
    if len(aligning_cols) != 2:
        print("Should be given two separate columns of coordinates.")
        return df

    hdf = []
    for group_name, group_df in tqdm(df.groupby(by=grouping_cols)):
        group_df = pd.DataFrame(group_df[aligning_cols].T.values.flatten(order='F').reshape(1, -1),
                                columns=list(chain.from_iterable([[col_name + str(col_n) for col_name in ["x_", "y_"]]
                                                                  for col_n in range(group_df.shape[0])])))
        for i, col_name in enumerate(grouping_cols):
            group_df[col_name] = group_name[i]
        hdf.append(group_df)
    hdf = pd.concat(hdf, axis=0)
    return hdf


def vertical_align_data(data: pd.DataFrame,
                        data_col: Union[str, List[str]],
                        target_col: str,
                        guid_col: str) -> pd.DataFrame:
    """
    Transforms to long formed DataFrame: as one row == one gaze coordinate.
    :param data: initial dataset;
    :param data_col: x and y gaze coordinates columns names;
    :param target_col: target (user id);
    :param guid_col: unique identifier for session;
    :return: long DataFrame.
    """
    #
    df = pd.DataFrame(columns=['x', 'y', 'label', 'guid'])
    for i, row in data.iterrows():
        vrow = {}
        if type(data_col) == str:
            # Joined array of x and y
            vrow['x'] = row[data_col].reshape(-1, 2)[:, 0]
            vrow['y'] = row[data_col].reshape(-1, 2)[:, 1]
        else:
            # Separately x and y
            vrow['x'] = row[data_col[0]]
            vrow['y'] = row[data_col[1]]
        vrow['label'] = row[target_col]
        vrow['guid'] = row[guid_col]
        df = df.append(vrow, ignore_index=True)

    data = df.reset_index().rename({"index": "i"}, axis=1)
    del df
    data.label = data.label.astype(int)
    return data


def split_dataset(dataset: pd.DataFrame, label_col_name: str,
                  max_seq_len: int) -> List[Dict[str, Any]]:
    data = []
    guid_cnt = 0
    for i, (label, xy) in tqdm(enumerate(zip(dataset[label_col_name].values,
                                             dataset.filter(regex=("[\d]+")).values))):
        xy = xy[~np.isnan(xy)]  # not nan values

        if len(xy) >= max_seq_len:
            for i in range(len(xy) // (max_seq_len)):
                if len(xy[i * max_seq_len: (i + 1) * max_seq_len]) > 0.85 * max_seq_len:
                    guid_cnt += 1
                    data.append({"guid": guid_cnt,
                                 "data": xy[i * max_seq_len: (i + 1) * max_seq_len],
                                 "label": label})
        elif len(xy) > 0.85 * max_seq_len:
            guid_cnt += 1
            data.append({'guid': guid_cnt,
                         'data': xy,
                         'label': label})

    return data


def pad_dataset(data: List[Dict[str, Any]], max_seq_len: int,
                pad_symbol: float) -> List[Dict[str, Any]]:
    ret_data = []
    try:
        for _ in range(len(data)):
            data_pair = data.pop()
            if len(data_pair['data']) < max_seq_len:
                ret_data.append({'guid': data_pair['guid'],
                                 'data': np.pad(data_pair['data'],
                                                pad_width=(0, max_seq_len - len(data_pair['data'])),
                                                mode='constant', constant_values=pad_symbol),
                                 'label': data_pair['label']})
            else:
                ret_data.append(data_pair)
    except:
        print("Data list ended.")
    del data
    return ret_data


def truncate_dataset(data: List[Dict[str, Any]], max_seq_len: int) -> List[Dict[str, Any]]:
    ret_data = []
    try:
        for _ in range(len(data)):
            data_pair = data.pop()
            if len(data_pair['data']) > max_seq_len:
                ret_data.append({'guid': data_pair['guid'],
                                 'data': data_pair['data'][:max_seq_len],
                                 'label': data_pair['label']})
            else:
                ret_data.append(data_pair)
    except:
        print("Data list ended.")
    del data
    return ret_data


def interpolate_sessions(sessions: pd.DataFrame, x: str, y: str,
                         beaten_ratio: float=30, min_length: int=500,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Clear missing data to make correct filtration.
    (Disable its effect on filtration).
    """
    sessions['bad_sample'] = sessions.apply(lambda row: 1 if (row[x] < 0 and row[y] < 0) else 0, axis=1)

    # Make non valid frame x any as Nans
    sessions[x] = sessions[x]
    sessions[y] = sessions[y]
    sessions.loc[sessions.bad_sample == 1, x] = np.nan
    sessions.loc[sessions.bad_sample == 1, y] = np.nan

    # Inside one session - fill with interpolate values with Pandas splines
    sess_df_filled = []
    beaten_cnt = 0
    for group_name, group in tqdm(sessions.groupby(by=['user_id', 'session_id'])):
        if 100 * (group.bad_sample.sum() / group.shape[0]) >= beaten_ratio:
            print(f"Broken session with ratio of beaten rows > {beaten_ratio}%")
            beaten_cnt += 1
            continue
        if group.shape[0] < min_length:
            print(f"Too small session with length < {min_length}: {group.shape[0]}")
            beaten_cnt += 1
            continue

        print(f"From data shape: {group.shape[0]} there are {group[x].isna().sum()} NaNs")
        group[x] = group[x].interpolate(method='polynomial', order=3)
        group[y] = group[y].interpolate(method='polynomial', order=3)

        if (sum(group[x].isna()) > 0) or (sum(group[y].isna()) > 0):
            group = group.loc[~group[x].isna()]
            group = group.loc[~group[y].isna()]
        sess_df_filled.append(group.reset_index(drop=True))

    sessions = pd.concat(sess_df_filled, axis=0)
    if verbose:
        print(f"Cleaned sessions as beaten: {beaten_cnt}")
        print(f"Sessions after interpolation shape: {sessions.shape}")
    del sess_df_filled
    return sessions


def preprocess_data(data: pd.DataFrame, is_train: bool,
                    params_path: str) -> pd.DataFrame:
    """
    Split, pad and truncate time series data.
    :param data: dataframe with samples
    :param is_train: mode
    :param max_seq_length: selected maximum length of sample
    :param padding_symbol: symbol for padding (default is 0.0)
    :return: dataframe with processed samples.
    """
    meta_columns = ['user_id', 'session_id', 'stimulus_type', 'move_id', 'sp_id']
    params = dict(read_json(params_path))

    # If stimulus name contains '_' change it to '-'
    data.stimulus_type = data.stimulus_type.str.replace('_', '-', regex=True)
    data['length'] = data.apply(lambda row: sum(row.notnull()) - len(meta_columns), axis=1)

    # fixme: There is a bug in columns mapping
    data.sp_id = data.move_id
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
    data = split_dataset(data, label_col_name='ts_id', max_seq_len=params.get('max_seq_length', 100))
    data = pad_dataset(data, max_seq_len=params.get('max_seq_length', 100), pad_symbol=params.get('padding_symbol', 0))
    data = pd.DataFrame(list(truncate_dataset(data, max_seq_len=params.get('max_seq_length', 100))))
    if is_train:
        data['user_id'] = data.label.apply(lambda x: x.split("_")[0]).astype(int)
    else:
        data['user_id'] = 0

    # data['guid'] = data.label + "_" + data.guid.map(str)
    data["user_session_id"] = data.label.apply(lambda x: x.split("_")[1]).astype(int)
    data["sp_id"] = data.label.apply(lambda x: x.split("_")[2]).astype(int)
    data["stimulus_type"] = data.label.apply(lambda x: x.split("_")[3])
    data["unique_session_id"] = data.groupby(["user_id", "user_session_id"]).ngroup()
    data.rename({"guid": "splitted_sp_id"}, axis=1, inplace=True)
    return data


def normalize_gaze(data: pd.DataFrame, to_restore: bool=False,
                   to_save: bool=True, checkpoint_dir: str="models_checkpoints") -> pd.DataFrame:
    """
    Re-scale gaze data to zero mean and singular std.
    """
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)
                   if "scaler" in name]
    if to_restore and checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Restoring normalizer from: {latest_checkpoint}")
        scaler = joblib.load(latest_checkpoint)
        data["data_scaled"] = [list(vec) for vec in scaler.transform(data['data'].to_list())]
    else:
        scaler = StandardScaler()
        data["data_scaled"] = [list(vec) for vec in scaler.fit_transform(data['data'].to_list())]
    if to_save:
        _ = joblib.dump(scaler, os.path.join("scaler.pkl"), compress=9)

    return data



