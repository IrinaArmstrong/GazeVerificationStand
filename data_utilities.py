
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from typing import (List, Union, Dict, Any)


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
    # Transforms to long forme DF
    ts_df = []
    for i, row in data.iterrows():
        df = pd.DataFrame(columns=['x', 'y', 'label', 'guid'])
        if type(data_col) == str:
            # Joined array of x and y
            df['x'] = row[data_col].reshape(-1, 2)[:, 0]
            df['y'] = row[data_col].reshape(-1, 2)[:, 1]
        else:
            # Separately x and y
            df['x'] = row[data_col[0]]
            df['y'] = row[data_col[1]]
        df['label'] = row[target_col]
        df['guid'] = row[guid_col]
        ts_df.append(df)

    data = pd.concat(ts_df).reset_index().rename({"index": "i"}, axis=1)
    data.label = data.label.astype(int)
    return data



def split_dataset(dataset: pd.DataFrame, label_col_name: str,
                  max_seq_len: int):
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
                pad_symbol: float):
    ret_data = []
    try:
        for _ in range(len(data)):
            data_pair = data.pop()
            if len(data_pair['data']) < max_seq_len:
                ret_data.append({'guid': data_pair['guid'],
                                 'data': np.pad(data_pair['data'],
                                                pad_width=(0, max_seq_len - len(data_pair['data'])),
                                                mode='constant', constant_values=0.0),
                                 'label': data_pair['label']})
            else:
                ret_data.append(data_pair)
    except:
        print("Data list ended.")
    del data
    return ret_data


def truncate_dataset(data: List[Dict[str, Any]], max_seq_len: int):
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






