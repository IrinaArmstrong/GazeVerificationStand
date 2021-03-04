import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import (List, Union, Tuple)

from config import init_config, config


class Session:

    def __init__(self, session_path: str, session_id: int, user_id: int):
        self._session_id = session_id
        self._session_path = session_path
        self._user_id = user_id

        self._dataset_fn = "_".join(self._session_path.split("_")[:-1]) + ".csv"
        self._stimulus_type = "_".join(self._session_path.split(".")[0].split("_")[-2:])


    def get_gaze_data(self) -> pd.DataFrame:
        """
        Get data from eyetracker for current session.
        :param add_video: to add data from video and stimulus.
        :return: dataframe with data.
        """
        gaze_df = pd.read_csv(self._dataset_fn, sep='\t')
        gaze_df.columns = [col.strip() for col in gaze_df.columns]
        gaze_df['timestamp'] = pd.to_datetime(gaze_df['timestamp'], unit='s')
        gaze_df['session_id'] = self._session_id
        gaze_df['stimulus_type'] = self._stimulus_type
        gaze_df['user_id'] = self._user_id
        gaze_df["x_diff"] = gaze_df["stim_X"] - gaze_df["gaze_X"]
        gaze_df["y_diff"] = gaze_df["stim_Y"] - gaze_df["gaze_Y"]
        return gaze_df



class User:

    def __init__(self, user_id: int, user_name: str,
                 sessions_fns: List[str]):
        self._user_id = user_id
        self._user_name = user_name
        self._sessions_fns = sessions_fns
        self._sessions = self.__create_sessions()

    def __create_sessions(self):
        return [Session(session_path=sess_fn, session_id=i, user_id=self._user_id)
                for i, sess_fn in enumerate(self._sessions_fns)]

    def get_num_sessions(self):
        return len(self._sessions)

    def get_session(self, sess_id: int) -> Session:
        if sess_id > len(self._sessions):
            print("Session index out of range!")
            return None
        return self._sessions[sess_id]

    def __str__(self):
        s = f"User #{self._user_id}: {self._user_name}\n"
        s += f"\nUser has {len(self._sessions)} sessions."
        s += f"\nWith {np.unique([sess._stimulus_type for sess in self._sessions])} stimulus types."
        return s

    def __eq__(self, other: object) -> object:
        return True if self._user_name == other._user_name else False

    def __hash__(self):
        return hash(self._user_name)

    def __add__(self, other: object) -> object:
        self._sessions_fns.extend(other._sessions_fns)
        self._sessions.extend(other._sessions)
        return self


class TrainDataset:

    def __init__(self, ds_path: str):
        self._path = ds_path
        self._selected_columns = pd.read_csv(config.get("DataPaths", "selected_columns"), header = 0,
                                             index_col = 0, squeeze = True).to_list()
        self._users, self._users_ids = self.__create_users()
        self._sessions = self.__create_sessions()


    def __get_sessions_folders(self, outer_path: str):
        return [path for path in glob.glob(os.path.join(outer_path, 'dataset'))]


    def __create_users(self) -> Tuple[List[User], List[int]]:
        meta_files = glob.glob(self._path + "\\*metadata.txt")
        meta_df = []
        for mfn in meta_files:
            mdf = pd.read_csv(mfn, delimiter=";", encoding="Windows-1251",
                              header=None, names=['name'])
            mdf['name'] = mdf['name'].str.replace('\t', ' ', regex=True)
            mdf['name'] = mdf['name'].str.strip()
            mdf['filename'] = mfn
            meta_df.append(mdf)
        meta_df = pd.concat(meta_df).groupby(by='name').agg({'filename': lambda x: list(x)}).reset_index()
        meta_df['user_id'] = np.arange(0, len(meta_df))
        users = [User(user_id=row['user_id'], user_name=row['name'], sessions_fns=row['filename'])
                 for i, row in meta_df[['user_id', 'name', 'filename']].iterrows()]
        return (users, meta_df['user_id'].to_list())

    def __create_sessions(self):
        return [user.get_session(sess_num) for user in self._users
                for sess_num in range(user.get_num_sessions())]

    def get_user(self, user_id: int) -> Union[User, None]:
        """
        Get User object by id.
        :type user_id: int, id of desires user
        """
        if user_id in self._users_ids:
            return self._users[self._users_ids.index(user_id)]
        else:
            print("User index out of range!")
            return None

    def create_dataset(self) -> pd.DataFrame:
        sess_df = pd.concat([sess.get_gaze_data()
                             for sess in tqdm(self._sessions, total=len(self._sessions))], axis=0)[self._selected_columns]
        return sess_df


class RunDataset:

    def __init__(self, owner_path: str, others_path: str,
                 estimate_quality: bool):
        self._owner_path = owner_path
        self._others_path = others_path
        self._estimate_quality = estimate_quality
        self._selected_columns = pd.read_csv(config.get("DataPaths", "selected_columns"), header = 0,
                                             index_col = 0, squeeze = True).to_list()
        self._owner = self.__create_owner()
        self._others = self.__create_others()
        self._others_sessions = self.__create_sessions()


    def __create_owner(self):
        meta_file = glob.glob(self._owner_path + "\\*metadata.txt")
        if len(meta_file) > 1:
            raise ValueError("Более чем один файл, идентифицирующий владельца!",
                             "\nВыберете один и попробуйте снова.")
        mdf = pd.read_csv(meta_file[0], delimiter=";", encoding="Windows-1251",
                          header=None, names=['name'])
        mdf['name'] = mdf['name'].str.replace('\t', ' ', regex=True)
        mdf['name'] = mdf['name'].str.strip()
        mdf['filename'] = meta_file[0]
        mdf['user_id'] = 1
        return User(user_id=1, user_name=mdf['name'], sessions_fns=mdf['filename'])


    def __create_others(self):
        # If _estimate_quality - then we know target values
        # so create users with their real ids
        if self._estimate_quality:
            meta_files = glob.glob(self._others_path + "\\*metadata.txt")
            meta_df = []
            for mfn in meta_files:
                mdf = pd.read_csv(mfn, delimiter=";", encoding="Windows-1251",
                                  header=None, names=['name'])
                mdf['name'] = mdf['name'].str.replace('\t', ' ', regex=True)
                mdf['name'] = mdf['name'].str.strip()
                mdf['filename'] = mfn
                meta_df.append(mdf)
            meta_df = pd.concat(meta_df).groupby(by='name').agg({'filename': lambda x: list(x)}).reset_index()
            meta_df['user_id'] = np.arange(0, len(meta_df))
            return [User(user_id=row['user_id'], user_name=row['name'], sessions_fns=row['filename'])
                     for i, row in meta_df[['user_id', 'name', 'filename']].iterrows()]
        # Otherwise all ids are 0 and user is "Unknown"
        else:
            data_files = glob.glob(self._others_path + "\\*.csv")
            return User(user_id=0, user_name="Unknown", sessions_fns=data_files)


    def __create_sessions(self):
        if self._estimate_quality:
            return [user.get_session(sess_num) for user in self._others
                    for sess_num in range(user.get_num_sessions())]
        else:
            self._others.get_session(0)


    def get_owner_data(self) -> pd.DataFrame:
        return self._owner.get_session(0).get_gaze_data()[self._selected_columns]

    def get_others_data(self) -> pd.DataFrame:
        sess_df = []
        for i, sess in tqdm(enumerate(self._others_sessions),
                         total=len(self._others_sessions)):
            sess_part_df = sess.get_gaze_data()
            sess_part_df['session_id'] = i
            sess_df.append(sess_part_df)
        sess_df = pd.concat(sess_df, axis=0)[self._selected_columns]
        return sess_df





if __name__ == "__main__":
    dataset_path = "D:\\Data\\EyesSimulation Sessions\\Export3"
    dataset = TrainDataset(dataset_path)
    for user in dataset._users:
        print(user)
    gaze_data = dataset.create_dataset()
    # gaze_data.to_csv(os.path.join(dataset_path + "_results", "all_data.csv"), sep=';', encoding='utf-8')
