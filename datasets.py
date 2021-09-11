import os
import glob
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import (List, Union, Tuple, Any)

from config import init_config, config

import logging_handler
logger = logging_handler.get_logger(__name__)


class Session:

    def __init__(self, session_path: str, session_id: int, user_id: int):
        self._session_id = session_id
        self._session_path = session_path
        self._user_id = user_id

        self._dataset_fn = "_".join(self._session_path.split("_")[:-1]) + ".csv"
        if not Path(self._dataset_fn).exists():
            logger.error(f"""Error during session #{self._session_id} creation. 
            Given session path do not exists: {self._dataset_fn}""")
            raise FileNotFoundError(f"""Error during session #{self._session_id} creation. 
            Given session path do not exists: {self._dataset_fn}""")

        self._stimulus_type = "_".join(self._session_path.split("\\")[-1].replace("__", "_").split("_")[-12:-10])
        self._stimulus_type = "kot#0" if "kot" in self._stimulus_type else self._stimulus_type
        self._stimulus_type = "sobaka#0" if "sobaka" in self._stimulus_type else self._stimulus_type

    def get_gaze_data(self) -> pd.DataFrame:
        """
        Get data from eyetracker for current session.
        todo add_video: to add data from video and stimulus.
        """
        gaze_df = pd.read_csv(self._dataset_fn, sep='\t')
        gaze_df.columns = [col.strip() for col in gaze_df.columns]
        gaze_df['timestamp'] = pd.to_datetime(gaze_df['timestamp'], unit='s')
        gaze_df['session_id'] = self._session_id
        gaze_df['stimulus_type'] = self._stimulus_type
        gaze_df['user_id'] = self._user_id
        gaze_df["x_diff"] = gaze_df["stim_X"] - gaze_df["gaze_X"]
        gaze_df["y_diff"] = gaze_df["stim_Y"] - gaze_df["gaze_Y"]
        gaze_df['filename'] = self._dataset_fn
        return gaze_df


class User:

    def __init__(self, user_id: int, user_name: str,
                 sessions_fns: List[str]):
        self._user_id = user_id
        self._user_name = user_name
        self._sessions_fns = sessions_fns
        self._sessions = self.__create_sessions()
        logger.info(f"For user #{self._user_id} `{self._user_name}` created {len(self._sessions)} sessions.")

    def __create_sessions(self) -> List[Session]:
        sessions = []
        for i, sess_fn in enumerate(self._sessions_fns):
            try:
                s = Session(session_path=sess_fn, session_id=i, user_id=self._user_id)
                sessions.append(s)
            except Exception as ex:
                logger.error(f"""Exception occurred during creating session: {i}\n
                                {traceback.print_tb(ex.__traceback__)}""")
        return sessions

    def get_num_sessions(self) -> int:
        return len(self._sessions)

    def get_user_name(self) -> str:
        return self._user_name

    def get_sessions(self) -> List[Session]:
        return self._sessions

    def get_sessions_fns(self) -> List[str]:
        return self._sessions_fns

    def get_session(self, sess_id: int) -> Session:
        if sess_id > len(self._sessions):
            logger.error(f"Session id:{sess_id} index out of range of available sessions: {len(self._sessions)}")
            return None
        return self._sessions[sess_id]

    def __str__(self) -> str:
        s = f"User #{self._user_id}: {self._user_name}\n"
        s += f"\nUser has {len(self._sessions)} sessions."
        s += f"\nWith {np.unique([sess._stimulus_type for sess in self._sessions])} stimulus types."
        s += f"\nSessions"
        return s

    def __eq__(self, other: Any) -> Any:
        return True if self._user_name.strip().lower() == other.get_user_name().strip().lower() else False

    def __hash__(self):
        return hash(self._user_name)

    def __add__(self, other: Any) -> Any:
        self._sessions_fns.extend(other.get_sessions_fns())
        self._sessions.extend(other.get_sessions())
        return self


class TrainDataset:

    def __init__(self, ds_path: str):

        self._path = Path(ds_path).resolve()
        if not self._path.exists():
            logger.error(f"Provided in configuration data path do not exists: {self._path}")
            logger.error(f"Check it and try again.")
            raise FileNotFoundError(f"Provided in configuration data path do not exists: {self._path}")

        self._current_dir = Path().resolve()

        if not Path(config.get("DataPaths", "selected_columns")).exists():
            logger.error(f"Provided in configuration selected columns file path do not exists: ")
            logger.error(f"{config.get('DataPaths', 'selected_columns')}")
            logger.error(f"Check it and try again.")
            raise FileNotFoundError(f"Provided in configuration selected columns file path do not exists.")

        # todo: change to simple selected_columns.txt file
        self._selected_columns = pd.read_csv(config.get("DataPaths", "selected_columns"), header=0,
                                             index_col=0, squeeze=True).to_list()
        self._users, self._users_ids = self.__create_users()
        self._sessions = self.__create_sessions()

    def __create_users(self) -> Tuple[List[User], List[int]]:
        """
        Create users based on metadata files provided in configuration data directory.
        """
        meta_files = list(self._path.glob("*metadata.txt"))
        if len(meta_files) == 0:
            logger.error(f"No metadata files found in data directory: {str(self._path)}")
            raise FileNotFoundError(f"No metadata files found in data directory: {str(self._path)}")

        meta_df = []
        for mfn in meta_files:
            try:
                mdf = pd.read_csv(str(mfn), delimiter="\t", encoding="Windows-1251",
                                  header=None, error_bad_lines=False).transpose()
                mdf.columns = mdf.iloc[0]
                mdf = mdf.drop(labels=0, axis=0).dropna(how='all')
                mdf['filename'] = str(mfn)
                meta_df.append(mdf)
            except Exception as ex:
                logger.error(f"""Exception occurred during reading metadata file: {str(mfn)}\n
                            {traceback.print_tb(ex.__traceback__)}""")
        if len(meta_df) == 0:
            logger.error(f"No metadata files read successfully. Check parameters and data formats.")
            raise FileNotFoundError(f"No metadata files read successfully.")

        # todo: Check here with try: except:
        meta_df = pd.concat(meta_df).reset_index(drop=True)
        meta_df['full_name'] = (meta_df['last_name'].fillna("") + " " + meta_df['first_name'].fillna("")).str.strip()
        meta_df['full_name'] = meta_df['full_name'].apply(lambda x: x.replace(r"  ", r" "))
        meta_df['session_filename'] = meta_df.filename.apply(lambda x: ("_".join(x.split("_")[:-1]) + ".csv"))
        meta_df['user_id'] = meta_df.full_name.replace(
            to_replace={user: i for i, user in enumerate(meta_df.full_name.unique())})

        meta_df = meta_df.groupby(by=['user_id', 'full_name']).agg({'filename': lambda x: list(x),
                                                                    'session_filename': lambda x: list(x)}).reset_index()

        users = []
        for i, row in meta_df[['user_id', 'full_name', 'filename']].iterrows():
            try:
                u = User(user_id=row['user_id'], user_name=row['full_name'], sessions_fns=row['filename'])
                users.append(u)
            except Exception as ex:
                logger.error(f"Error occurred during creation user: {traceback.print_tb(ex.__traceback__)}")

        if len(users) == 0:
            logger.error(f"No user was created successfully. Check parameters and data formats.")
            return users, []

        return users, meta_df['user_id'].to_list()

    def __create_sessions(self) -> List[Session]:
        sessions = []
        for user_num, ds_user in enumerate(self._users):
            for sess_num in range(ds_user.get_num_sessions()):
                try:
                    s = ds_user.get_session(sess_num)
                    sessions.append(s)
                except Exception as ex:
                    logger.error(f"Error occurred during creation dataset with session #{sess_num} of user #{user_num}")
                    logger.error(f"{traceback.print_tb(ex.__traceback__)}")

        return sessions

    def get_user(self, user_id: int) -> Union[User, None]:
        """
        Get User object by id.
        :type user_id: int, id of desires user
        """
        if user_id in self._users_ids:
            return self._users[self._users_ids.index(user_id)]
        else:
            logger.error(f"User index: {user_id} out of range of available users: {self._users_ids}")
            return None

    def get_users(self) -> List[User]:
        """
        Get all users provided in data.
        """
        return self._users

    def create_dataset(self) -> Union[pd.DataFrame, None]:
        sess_df = []
        for i, sess in tqdm(enumerate(self._sessions), total=len(self._sessions)):
            try:
                sess_data = sess.get_gaze_data()
                sess_df.append(sess_data)
            except Exception as ex:
                logger.error(f"Error occurred during creation dataset with session #{i}:")
                logger.error(f"{traceback.print_tb(ex.__traceback__)}")

        if len(sess_df) == 0:
            logger.error(f"No sessions were created successfully. Check parameters and data formats.")
            return None

        # Re-enumerate sessions ids
        for total_sess_ind, sess in enumerate(sess_df):
            sess['session_id'] = total_sess_ind
        sess_df = pd.concat(sess_df, axis=0)[self._selected_columns]
        return sess_df


class RunDataset:

    def __init__(self, owner_path: str, others_path: str,
                 estimate_quality: bool):

        self._current_dir = Path().resolve()

        self._owner_path = Path(owner_path).resolve()
        if not self._owner_path.exists():
            logger.error(f"Provided in configuration owner data path do not exists: {self._owner_path}")
            logger.error(f"Check it and try again.")
            raise FileNotFoundError(f"Provided in configuration owner data path do not exists: {self._owner_path}")

        self._others_path =  Path(others_path).resolve()
        if not self._others_path.exists():
            logger.error(f"Provided in configuration owner data path do not exists: {self._others_path}")
            logger.error(f"Check it and try again.")
            raise FileNotFoundError(f"Provided in configuration owner data path do not exists: {self._others_path}")

        self._estimate_quality = estimate_quality
        # todo: change to simple selected_columns.txt file
        self._selected_columns = pd.read_csv(config.get("DataPaths", "selected_columns"), header=0,
                                             index_col=0, squeeze=True).to_list()
        self._owner = self.__create_owner()
        self._others, self._users_targets = self.__create_others()
        self._others_sessions, self._sessions_targets = self.__create_sessions()

    def __create_owner(self):
        """
        Create `owner` i.e. the person whom we want to verify.
        """
        meta_file = list(self._owner_path.glob("*metadata.txt"))
        if len(meta_file) == 0:
            logger.error(f"No metadata files found in data directory: {str(self._owner_path)}")
            raise FileNotFoundError(f"No metadata files found in data directory: {str(self._owner_path)}")
        elif len(meta_file) > 1:
            logger.error(f"More then singe file identifying gaze owner found in: {str(self._owner_path)}")
            logger.error(f"Found files: {[str(m) for m in meta_file]}")
            raise ValueError("More then singe file identifying gaze owner. Select one and try again.")
        else:
            meta_file = str(meta_file[0])  # select one

        mdf = pd.read_csv(meta_file, delimiter="\t", encoding="Windows-1251",
                          header=None, names=['name'])
        mdf['name'] = mdf['name'].str.replace('\t', ' ', regex=True)
        mdf['name'] = mdf['name'].str.strip()
        mdf['filename'] = meta_file
        mdf['user_id'] = 1
        try:
            owner = User(user_id=1, user_name=mdf['name'].values[0], sessions_fns=mdf['filename'].to_list())
        except Exception as ex:
            logger.error(f"Error occurred during creation owner user with file #{meta_file}")
            logger.error(f"{traceback.print_tb(ex.__traceback__)}")
        return owner

    def __create_others(self):
        """
        Creates a test users based on whether we know the target variable of the sessions or not.
        If self._estimate_quality - then we know target values, so create users with their real ids
        otherwise - mark them all as single user and assume target variable as 0.
        """
        if self._estimate_quality:

            meta_files = list(self._others_path.glob("*metadata.txt"))
            if len(meta_files) == 0:
                logger.error(f"No metadata files found in verifiable users data directory: {str(self._others_path)}")
                raise FileNotFoundError(f"No metadata files found in verifiable users data directory.")

            meta_df = []
            for mfn in meta_files:
                try:
                    mdf = pd.read_csv(str(mfn), delimiter="\t", encoding="Windows-1251",
                                      header=None, error_bad_lines=False).transpose()
                    mdf.columns = mdf.iloc[0]
                    mdf = mdf.drop(labels=0, axis=0).dropna(how='all')
                    mdf['filename'] = str(mfn)
                    meta_df.append(mdf)
                except Exception as ex:
                    logger.error(f"""Exception occurred during reading metadata file: {str(mfn)}\n
                                        {traceback.print_tb(ex.__traceback__)}""")
            if len(meta_df) == 0:
                logger.error(f"No metadata files read successfully. Check parameters and data formats.")
                raise FileNotFoundError(f"No metadata files read successfully.")

            # todo: Check here with try: except:
            meta_df = pd.concat(meta_df).reset_index(drop=True)
            meta_df['full_name'] = (
                        meta_df['last_name'].fillna("") + " " + meta_df['first_name'].fillna("")).str.strip()
            meta_df['full_name'] = meta_df['full_name'].apply(lambda x: x.replace(r"  ", r" "))
            meta_df['session_filename'] = meta_df.filename.apply(lambda x: ("_".join(x.split("_")[:-1]) + ".csv"))
            meta_df['user_id'] = meta_df.full_name.replace(to_replace={u: i for i, u
                                                                       in enumerate(meta_df.full_name.unique())})

            # meta_df = pd.concat(meta_df).groupby(by='full_name').agg({'session_filename':
            #                                                               lambda x: list(x)}).reset_index()
            meta_df['user_id'] = np.arange(0, len(meta_df))

            users = []
            for i, row in meta_df[['user_id', 'full_name', 'session_filename']].iterrows():
                try:
                    u = User(user_id=row['user_id'], user_name=row['full_name'], sessions_fns=row['session_filename'])
                    users.append(u)
                except Exception as ex:
                    logger.error(f"Error occurred during creation user: {traceback.print_tb(ex.__traceback__)}")

            if len(users) == 0:
                logger.error(f"No user was created successfully. Check parameters and data formats.")
                return users, []

            users_targets = [1 if u == self._owner else 0 for u in users]
            logger.info(f"Found {users_targets.count(1)} sessions that belongs verified user and {users_targets.count(0)} other's sessions. ")
            return users, users_targets

        # Otherwise we do not know who's files are
        else:
            data_files = [str(fn) for fn in list(self._others_path.glob("*.csv"))]
            try:
                # all ids are 0 and user is "Unknown"
                users = [User(user_id=0, user_name="Unknown", sessions_fns=data_files)]
            except Exception as ex:
                logger.error(f"Error occurred during creation user: {traceback.print_tb(ex.__traceback__)}")
                raise FileNotFoundError(f"No user was created successfully. Check parameters and data formats.")

            users_targets = [0]  # all targets are set to 0
            return users, users_targets

    def __create_sessions(self) -> Tuple[List[Session], List[int]]:
        """
        Create other users sessions and their targets.
        """
        if self._estimate_quality:
            sessions = []
            sessions_targets = []
            for u, target in zip(self._others, self._users_targets):
                for sess in u.get_sessions():
                    sessions.append(sess)
                    sessions_targets.append(target)
            return sessions, sessions_targets
        else:
            return self._others[0].get_sessions(), [0] * self._others[0].get_num_sessions()

    def get_owner(self) -> User:
        return self._owner

    def get_others(self) -> List[User]:
        return self._others

    def get_owner_data(self) -> pd.DataFrame:
        """
        Get owner gaze datset.
        """
        return self._owner.get_session(0).get_gaze_data()[self._selected_columns]

    def get_others_data(self) -> pd.DataFrame:
        """
        Get other users joint gaze dataset.
        """
        sess_df = []
        for i, (sess, sess_targ) in tqdm(enumerate(zip(self._others_sessions, self._sessions_targets)),
                                         total=len(self._others_sessions)):
            sess_part_df = sess.get_gaze_data()
            sess_part_df['session_id'] = i
            sess_part_df['session_target'] = sess_targ
            sess_df.append(sess_part_df)

        if self._estimate_quality:
            self._selected_columns += ['session_target']
        sess_df = pd.concat(sess_df, axis=0)[self._selected_columns]
        logger.info(f"Other users dataset of shape: {sess_df.shape}")
        return sess_df





if __name__ == "__main__":

    config_path = ".\\set_locations.ini"
    # test_unseen train
    dataset_path = "D:\\Data\\EyesSimulation Sessions\\Export_full\\test_seen"
    init_config(config_path)
    dataset = TrainDataset(dataset_path)
    print(f"Unique users: {len(dataset._users)} with sessions: {len(dataset._sessions)}")
    for user in dataset._users:
        print(user)
    gaze_data = dataset.create_dataset()
    print(gaze_data.shape)
    # all_test_unseen_data
    gaze_data.to_csv(os.path.join(os.path.dirname(dataset_path), "results", "all_test_seen_data.csv"),
                     sep=';', encoding='utf-8')
