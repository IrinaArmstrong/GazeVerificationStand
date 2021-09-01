import pandas as pd
from typing import List, Callable, Dict, AnyStr, Union, TypeVar


class EyemovementsEstimator:

    def __init__(self, metrics_list: List[Callable]):
        self.__metrics_list = metrics_list


    def estimate_session(self, data: pd.DataFrame) -> Dict[AnyStr, AnyStr]:
        """
        Estimate single session quality.
        :return: dict with key - metric name, value - score.
        """
        pass

    def estimate_dataset(self, data: List[pd.DataFrame],
                         to_average: bool) -> Union[Dict[AnyStr, AnyStr], List[Dict[AnyStr, AnyStr]]]:
        """
        Estimate single session quality.
        :return: dict with key - metric name, value - score.
        """
        pass
