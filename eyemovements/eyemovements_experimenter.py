import pandas as pd
from typing import List, Callable, Dict, AnyStr, Union, TypeVar


class Experiment:
    """
    DTO for storing and saving results of eye movements classification experiment.
    """

    def __init__(self):
        """
        Initial parameters.
        """
        self._data_filenames = ['']
        self._metrics_list = ['']
        self._experiment_name = ''

    def result_to_file(self):
        """
        Save experiment's result to separate file (xlsx/csv).
        """
        pass

    def result_to_string(self) -> str:
        """
        Save experiment's result to string (human-readable/comma-separated).
        """
        pass


class EyemovementsExperimenter:

    """
    Class for creating, running, saving an experiment (with parameters)
    of eye movements classification.
    """

    def __init__(self, metrics_list: List[Callable], **kwargs):
        # experiments settings
        self.__metrics_list = metrics_list

        # saving parameters
        self.__to_save = kwargs.get("to_save", False)
        self.__to_existing_file = kwargs.get("to_existing_file", False)
        self.__to_new_folder = kwargs.get("to_new_folder", False)

        saving_kwargs = kwargs.get("saving_kwargs", {})  # pass to saving func

    def __save_results(self):
        """
        Save experiment's results (as metrics / output visulizations / data samples)
        with used parameters.
        """
        pass