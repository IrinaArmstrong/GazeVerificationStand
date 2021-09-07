import pandas as pd
from typing import List, Callable, Dict, AnyStr, Union, TypeVar

import logging_handler
logger = logging_handler.get_logger(__name__)

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

available_modes = [
    'folder',
    'file',
    'stdout'
]

class EyemovementsExperimenter:

    """
    Class for creating, running, saving an experiment (with parameters)
    of eye movements classification.
    """


    def __init__(self, output_mode: str, **kwargs):
        # experiments settings
        self.__metrics_list = kwargs.get("metrics_list", [])

        # saving parameters
        self.__output_mode = output_mode
        if self.__output_mode not in available_modes:
            logger.error(f"""Eye movements Experiment mode should be one from: {available_modes}.
                         Given type {self.__output_mode} is unrecognized.""")
            raise AttributeError(f"""Eye movements Experiment mode should be one from: {available_modes}.
                         Given type {self.__output_mode} is unrecognized.""")

        self.__visualize = kwargs.get("visualize", False)
        self.__saving_kwargs = kwargs.get("saving_kwargs", {})  # pass to saving func

    def __save_results(self):
        """
        Save experiment's results (as metrics / output visulizations / data samples)
        with used parameters.
        """
        pass