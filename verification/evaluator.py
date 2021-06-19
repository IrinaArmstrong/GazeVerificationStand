# Basic
import os
import sys
sys.path.insert(0, "..")
import numpy as np
from datetime import datetime
from typing import Union

import torch
import torch.nn as nn

from config import config
from helpers import read_json
from verification.model import EmbeddingNet, PrototypeNet
from verification.train_utils import (seed_everything, copy_data_to_device, init_model)
import logging_handler
logger = logging_handler.get_logger(__name__)


class PrototypicalEvaluator:

    def __init__(self, embedding_model: nn.Module=None, to_load: bool=True):
        self._eval_parameters  = dict(read_json(config.get("GazeVerification", "verification_params")))
        self._device = torch.device(self._eval_parameters.get("training_options", {}).get("device", "cpu"))
        self.__init_prototypical_model(embedding_model, to_load)
        self.__init_parameters()
        seed_everything(seed_value=11)


    def __init_parameters(self, ):
        """

        :return:
        """


    def __init_prototypical_model(self, embedding_model: nn.Module, to_load: bool):
        """
        Create Prototypical model form given Embedding base model or loaded from file.
        :param embedding_model: pre-trained model;
        :param to_load: whether to load pre-trained weights from file;
        """
        self._model_parameters = dict(read_json(config.get("GazeVerification", "model_params")))
        if to_load:
            fname = os.path.join(sys.path[0],
                                 self._eval_parameters.get("pretrained_model_location", "."),
                                 self._eval_parameters.get("model_name", "model.pt"))
            if not os.path.isfile(fname):
                logger.error(f"No pretrained model file found in given path: {fname}.\n",
                             f"Check path and retry.")
                return
            else:
                logger.info(f"Loading model from: {fname}")
                embedding_model = init_model(EmbeddingNet, parameters=self._model_parameters,
                                          dir=self._eval_parameters.get("pretrained_model_location", "."),
                                          filename=self._eval_parameters.get("model_name", "model.pt"))
        self._protypical_model = PrototypeNet(embedding_model)
        logger.info(f"Prototypical model created.")
