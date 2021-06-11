# Basic
import os
import sys
import random
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import (List, NoReturn, Tuple, Union)

from config import config
from helpers import read_json
from verification.model import EmbeddingNet
from verification.loss import PrototypicalLoss
from verification.train_utils import seed_everything
from verification.train_metrics import LossCallback

class Trainer:

    def __init__(self):
        self.__is_fitted = False
        self._parameters  = dict(read_json(config.get("GazeVerification", "model_params")))
        # self._parameters.get("batching_options").get("num_support_train")
        self._device = torch.device(self._parameters.get("training_options", {}).get("device", "cpu"))
        self.__init_train_options()
        seed_everything(seed_value=11)


    def __init_train_options(self):

        self._model = EmbeddingNet(**self._parameters.get("model_params"))
        self.__loss_fn = PrototypicalLoss(self._device)
        self.__optimizer = torch.optim.Adam(params=self._model.parameters(),
                                     lr=self._parameters.get("training_options", {}).get("base_lr", 1e-4))
        self.__scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.__optimizer,
                                                           gamma=self._parameters.get("training_options", {}).get("lr_scheduler_gamma", 0.1),
                                                           step_size=self._parameters.get("training_options", {}).get("lr_step_size_up", 5))
        self._metrics = [LossCallback()]

