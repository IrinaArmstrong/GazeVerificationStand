# Basic
import os
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import config
from helpers import read_json
from verification.early_stopping import EarlyStopping
from verification.metrics import LossCallback
from visualizations.visualization import visualize_training_process
from verification.train_utils import (seed_everything, copy_data_to_device, save_model)

import logging_handler
logger = logging_handler.get_logger(__name__)


class Trainer:

    def __init__(self, **kwargs):
        """
        Class for training selected model with provided parameters.
        It keeps saving training statistics, checkpoints (is selected such option)
        and final model weights in to new directory named in unique way.
        kwargs: {'model', 'loss'}
        """
        self.__is_fitted = False

        parameters_fn = config.get("GazeVerification", "model_params")
        if not Path(parameters_fn).exists():
            logger.error(f"File with training model parameters was not found by the provided path: {parameters_fn}")
            raise FileNotFoundError(f"File with training model parameters was not found by the provided path.")

        logger.info(f"Loading training model parameters from {parameters_fn}")
        self.__general_parameters = dict(read_json(parameters_fn))
        self.__models_parameters = self.__general_parameters.get("model_params", {})
        self.__batching_parameters = self.__general_parameters.get("batching_options", {})
        self.__training_parameters = self.__general_parameters.get("training_options", {})
        logger.info(f"Training general parameters: {self.__general_parameters}")

        # Set device type
        self.__acquire_device()
        self.__init_train_options(kwargs)
        seed_everything(seed_value=11)

    def __acquire_device(self):
        """ Init training device. """
        self.__device = torch.device(self.__training_parameters.get("device", "cpu"))
        logger.info(f"Training device: {self.__device.type}")

        if self.__device == 'gpu':
            if not torch.cuda.is_available():
                logger.error(f"Device provided is CUDA, but it is available. Change to CPU.")
                self.__device = torch.device('cpu')
            else:
                logger.info(torch.cuda.get_device_name(0))
                logger.info('Memory Usage, Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                logger.info('Memory Usage, Cached:', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    def __init_train_options(self, **kwargs):
        """
        Initialize training parameters, model and optimizer hyperparameters.
        """
        # Init model
        self.__model = kwargs.get('model', None)
        if self.__model is None:
            logger.error(f"No model class provided in kwargs for training.")
            raise AttributeError(f"No model class provided in kwargs for training.")

        try:
            self.__model = self.__model(**self.__models_parameters)
        except Exception as ex:
            logger.error(f"Exception occurred during initializing model: {traceback.print_tb(ex.__traceback__)}")
            raise ex

        # Init loss
        self.__loss_fn = kwargs.get('loss', None)
        if self.__loss_fn is None:
            logger.error(f"No loss class provided in kwargs for training.")
            raise AttributeError(f"No loss class provided in kwargs for training.")

        try:
            self.__loss_fn = self.__loss_fn(self.__device)
        except Exception as ex:
            logger.error(f"Exception occurred during initializing loss: {traceback.print_tb(ex.__traceback__)}")
            raise ex

        # Optimizer
        self.__optimizer = kwargs.get('optimizer', None)
        if self.__optimizer is None:
            logger.error(f"No optimizer class provided in kwargs for training, set default.")
            self.__optimizer = torch.optim.Adam(params=self.__model.parameters(), lr=1e-4)

        try:
            self.__optimizer = self.__optimizer(params=self.__model.parameters(),
                                                **self.__training_parameters.get("optimizer_kwargs", {}))
        except Exception as ex:
            logger.error(f"Exception occurred during initializing optimizer: {traceback.print_tb(ex.__traceback__)}")
            raise ex

        # Optimizer
        self.__scheduler = kwargs.get('lr_scheduler', None)
        if self.__scheduler is None:
            logger.error(f"No lr_scheduler class provided in kwargs for training, set default.")
            self.__scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.__optimizer,
                                                               gamma=0.1, step_size=5)
        try:
            self.__scheduler = self.__scheduler(optimizer=self.__optimizer,
                                                **self.__training_parameters.get("lr_scheduler_kwargs", {}))
        except Exception as ex:
            logger.error(
                f"Exception occurred during initializing lr_scheduler: {traceback.print_tb(ex.__traceback__)}")
            raise ex

        # Metrics
        self.__metrics = {}
        metrics = kwargs.get('metrics', None)
        # Metrics not provided
        if metrics is None:
            logger.error(f"No metrics list provided in kwargs for training, set default: Loss Callback")
            self.__metrics = {LossCallback.name(): LossCallback()}

        # Metrics are user provided
        else:
            for metric in metrics:
                try:
                    metrics_params = self.__training_parameters.get("metrics_kwargs", {}).get(metric.name(), {})
                    self.__metrics.update({metric.name: metric(**metrics_params)})
                except Exception as ex:
                    logger.error(
                        f"Exception occurred during initializing metric: {traceback.print_tb(ex.__traceback__)}")

        logger.info(f"Metrics for training: {[m.name() for m in self.__metrics]}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:

        """
        Loaders, model, loss function and metrics should work together for a given task,
        i.e. The model should be able to process data output of loaders,
        loss function should process target output of loaders and outputs from the model

        Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
        Siamese network: Siamese loader, siamese model, contrastive loss
        Online triplet learning: batch loader, embedding model, online triplet loss
        """

        current_time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        logger.info(f"Training started at {current_time}")

        self.__model.to(self.__device)
        logger.info(f"Model moved to device: {self.__device}")

        # Training
        logger.info(f'Start training with number of epochs = {self.__training_parameters.get("n_epochs", 0)}')

        for epoch in range(0, self.__training_parameters.get("start_epoch", 0)):
            self.__scheduler.step()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.__training_parameters.get("es_patience", 0),
                                       verbose=True,
                                       path=self.__training_parameters.get("checkpoints_dir", "checkpoints_dir"))

        for epoch in range(self.__training_parameters.get("start_epoch", 0),
                           self.__training_parameters.get("n_epochs", 0)):

            # Train stage
            for metric_name, metric in self.__metrics.items():
                metric.on_start()
            train_loss = self._train_epoch(train_loader, epoch)
            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1,
                                                                             self.__general_parameters.get(
                                                                                 "training_options",
                                                                                 {}).get("n_epochs", 0),
                                                                             train_loss)
            for metric_name, metric in self.__metrics.items():
                message += '\t{}: {}'.format(metric_name, metric.value())

            val_loss = self._test_epoch(val_loader, epoch)
            val_loss /= len(val_loader)

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, self.__model)
            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1,
                                                                                     self.__general_parameters.get(
                                                                                         "training_options",
                                                                                         {}).get("n_epochs", 0),
                                                                                     val_loss)
            for metric_name, metric in self.__metrics.items():
                message += '\t{}: {}'.format(metric_name, metric.value())
                metric.on_close()
            logger.info(message)

            self.__scheduler.step(val_loss)

            if early_stopping.need_to_stop():
                logger.info(f"Early stopping at {epoch} epoch.")
                break

        # Saving final version
        save_model(self.__model,
                   dir=self.__training_parameters.get("checkpoints_dir", "checkpoints_dir"),
                   filename=self.__training_parameters.get("model_name", "model") + "_last_epoch.pt")

        # Get loss history from LossCallback. Always first.
        loss_metric = self.__metrics.get(LossCallback.name(), None)
        if loss_metric is not None:
            # todo: create separate folder for each experiment and provide it through kwargs - ???
            loss_metric.to_file(file_path=self.__training_parameters.get("output_dir", "."))

        # Save and show training history
        try:
            visualize_training_process(loss_fn=os.path.join(self.__general_parameters.get("training_options",
                                                                                          {}).get("output_dir", "."),
                                                            self.__general_parameters.get(
                                                                "training_options", {}).get("model_name", "model")
                                                            + "_losses.csv"))
        except FileNotFoundError:
            logger.error(f"Losses file not found for visualization.")

        return self.__model

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader,
                     epoch_num: int):

        for metric in self.__metrics:
            metric.reset()

        self.__model.train()
        losses = []
        accs = []
        total_loss = 0
        logger.info(f"---------------- Train step --------------------")
        for batch_idx, batch in enumerate(train_loader):
            data = batch[:-1]
            target = batch[-1]

            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)

            data = copy_data_to_device(data, self.__device)
            if target is not None:
                target = copy_data_to_device(target, self.__device)

            self.__optimizer.zero_grad()
            outputs = self.__model(*data)
            if type(outputs) not in (tuple, list):
                outputs = (outputs.float(),)

            loss_inputs = outputs
            if target is not None:
                if bool(self.__general_parameters.get("training_options", {}).get("to_unsqueeze", False)):
                    target = (target.float().unsqueeze(1),)
                else:
                    target = (target.float(),)
                loss_inputs += target
            loss_inputs += (self.__general_parameters.get("batching_options", {}).get("num_support_train", 10),)

            # print("loss_inputs", loss_inputs)
            loss_outputs = self.__loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            acc = loss_outputs[1] if type(loss_outputs) in (tuple, list) else 0.0

            losses.append(loss.item())
            accs.append(acc.item())

            total_loss += loss.item()
            loss.backward()
            self.__optimizer.step()

            for metric in self.__metrics:
                metric(outputs, target[0], loss.item(), "train", epoch_num)

            if batch_idx % self.__general_parameters.get("training_options", {}).get("log_interval", False) == 0:

                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data[0]), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(losses))
                for metric in self.__metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                logger.info(message)
                losses = []

        total_loss /= (batch_idx + 1)
        return total_loss

    def _test_epoch(self, val_loader: torch.utils.data.DataLoader,
                    epoch_num: int):
        with torch.no_grad():
            for metric in self.__metrics:
                metric.reset()

            self.__model.eval()
            val_loss = 0
            accs = []
            logger.info(f"---------------- Validation step --------------------")
            for batch_idx, batch in enumerate(val_loader):
                data = batch[:-1]
                target = batch[-1]
                target = target if len(target) > 0 else None
                if not type(data) in (tuple, list):
                    data = (data,)
                data = copy_data_to_device(data, self.__device)
                if target is not None:
                    target = copy_data_to_device(target, self.__device)

                outputs = self.__model(*data)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs.float(),)
                loss_inputs = outputs
                if target is not None:
                    if bool(self.__general_parameters.get("training_options", {}).get("to_unsqueeze", False)):
                        target = (target.float().unsqueeze(1),)
                    else:
                        target = (target.float(),)
                    loss_inputs += target
                loss_inputs += (self.__general_parameters.get("batching_options", {}).get("num_support_train", 10),)

                loss_outputs = self.__loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                acc = loss_outputs[1] if type(loss_outputs) in (tuple, list) else 0.0

                val_loss += loss.item()
                accs.append(acc.item())

                for metric in self.__metrics:
                    metric(outputs, target[0], loss.item(), "val", epoch_num)

        return val_loss

