# Basic
import sys
sys.path.insert(0, "..")
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn

from config import config
from helpers import read_json
from verification.model import EmbeddingNet
from verification.early_stopping import EarlyStopping
from verification.loss import PrototypicalLoss
from visualization import visualize_training_process
from verification.train_metrics import LossCallback, TensorboardCallback
from verification.train_utils import (seed_everything, clear_logs_dir,
                                      copy_data_to_device, save_losses_to_file)

import logging_handler
logger = logging_handler.get_logger(__name__)

class Trainer:

    def __init__(self):
        self.__is_fitted = False
        self._parameters  = dict(read_json(config.get("GazeVerification", "model_params")))
        self._device = torch.device(self._parameters.get("training_options", {}).get("device", "cpu"))
        self.__init_train_options()
        seed_everything(seed_value=11)


    def __init_train_options(self):

        self._model = EmbeddingNet(**self._parameters.get("model_params"))
        self.__loss_fn = PrototypicalLoss(self._device)
        self.__optimizer = torch.optim.Adam(params=self._model.parameters(),
                                            lr=self._parameters.get("training_options", {}).get("base_lr", 1e-4))
        self.__scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.__optimizer,
                                                           gamma=self._parameters.get(
                                                               "training_options",
                                                               {}).get("lr_scheduler_gamma", 0.1),
                                                           step_size=self._parameters.get(
                                                               "training_options",
                                                               {}).get("lr_step_size_up", 5))
        self._metrics = [LossCallback(), TensorboardCallback(log_dir=self._parameters.get(
            "training_options", {}).get("tensorboard_log_dir", "tblogs"))]


    def fit(self, train_loader:  torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader) -> nn.Module:

        """
        Loaders, model, loss function and metrics should work together for a given task,
        i.e. The model should be able to process data output of loaders,
        loss function should process target output of loaders and outputs from the model

        Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
        Siamese network: Siamese loader, siamese model, contrastive loss
        Online triplet learning: batch loader, embedding model, online triplet loss
        """

        # Add Tensorboard writer
        current_time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        clear_logs_dir(self._parameters.get("training_options",
                                            {}).get("tensorboard_log_dir", "tblogs"))

        if type(self._device) == str:
            self._device = torch.device(self._device)
        self._model.to(self._device)
        logger.info(f"Model moved to device: {self._device}")

        # Training
        logger.info(f'--- Start training with number of epochs = ',
                    f'{self._parameters.get("training_options", {}).get("n_epochs", 0)} ---')

        for epoch in range(0, self._parameters.get("training_options", {}).get("start_epoch", 0)):
            self.__scheduler.step()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self._parameters.get("training_options", {}).get("es_patience", 0),
                                       verbose=True)

        for epoch in range(self._parameters.get("training_options", {}).get("start_epoch", 0),
                           self._parameters.get("training_options", {}).get("n_epochs", 0)):

            # Train stage
            train_loss = self._train_epoch(train_loader, epoch)

            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1,
                                                                             self._parameters.get(
                                                                                 "training_options",
                                                                                 {}).get("n_epochs", 0),
                                                                             train_loss)
            for metric in self._metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            val_loss = self._test_epoch(val_loader)
            val_loss /= len(val_loader)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, self._model)

            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1,
                                                                                     self._parameters.get(
                                                                                         "training_options",
                                                                                         {}).get("n_epochs", 0),
                                                                                     val_loss)
            for metric in self._metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())
            print(message)

            self.__scheduler.step(val_loss)

            if early_stopping._early_stop:
                logger.info(f"Early stopping at {epoch} epoch.")
                break

        # Get loss history from LossCallback. Always first.
        save_losses_to_file(self._metrics[0].train_losses, self._metrics[0].train_losses,
                            save_path=self._parameters.get("training_options", {}).get("output_dir", "."))
        visualize_training_process()
        return self._model


    def _train_epoch(self, train_loader: torch.utils.data.DataLoader,
                     epoch_num: int):

        for metric in self._metrics:
            metric.reset()

        self._model.train()
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

            data = copy_data_to_device(data, self._device)
            if target is not None:
                target = copy_data_to_device(target, self._device)

            self.__optimizer.zero_grad()
            outputs = self._model(*data)
            if type(outputs) not in (tuple, list):
                outputs = (outputs.float(),)

            loss_inputs = outputs
            if target is not None:
                if bool(self._parameters.get("training_options", {}).get("to_unsqueeze", False)):
                    target = (target.float().unsqueeze(1),)
                else:
                    target = (target.float(),)
                loss_inputs += target
            loss_inputs += (self._parameters.get("batching_options", {}).get("num_support_train", 10),)

            # print("loss_inputs", loss_inputs)
            loss_outputs = self.__loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            acc = loss_outputs[1] if type(loss_outputs) in (tuple, list) else 0.0

            losses.append(loss.item())
            accs.append(acc.item())

            total_loss += loss.item()
            loss.backward()
            self.__optimizer.step()

            for metric in self._metrics:
                metric(outputs, target[0], loss.item(), "train", epoch_num)

            if batch_idx % self._parameters.get("training_options", {}).get("log_interval", False) == 0:

                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data[0]), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(losses))
                for metric in self._metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                logger.info(message)
                losses = []

        total_loss /= (batch_idx + 1)
        return total_loss

    def _test_epoch(self, val_loader: torch.utils.data.DataLoader,
                    epoch_num: int):
        with torch.no_grad():
            for metric in self._metrics:
                metric.reset()

            self._model.eval()
            val_loss = 0
            accs = []
            logger.info(f"---------------- Validation step --------------------")
            for batch_idx, batch in enumerate(val_loader):
                data = batch[:-1]
                target = batch[-1]
                target = target if len(target) > 0 else None
                if not type(data) in (tuple, list):
                    data = (data,)
                data = copy_data_to_device(data, self._device)
                if target is not None:
                    target = copy_data_to_device(target, self._device)

                outputs = self._model(*data)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs.float(),)
                loss_inputs = outputs
                if target is not None:
                    if bool(self._parameters.get("training_options", {}).get("to_unsqueeze", False)):
                        target = (target.float().unsqueeze(1),)
                    else:
                        target = (target.float(),)
                    loss_inputs += target
                loss_inputs += (self._parameters.get("batching_options", {}).get("num_support_train", 10),)

                loss_outputs = self.__loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                acc = loss_outputs[1] if type(loss_outputs) in (tuple, list) else 0.0

                val_loss += loss.item()
                accs.append(acc.item())

                for metric in self._metrics:
                    metric(outputs, target[0], loss.item(), "val", epoch_num)

        return val_loss

