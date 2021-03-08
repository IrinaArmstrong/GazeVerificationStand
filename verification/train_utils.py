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
from typing import (List, NoReturn, Tuple)
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt

from config import config
from helpers import read_json
from verification.model import Siamese
from verification.train_metrics import AccumulatedAccuracyMetric, LossCallback

sys.path.insert(0, "..")
import warnings
warnings.filterwarnings('ignore')



class Trainer:

    def __init__(self, device: str='cpu'):
        self._is_fitted = False
        assert device in ['cpu', 'gpu']
        self._device = torch.device(device)
        self.__init_train_options()
        seed_everything(seed_value=11)


    def __init_train_options(self):
        parameters  = dict(read_json(config.get("GazeVerification", "model_params")))
        self.__lr = parameters.get("training_options", {}).get("base_lr", 1e-3)
        self.__n_epochs = parameters.get("training_options", {}).get("n_epochs", 10)
        self.__log_interval = parameters.get("training_options", {}).get("log_interval", 1)
        self.__scheduler_patience = parameters.get("training_options", {}).get("scheduler_patience", 1)
        self.__scheduler_threshold = parameters.get("training_options", {}).get("scheduler_max_lr", 1)
        self.__step_size_up = parameters.get("training_options", {}).get("step_size_up", 10)
        self.__warm_start_epoch = parameters.get("training_options", {}).get("warm_start_epoch", 0)

        self._model = Siamese(**parameters.get("siamese_model_params"))
        self.__loss_fn = torch.nn.BCEWithLogitsLoss()
        self.__optimizer = torch.optim.Adam(self._model.parameters(), lr=self.__lr)
        self.__scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer,
                                                                      patience=self.__scheduler_patience,
                                                                      threshold=self.__scheduler_threshold,
                                                                      verbose=True)
        self.__scheduler = torch.optim.lr_scheduler.CyclicLR(self.__optimizer,
                                                             base_lr=self.__lr,
                                                             max_lr=self.__scheduler_threshold,
                                                             step_size_up=self.__step_size_up,
                                                             mode="triangular", cycle_momentum=False)
        self.__metrics =  [AccumulatedAccuracyMetric(), LossCallback()]


    def fit(self, train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader, model_dir_to_save: str=".",
            model_filename: str="model.pt") -> nn.Module:

        current_time = datetime.now().strftime("%Y%m%d-%H_%M")
        print(f"Started training at: {current_time}")

        if self._device.type == 'cuda':
            self._model.to(self._device)
            print(f"Model on device: {self._device}")

        # Training
        print(f'---Start training with number of epochs = {self.__n_epochs}---')

        for epoch in range(0, self.__warm_start_epoch):
            self.__scheduler.step()

        for epoch in range(self.__warm_start_epoch, self.__n_epochs):

            # Train stage
            train_loss = self.train_epoch(train_loader, epoch)

            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, self.__n_epochs, train_loss)
            for metric in self.__metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            val_loss = self.test_epoch(val_loader, epoch)
            val_loss /= len(val_loader)

            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, self.__n_epochs,
                                                                                     val_loss)
            for metric in self.__metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)

            self.__scheduler.step(val_loss)

        self._is_fitted = True

        print("Saving...")
        save_model(self._model, filename=model_filename)
        self.plot_loss()
        return self._model


    def train_epoch(self, train_loader, epoch_num):

        for metric in self.__metrics:
            metric.reset()

        self._model.train()
        losses = []
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            data = batch[:-1]
            target = batch[-1]
            target = target.unsqueeze(-1).float() if len(target) > 0 else None

            if not type(data) in (tuple, list):
                data = (data,)

            if self._device.type == 'cuda':
                data = copy_data_to_device(data, self._device)
                if target is not None:
                    target = copy_data_to_device(target, self._device)

            self.__optimizer.zero_grad()
            outputs = self._model(*data)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = self.__loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            self.__optimizer.step()

            for metric in self.__metrics:
                metric(outputs, target[0], loss.item(), epoch_num)

            if batch_idx % self.__log_interval == 0:

                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data[0]), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(losses))
                for metric in self.__metrics:
                    message += '\t{}: {}'.format(metric.name(), metric.value())

                print(message)
                losses = []

        total_loss /= (batch_idx + 1)
        return total_loss


    def test_epoch(self, val_loader, epoch_num):

        with torch.no_grad():
            for metric in self.__metrics:
                metric.reset()

            self._model.eval()
            val_loss = 0
            for batch_idx, batch in enumerate(val_loader):
                data = batch[:-1]
                target = batch[-1]
                target = target.unsqueeze(-1).float() if len(target) > 0 else None
                if not type(data) in (tuple, list):
                    data = (data,)

                if self._device.type == 'cuda':
                    data = copy_data_to_device(data, self._device)
                    if target is not None:
                        target = copy_data_to_device(target, self._device)

                outputs = self._model(*data)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                loss_inputs = outputs
                if target is not None:
                    target = (target,)
                    loss_inputs += target

                loss_outputs = self.__loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                val_loss += loss.item()

                for metric in self.__metrics:
                    metric(outputs, target[0], loss.item(), epoch_num)

        return val_loss

    def plot_loss(self):
        """
        Plotting loss for both train and validation and save figure.
        :return: -
        """
        # Create a DataFrame from our training statistics and use the 'epoch' as the row index.
        df_stats = {k: np.mean(v) for k, v in self.__metrics[-1].losses.items()}
        df_stats = pd.DataFrame({"Epoch": list(df_stats.keys()), "Losses": list(df_stats.values())})

        # Plot the learning curve.
        plt.plot(df_stats['Losses'], 'b-o', label="Loss")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks(list(df_stats.index))
        plt.savefig('train_validation_loss.png')



def evaluate(model: torch.nn.Module, dataloader,
             estim_quality: bool, threshold: float,
             print_metrics: bool=True, binarize: bool=True) -> List[int]:
    """
    Making predictions with model.
    :param model: model instance to run;
    :param dataloader: DataLoader instance;
    :param estim_quality: to estimate quality of predictions;
    :return: predictions for given dataset.
    """
    seed_everything(11)
    eval_start = time.time()
    model.eval()
    # To store predictions and true labels
    pred_labels = []
    if estim_quality:
        true_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if estim_quality:
                data = batch[:-1]
                target = batch[-1]
            else:
                data = batch
                target = None

            if not type(data) in (tuple, list):
                data = (data,)

            outputs = torch.nn.functional.sigmoid(model(*data))
            if binarize:
                batch_pred = [1 if out > threshold else 0 for out in outputs.detach().numpy()]
            else:
                batch_pred = outputs.detach().tolist()

                # Store labels
            if estim_quality:
                true_labels.extend(target.numpy().astype(int).tolist())

            # Store predictions
            pred_labels.extend(batch_pred)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - eval_start)

    if estim_quality and print_metrics:
        compute_metrics(true_labels, pred_labels)

    print("\tTime elapsed for evaluation: {:} with {} samples.".format(validation_time, len(dataloader.dataset)))
    return pred_labels


def init_model(dir: str='models_checkpoints',
               filename: str='model.pt') -> nn.Module:
    """
    Initialize model and load state dict.
    :param dir: model dir
    :param filename: model filename
    :return: initialized model.
    """
    parameters = dict(read_json(config.get("GazeVerification", "model_params")))
    model = Siamese(**parameters.get("siamese_model_params"))
    _ = load_model(model, dir=dir,  filename=filename)
    print(model)
    return model


def aggregate_SP_predictions(predictions: List[float],
                             threshold: float, policy: str='mean') -> Tuple[int, float]:
    """
    Aggregate predictions for full session
    from list of predictions for each SP movement.
    :param predictions: list of predictions for each SP movement
    :param threshold: value above which verification is "successful" (1)
    :return: 1 - if verification is "successful"
             0 - if verification is "failed"
    """
    if policy == "mean":
        m = np.mean(predictions)
        return (1, m) if m > threshold else (0, m)
    elif policy == "max":
        m = np.max(predictions)
        return (1, m) if m > threshold else (0, m)
    elif policy.startswith('quantile'):
        q = float(policy.split("_")[-1])
        m = np.quantile(predictions, q=q)
        return (1, m) if m > threshold else (0, m)
    else:
        print("Specify correct predictions aggregation policy and try again.")
        return (0, 0.0)



#---------------------------- UTILITIES ----------------------------

def seed_everything(seed_value: int) -> NoReturn:
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # for using CUDA backend
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # get rid of nondeterminism
        torch.backends.cudnn.benchmark = True


def clean_GPU_memory() -> NoReturn:
    torch.cuda.empty_cache()


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Invalid data type {}'.format(type(data)))


def format_time(elapsed):
    """
    Service function. Pre-process timestamps during training.
    :param elapsed: time in seconds;
    :return: string with format: hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    return str(timedelta(seconds=elapsed_rounded))


def save_model(model, dir: str='models_checkpoints', filename: str='model.pt'):
    """
    Trained model, configuration and tokenizer,
    they can then be reloaded using `from_pretrained()` if using default names.
    """
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model.state_dict()
    torch.save(model_to_save, os.path.join(dir, filename))
    # models_checkpoints
    print("Model successfully saved.")


def load_model(model, dir: str, filename: str):
    """
    Loads a model’s parameter dictionary using a deserialized state_dict.
    :param model: model instance (uninitialized)
    :param dir: folder/path
    :param filename: state_dict filename
    :return: initialized model
    """
    return model.load_state_dict(torch.load(os.path.join(dir, filename)))



def compute_metrics(true_labels: List[int],
                    pred_labels: List[int]) -> NoReturn:
    print("***** Eval results {} *****")

    ac = accuracy_score(true_labels, pred_labels)
    bac = balanced_accuracy_score(true_labels, pred_labels)

    print('Accuracy score:', ac)
    print('Balanced_accuracy_score:', bac)
    print(classification_report(true_labels, pred_labels))