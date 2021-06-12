# Basic
import os
import sys
import random
import time
import torch
import shutil
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import (List, NoReturn, Tuple, Union, Dict, Any)
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                             classification_report)

sys.path.insert(0, "..")
import warnings
warnings.filterwarnings('ignore')

import logging_handler
logger = logging_handler.get_logger(__name__)

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


def init_model(model: nn.Module, parameters: Dict[str, Any],
               dir: str='models_checkpoints',
               filename: str='model.pt') -> nn.Module:
    """
    Initialize model and load state dict.
    """
    model = model(**parameters.get("model_params"))
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

def create_embeddings(model: torch.nn.Module, dataloader,
                      estim_quality: bool) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[int]]]:
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
    embeddings = []
    if estim_quality:
        true_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if estim_quality:
                data = batch[0]
                target = batch[-1]
            else:
                data = batch
                target = None

            embedding = model.forward_one(data)
            embedding = embedding.detach().numpy()

            # Store labels
            if estim_quality:
                true_labels.extend(target.numpy().astype(int).tolist())

            # Store predictions
            embeddings.append(embedding)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - eval_start)

    print("\tTime elapsed for evaluation: {:} with {} samples.".format(validation_time, len(dataloader.dataset)))
    if estim_quality:
        return (embeddings, true_labels)
    else:
        return embeddings

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
    :param dir: folder/_path
    :param filename: state_dict filename
    :return: initialized model
    """
    return model.load_state_dict(torch.load(os.path.join(dir, filename)))



def compute_metrics(true_labels: List[int],
                    pred_labels: List[int]) -> NoReturn:

    print("***** Eval results *****")
    ac = accuracy_score(true_labels, pred_labels)
    bac = balanced_accuracy_score(true_labels, pred_labels)

    print('Accuracy score:', ac)
    print('Balanced_accuracy_score:', bac)
    print(classification_report(true_labels, pred_labels))


def clear_logs_dir(dir: str, ignore_errors: bool=True):
    """
    Reset logging directory with deleting all files inside.
    """
    files = len(os.listdir(dir))
    shutil.rmtree(dir, ignore_errors=ignore_errors)
    os.mkdir(dir)
    logger.info(f"Folder {dir} cleared, deleted {files} files.")


def save_losses_to_file(train_losses: Dict[int, List[float]],
                        val_losses: Dict[int, List[float]],
                        save_path: str=".", model_name: str="model"):
    """
    Create a DataFrame from training statistics with using the 'epoch' as the row index.
    Save it to .csv file.
    """
    trains = {k: np.mean(v) for k, v in train_losses.items()}
    vals = {k: np.mean(v) for k, v in val_losses.items()}
    df = pd.DataFrame({"Epoch": list(trains.keys()),
                       "Train Losses": list(trains.values()),
                       "Val Losses": list(vals.values()),})
    df.to_csv(os.path.join(save_path, model_name), sep=';')
