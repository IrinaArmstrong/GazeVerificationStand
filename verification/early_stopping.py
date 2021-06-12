import os
import sys
sys.path.insert(0, "..")
import torch
import numpy as np
from datetime import datetime

import logging_handler
logger = logging_handler.get_logger(__name__)

class EarlyStopping:
    """
    Early stops the training if validation loss
    doesn't improve after a given _patience.
    """

    def __init__(self, patience: int, verbose=False, delta=0,
                 path='models_checkpoints', trace_func=logger.info):
        """
        :param patience: How long to wait after last time validation loss improved.
        :param verbose: If True, logs a message for each validation loss improvement.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        :param path: Path for the checkpoint to be saved to.
        :param trace_func: function to log or print in stdout.
        """
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._best_score = None
        self._early_stop = False
        self._val_loss_min = np.Inf
        self._delta = delta
        self._path = path
        self._trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        Calls after each epoch end.
        """
        score = -val_loss
        if self._best_score is None:
            self._best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self._best_score + self._delta:
            self._counter += 1
            self._trace_func(f'EarlyStopping counter: {self._counter} out of {self._patience}')
            if self._counter >= self._patience:
                self._early_stop = True
        else:
            self._best_score = score
            self.save_checkpoint(val_loss, model)
            self._counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self._verbose:
            self._trace_func \
                (f'Validation loss decreased ({self._val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), os.path.join(self._path,
                                                    f"checkpoint_{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}.pt"))
        self._val_loss_min = val_loss