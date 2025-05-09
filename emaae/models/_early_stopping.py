"""
Early stopping

Source: https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
Last modified: 02/10/2025
"""
# IMPORTS
## built-in
from typing import Union
## local
from ._cnn_autoencoder import CNNAutoEncoder

class EarlyStopping:
    """
    Early stopping class

    :param patience: Number of epochs to wait before stopping if no improvement. (default = 5)
    :param delta: Minimum change in the monitored quantity to qualify as an improvement. (default = 0)
    """
    def __init__(self, patience:int=5, delta:int=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.best_model = None

    def __call__(self, val_loss: float, model:Union[CNNAutoEncoder], epoch: int):
        """
        Keep track of best loss and best model

        :param val_loss: float, validation loss
        :param model: current model
        :param epoch: int, current epoch
        """
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_model = model
            self.best_epoch = epoch
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_model = model
            self.best_epoch = epoch
            self.counter = 0

    def load_best_model(self, model: Union[CNNAutoEncoder]) -> Union[CNNAutoEncoder]:
        """
        Take in a model and load in the best weights

        :param model: current model
        :return: model with the best weights loaded
        """
        return model.load_state_dict(self.best_model_state)
    
    def get_best_model(self) -> tuple[Union[CNNAutoEncoder],int,int]:
        """
        :return self.best_model: best model during training
        :return self.best_epoch: best epoch
        """
        return self.best_model, self.best_epoch, self.best_score