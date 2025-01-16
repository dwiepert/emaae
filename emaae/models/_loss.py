"""
Custom loss function

Author(s): Daniela Wiepert
Last modified: 01/10/2025
"""
#IMPORTS
#built-in
from typing import List, Dict, Optional
##third-party
import numpy as np
import torch
import torch.nn as nn
##local
from ._scheduler import StepAlpha
from emaae.utils import fro_norm3d_list

class SparseLoss(nn.Module):
    """
    Custom loss for training autoencoder - adds a sparsity loss and weights each loss

    :param alpha: float, loss weight (0-1)
    :param loss1_type: str, base autoencoder loss (default='mse')
    :param loss2_type: str, sparsity loss (default=L1 Norm)
    :param penalty_scheduler: str, scheduler type for updating alpha (default='step')
    :param **kwargs: additional penalty scheduler parameters
    """
    def __init__(self, alpha:float, loss1_type:str='mse',loss2_type:str='l1',penalty_scheduler:str='step',weight_penalty:bool=False, **kwargs):
        super(SparseLoss, self).__init__()
        self.alpha = alpha
        self.loss1_type = loss1_type.lower()
        if self.loss1_type == 'mse':
            self.loss1 = nn.MSELoss()
        else:
            raise NotImplementedError(f'{self.loss1_type} is not an implemented autoencoder loss function.')
        
        self.loss2_type = loss2_type.lower()
        if self.loss2_type == 'l1':
            self.loss2 = nn.L1Loss()
        else:
            raise NotImplementedError(f'{self.loss2_type} is not an implemented sparsity loss function.')
        
        self.penalty_scheduler = penalty_scheduler.lower()
        if self.penalty_scheduler == 'step':
            self.penalty_scheduler = StepAlpha(epochs=kwargs['epochs'])
        else:
            raise NotImplementedError(f'{self.penalty_scheduler} is not an implemented penalty scheduler function.')

        self.track_loss1 = []
        self.track_loss2 = []
        self.track_alpha = []

        self.weight_penalty = weight_penalty
        if self.weight_penalty:
            self.track_weight= []

    def step(self) -> None:
        """
        Increase alpha using penalty scheduler
        """
        new_alpha = self.penalty_scheduler(self.alpha)
        self.alpha = new_alpha

    def _weight_norm(self, weights: List[torch.Tensor]) -> float:
        """
        :param weights: list of torch.Tensors of all convolutional layer weights
        :return penalty: float, weight penalty
        """
        norm = fro_norm3d_list([w.numpy() for w in weights])
        penalty = np.square(norm-1)
        return penalty

    
    def forward(self, input:torch.Tensor, target:torch.Tensor, weights:Optional[List[torch.Tensor]]=None) -> float:
        """
        Calculate loss and add to log

        :param input: torch.Tensor, model output
        :param target: torch.Tensor, target matrix for comparison
        :param weights: optionally give list of torch.Tensors of all convolutional layer weights
        :return total_loss: calculated loss
        """
        loss1 = self.loss1(input, target)
        loss2 = self.loss2(input, target)

        total_loss = (1-self.alpha)*loss1 + self.alpha*loss2

        self.track_alpha.append(self.alpha)
        self.track_loss1.append(loss1.item())
        self.track_loss2.append(loss2.item())

        if self.weight_penalty:
            penalty = self._weight_norm(weights)
            total_loss += penalty

            self.track_weight.append(penalty)

        return total_loss
    
    def get_log(self) -> Dict[str,List[float]]:
        """
        Return log of alpha values and loss1/loss2

        :return log: Dictionary of tracked loss values
        """
        log = {'alpha':self.track_alpha, self.loss1_type:self.track_loss1, self.loss2_type:self.track_loss2}
        
        if self.weight_penalty:
            log['weight_penalty'] = self.track_weight
        return log
    
    def clear_log(self) -> None:
        self.track_loss1 = []
        self.track_loss2 = []
        self.track_alpha = []

        if self.weight_penalty:
            self.track_weight = []