"""
Custom loss function

Author(s): Daniela Wiepert
Last modified: 01/10/2025
"""
#IMPORTS
#built-in
from typing import List, Dict
##third-party
import torch
import torch.nn as nn
##local
from ._scheduler import StepAlpha

class SparseLoss(nn.Module):
    """
    Custom loss for training autoencoder - adds a sparsity loss and weights each loss

    :param alpha: float, loss weight (0-1)
    :param loss1_type: str, base autoencoder loss (default='mse')
    :param loss2_type: str, sparsity loss (default=L1 Norm)
    :param penalty_scheduler: str, scheduler type for updating alpha (default='step')
    :param **kwargs: additional penalty scheduler parameters
    """
    def __init__(self, alpha:float, loss1_type:str='mse',loss2_type:str='l1',penalty_scheduler:str='step', **kwargs):
        super(SparseLoss, self).__init__()
        self.alpha = alpha
        self.loss1_type = loss1_type.lower()
        if self.loss1_type == 'mse':
            self.loss1 = nn.MSELoss()
        else:
            raise NotImplementedError(f'{self.loss1_type} is not an implemented autoencoder loss function.')
        
        self.loss2_type = loss2_type.lower()
        if self.loss2_type == 'l1':
            self.sparse_loss = nn.L1Loss()
        else:
            raise NotImplementedError(f'{self.loss2_type} is not an implemented sparsity loss function.')
        
        self.penalty_scheduler = penalty_scheduler.lower()
        if self.penalty_scheduler == 'step':
            self.penalty_scheduler = StepAlpha(kwargs['penalty_gamma'])
        else:
            raise NotImplementedError(f'{self.penalty_scheduler} is not an implemented penalty scheduler function.')

        self.track_loss1 = []
        self.track_loss2 = []
        self.track_alpha = []


    def step(self) -> None:
        """
        Increase alpha using penalty scheduler
        """
        new_alpha = self.penalty_scheduler(self.alpha)
        self.alpha = new_alpha

    def forward(self, input:torch.Tensor, target:torch.Tensor, idx:int, e:int) -> None:
        """
        Calculate loss and add to log

        :param input: torch.Tensor, model output
        :param target: torch.Tensor, target matrix for comparison
        :return: calculated loss
        """
        self.track_alpha.append(self.alpha)
        loss1 = self.loss1(input, target)
        loss2 = self.loss2(input, target)

        self.track_alpha.append(self.alpha)
        self.track_loss1.append(loss1)
        self.track_loss2.append(loss2)
        self.track_batch.append(idx)

        return (1-self.alpha)*loss1 + self.alpha*loss2
    
    def get_log(self) -> Dict[str,List[float]]:
        """
        Return log of alpha values and loss1/loss2

        :return: Dictionary of tracked  loss values
        """
        return {'alpha':self.track_alpha, self.loss1_type:self.track_loss1, self.loss2_type:self.track_loss2}
    
    def clear_log(self) -> None:
        self.track_loss1 = []
        self.track_loss2 = []
        self.track_alpha = []