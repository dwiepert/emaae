"""
Custom loss function

Author(s): Daniela Wiepert
Last modified: 04/13/2025
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
from emaae.utils import fro_norm3d_list, calc_sparsity

class CustomLoss(nn.Module):
    """
    Custom loss for training autoencoder

    :param loss1_type: str, base autoencoder loss (default='mse')
    :param loss2_type: str, encoding loss (default=L1 Norm)
    :param alpha: float, loss weight (0-1, default = 0.25)
    :param weight_penalty: boolean, indicate whether weight penalty is being added to loss (default = False)
    :param penalty_scheduler: str, scheduler type for updating alpha (default='step')
    :param **kwargs: additional penalty scheduler parameters
    """
    def __init__(self, device, loss1_type:str='mse',loss2_type:str='l1',alpha:float=0.25,
                 weight_penalty:bool=False, penalty_scheduler:str='step', **kwargs):
        super(CustomLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.loss1_type = loss1_type.lower()
        if self.loss1_type == 'mse':
            self.loss1 = nn.MSELoss()
        else:
            raise NotImplementedError(f'{self.loss1_type} is not an implemented autoencoder loss function.')
        
        self.loss2_type = loss2_type.lower()
        if self.loss2_type == 'l1':
            self.loss2 = nn.L1Loss()
        elif self.loss2_type == 'tvl2':
            self.loss2 = self._tvl2
        elif self.loss2_type == 'filter':
            self.loss2 = None
        else:
            raise NotImplementedError(f'{self.loss2_type} is not an implemented sparsity loss function.')
        
        self.penalty_scheduler = penalty_scheduler.lower()
        if self.penalty_scheduler == 'step':
            self.penalty_scheduler = StepAlpha(alpha=self.alpha, epochs=kwargs['epochs'])
        else:
            raise NotImplementedError(f'{self.penalty_scheduler} is not an implemented penalty scheduler function.')

        self.track_loss1 = []
        self.track_loss2 = []
        self.track_sparsity = []
        self.track_alpha = []

        self.weight_penalty = weight_penalty
        if self.weight_penalty:
            self.track_weight= []

    def _tvl2(self, encoded:torch.Tensor, enc_target:torch.Tensor) -> float:
        """
        Total variation loss L2 (sum of squares of the gradient)
        :param encoded: tensor of the model encoding
        :param enc_target: tensor the model encoding shifted by 1 (encoded[:,:,:,1:]) (given to ensure same arguments for all loss functions)
        :return loss: float, calculated tvl2 loss
        """
        cut_encoded = encoded[:,:,:-1]
        #print(cut_encoded.shape)
        loss = torch.sum(torch.square(torch.sub(cut_encoded, enc_target)))

        return loss

    def step(self) -> None:
        """
        Increase alpha using penalty scheduler
        """
        self.penalty_scheduler.step()
        self.alpha = self.penalty_scheduler.get_alpha()

    def _weight_norm(self, weights: List[torch.Tensor]) -> float:
        """
        :param weights: list of torch.Tensors of all convolutional layer weights
        :return penalty: float, weight penalty
        """
        norm = fro_norm3d_list([w.numpy() for w in weights])
        penalty = np.square(norm-1)
        return penalty

    
    def forward(self, decoded:torch.Tensor, dec_target:torch.Tensor, 
                encoded:torch.Tensor, weights:Optional[List[torch.Tensor]]=None) -> float:
        """
        Calculate loss and add to log

        :param decoding: torch.Tensor, model output after decoding (batch_size, feature_dim, time)
        :param dec_target: torch.Tensor, target decoding
        :param encoded: torch.Tensor, model output after encoding (batch_size, encoding_dim, time)
        :param weights: optionally give list of torch.Tensors of all convolutional layer weights (default = None)
        :return total_loss: calculated loss
        """
        loss1 = self.loss1(decoded, dec_target)

        ## DEALING WITH LOSS2
        if self.loss2_type == 'l1':
            enc_target = torch.zeros(encoded.shape)
        elif self.loss2_type == 'tvl2':
            enc_target = encoded[:,:,1:]

        if self.loss2_type != 'filter':
            #print(enc_target.shape)
            #enc_target = torch.roll(encoded, shifts=1, dims=2)
            enc_target.to(self.device)

            loss2 = self.loss2(encoded, enc_target) 
            self.track_loss2.append(loss2.item())

            total_loss = (1-self.alpha)*loss1 + self.alpha*loss2
        else:
            total_loss = loss1
            self.track_loss2.append(None)

        self.track_alpha.append(self.alpha)
        self.track_loss1.append(loss1.item())
        self.track_sparsity.append(calc_sparsity(encoded))

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
        log = {'alpha':self.track_alpha, self.loss1_type:self.track_loss1, self.loss2_type:self.track_loss2, 'sparsity':self.track_sparsity}
        
        if self.weight_penalty:
            log['weight_penalty'] = self.track_weight
        return log
    
    def clear_log(self) -> None:
        """
        CLEAR THE LOG - used between training and validation to keep a training and validation log
        """
        self.track_loss1 = []
        self.track_loss2 = []
        self.track_alpha = []
        self.track_sparsity = []

        if self.weight_penalty:
            self.track_weight = []