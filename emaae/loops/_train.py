"""
Training loop

Author(s): Daniela Wiepert
Last modified: 01/10/2025
"""
#IMPORTS
##built-in
import json
import os
from pathlib import Path
from typing import Union
##third party
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
##local
from emaae.models import SparseLoss, CNNAutoEncoder


def set_up_train(model:Union[CNNAutoEncoder], optim_type:str='adamw', lr:float=0.0001, loss1_type:str='mse',
                 loss2_type:str='l1', penalty_scheduler:str='step', alpha:float=0.1, **kwargs) -> tuple[Union[torch.optim.AdamW, torch.optim.Adam], SparseLoss] :
    """
    Set up optimizer and and loss functions

    :param optim_type: str, type of optimizer to initialize (default = adamw)
    :param lr: float, learning rate (default = 0.0001)
    :param loss1_type: str, basic autoencoder loss type (default = mse)
    :param loss2_type: str, sparse loss type (default = l1)
    :param penalty_scheduler: str, type of penalty scheduler (default = step)
    :param alpha: float, alpha for loss function (default=0.1)
    :return optim: torch.optim, initialized optimizer
    :return loss_fn: initialized SparseLoss 
    """
    if optim_type=='adamw':
        optim = torch.optim.AdamW(params=model.parameters(),lr=lr)
    elif optim_type=='adam':
        optim = torch.optim.Adam(params=model.parameters,lr=lr)
    else:
        return NotImplementedError(f'{optim_type} not implemented.')

    loss_fn = SparseLoss(alpha=alpha, loss1_type=loss1_type, loss2_type=loss2_type, penalty_scheduler=penalty_scheduler, **kwargs)

    return optim, loss_fn

def train(train_loader:DataLoader, val_loader:DataLoader, model:Union[CNNAutoEncoder], optim, criterion, 
          epochs:int, save_path:Union[str, Path], device):
    """
    Train model

    :param train_loader: DataLoader, train dataloader
    :param val_loader: DataLoader, validation dataloader
    :param model: autoencoder model
    :param optim: torch.optim, initialized optimizer
    :param criterion: initialized SparseLoss 
    :param epochs: int, number of epochs to train for
    :param save_path: str/Path, path to save to
    :param device: torch device
    """
    os.makedirs(save_path, exist_ok=True)

    for e in range(epochs):
        print('EPOCH {}:'.format(e + 1))

        model.train(True)
        running_loss = 0.

        for data in tqdm(train_loader):
            inputs = data[0].to(device)
            optim.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()

            optim.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        log = criterion.get_log()
        log['avg_loss'] = avg_loss
        log['epoch'] = e

        with open(str(save_path / f'trainlog{e}.json'), 'w') as f:
            json.dump(log, f)

        criterion.clear_log()

        model.eval()
        running_vloss=0.0
        with torch.no_grad():
            for vdata in tqdm(val_loader):
                vinputs= vdata[0].to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vinputs)
                running_vloss += vloss.item()
        
        avg_vloss = running_vloss / len(val_loader)
        vlog = criterion.get_log()
        vlog['avg_loss'] = avg_vloss
        vlog['epoch'] = e
        
        with open(str(save_path / f'vallog{e}.json'), 'w') as f:
            json.dump(log, f)
        
        criterion.clear_log()

        criterion.step()
    
    return model

