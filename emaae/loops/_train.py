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
import time
##third party
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
##local
from emaae.models import SparseLoss, CNNAutoEncoder, EarlyStopping

def set_up_train(model:Union[CNNAutoEncoder], optim_type:str='adamw', lr:float=0.0001, loss1_type:str='mse',
                 loss2_type:str='l1', penalty_scheduler:str='step', alpha:float=0.1, weight_penalty:bool=False, **kwargs) -> tuple[Union[torch.optim.AdamW, torch.optim.Adam], SparseLoss] :
    """
    Set up optimizer and and loss functions

    :param optim_type: str, type of optimizer to initialize (default = adamw)
    :param lr: float, learning rate (default = 0.0001)
    :param loss1_type: str, basic autoencoder loss type (default = mse)
    :param loss2_type: str, sparse loss type (default = l1)
    :param penalty_scheduler: str, type of penalty scheduler (default = step)
    :param alpha: float, alpha for loss function (default=0.1)
    :param weight_penalty: boolean, indicate whether weight penalty is being added to loss (default = False)
    :param **kwargs: any additional penalty scheduler parameters
    :return optim: torch.optim, initialized optimizer
    :return loss_fn: initialized SparseLoss 
    """
    if optim_type=='adamw':
        optim = torch.optim.AdamW(params=model.parameters(),lr=lr)
    elif optim_type=='adam':
        optim = torch.optim.Adam(params=model.parameters,lr=lr)
    else:
        return NotImplementedError(f'{optim_type} not implemented.')

    loss_fn = SparseLoss(alpha=alpha, loss1_type=loss1_type, loss2_type=loss2_type, penalty_scheduler=penalty_scheduler, weight_penalty=weight_penalty, **kwargs)

    return optim, loss_fn

def train(train_loader:DataLoader, val_loader:DataLoader, model:Union[CNNAutoEncoder], optim, criterion, 
          epochs:int, save_path:Union[str, Path], device, weight_penalty:bool=False, update:bool=False, debug:bool=False,
          alpha_epochs:int=15):
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
    :param weight_penalty: boolean, indicate whether weight penalty is being added to loss (default=False)
    :param debug: boolean, prints debugging statements (default=False)
    :param alpha_epochs: int, number of epochs to run alpha update for
    """
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping()
    start_time = time.time()
    alpha_update = False
    for e in range(epochs):
        print('EPOCH {}:'.format(e + 1))
        est = time.time()

        model.train(True)
        running_loss = 0.

        for data in tqdm(train_loader):
            inputs = data.to(device)
            optim.zero_grad()

            outputs = model(inputs, debug=debug)

            if weight_penalty:
                weights = model.get_weights()
                loss = criterion(outputs, inputs, weights)
            else:
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

        if e==0 or e % 5 == 0:
            eet = time.time()
            print(f'Average loss at Epoch {e}: {avg_loss}')
            print(f'Epoch {e} run time: {(eet-est)/60}')
            model.eval()
            running_vloss=0.0
            if weight_penalty:
                vweights = model.get_weights()
            with torch.no_grad():
                for vdata in tqdm(val_loader):
                    vinputs= vdata.to(device)
                    voutputs = model(vinputs, debug=debug)
                    if weight_penalty:
                        vloss = criterion(voutputs, vinputs, vweights)
                    else:
                        vloss = criterion(voutputs, vinputs)
                    running_vloss += vloss.item()
            
            avg_vloss = running_vloss / len(val_loader)
            vlog = criterion.get_log()
            vlog['avg_loss'] = avg_vloss
            vlog['epoch'] = e
            
            with open(str(save_path / f'vallog{e}.json'), 'w') as f:
                json.dump(vlog, f)
            
            criterion.clear_log()

            early_stopping(avg_vloss, model)
            if early_stopping.early_stop and not alpha_update:
                print("Early stopping. Switching to alpha update.") 
                torch.save(model, str(save_path / f'{model.get_type()}_best_model_lambda{criterion.alpha}.pth'))
                alpha_update=True
                new_epoch_counter = 0

            print(f'Average Validation Loss at Epoch {e}: {avg_vloss}')

        if alpha_update and update:
            print(f'Updating alpha....epoch {new_epoch_counter+1}')
            criterion.step()
            new_epoch_counter += 1
        
            if new_epoch_counter == alpha_epochs:
                break
    
    end_time = time.time()

    print(f'Model trained in {(end_time-start_time)/60} minutes.')
    return model, early_stopping.get_best_model()

