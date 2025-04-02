"""
Training loop

Author(s): Daniela Wiepert
Last modified: 02/10/2025
"""
#IMPORTS
##built-in
import json
import os
from pathlib import Path
from typing import Union
import time
##third party
from scipy.signal import firwin
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
##local
from emaae.models import SparseLoss, CNNAutoEncoder, EarlyStopping
from emaae.utils import calc_sparsity, filter_encoding, filter_matrix

def set_up_train(model:Union[CNNAutoEncoder], device, optim_type:str='adamw', lr:float=0.0001, 
                 loss1_type:str='mse', loss2_type:str='l1', alpha:float=0.1, weight_penalty:bool=False,
                 penalty_scheduler:str='step', lr_scheduler:bool=False, **kwargs) -> tuple[Union[torch.optim.AdamW, torch.optim.Adam], SparseLoss, Union[ExponentialLR, None]] :
    """
    Set up optimizer and and loss functions

    :param model: initialized model class (for getting parameters)
    :param device: device to send tensors to
    :param optim_type: str, type of optimizer to initialize (default = adamw)
    :param lr: float, learning rate (default = 0.0001)
    :param loss1_type: str, basic autoencoder loss type (default = mse)
    :param loss2_type: str, sparsity loss type (default = l1)
    :param alpha: float, alpha for loss function (default=0.1)
    :param weight_penalty: boolean, indicate whether weight penalty is being added to loss (default = False)
    :param penalty_scheduler: str, type of penalty scheduler (default = step)
    :param lr_scheduler: boolean, indicate whether to create a lr scheduler
    :param **kwargs: any additional penalty scheduler parameters
    :return optim: torch.optim, initialized optimizer
    :return loss_fn: initialized SparseLoss 
    :return scheduler: lr scheduler
    """

    # SET UP OPTIMIZER
    if optim_type=='adamw':
        optim = torch.optim.AdamW(params=model.parameters(),lr=lr)
    elif optim_type=='adam':
        optim = torch.optim.Adam(params=model.parameters(),lr=lr)
    else:
        return NotImplementedError(f'{optim_type} not implemented.')

    # SET UP LOSS
    loss_fn = SparseLoss(device=device, loss1_type=loss1_type, loss2_type=loss2_type, alpha=alpha,
                          weight_penalty=weight_penalty, penalty_scheduler=penalty_scheduler, **kwargs)

    # set up lr scheduler
    if lr_scheduler:
        end_lr = kwargs['end_lr']
        epochs = kwargs ['epochs']
        gamma = end_lr / (lr**(1/epochs))
        print(f'LR scheduler gamma: {gamma}')
        scheduler = ExponentialLR(optim, gamma=gamma)
    else:
        scheduler = None
    return optim, loss_fn, scheduler

def train(train_loader:DataLoader, val_loader:DataLoader, model:Union[CNNAutoEncoder], device, 
          optim:Union[torch.optim.AdamW, torch.optim.Adam], criterion:SparseLoss, lr_scheduler:ExponentialLR, save_path:Union[str, Path],
          epochs:int=500, alpha_epochs:int=15, update:bool=False, early_stop:bool=True, patience:int=5, weight_penalty:bool=False, 
          filter_loss:bool=False, maxt:int=None, filter_cutoff:float=0.2, ntaps:int=51) -> Union[CNNAutoEncoder]:
    """
    Train a model

    :param train_loader: DataLoader, train dataloader
    :param val_loader: DataLoader, validation dataloader
    :param model: initialized autoencoder model
    :param device: torch device
    :param optim: torch.optim, initialized optimizer
    :param criterion: initialized SparseLoss 
    :param lr_scheduler: lr scheduler
    :param save_path: str/Path, path to save logs and best model to
    :param epochs: int, number of epochs to train for
    :param alpha_epochs: int, number of epochs to run alpha update for
    :param update: boolean, indicate whether to allow alpha update
    :param early_stop: boolean, indicate whether to use early stopping
    :param patience: int, early stop patience
    :param weight_penalty: boolean, indicate whether weight penalty is being added to loss (default=False)
    :return model: trained autoencoder model
    
    """

    mpath = save_path / 'models'
    mpath.mkdir(exist_ok=True)
    lpath = save_path / 'logs'
    lpath.mkdir(exist_ok=True)

    early_stopping = EarlyStopping(patience=patience)
    start_time = time.time() #model start time
    alpha_update = False
    if update and not early_stop:
        alpha_update = True
        new_epoch_counter = 0

    if filter_loss:
        assert (maxt is not None)
        conv_matrix = filter_matrix(t=maxt, ntaps=ntaps, c=filter_cutoff)

    # START TRAINING
    for e in range(epochs):
        print('EPOCH {}:'.format(e + 1))
        est = time.time() #epoch start time

        model.train(True)
        running_loss = 0.
        
        # TRAINING LOOP
        for data in tqdm(train_loader):
            inputs = data['features'].to(device)
            optim.zero_grad()

            # ENCODE
            encoding = model.encode(inputs)

            if filter_loss:
                encoding = filter_encoding(encoding, device=device, f_matrix=conv_matrix, ntaps=ntaps, c=filter_cutoff).to(device)
                #encoding = encoding.to(device)
            # DECODE
            outputs = model.decode(encoding)
            
            # LOSS
            if weight_penalty:
                weights = model.get_weights()
            else:
                weights = None

            loss = criterion(decoded=outputs, dec_target=inputs, encoded=encoding, weights=weights)
            loss.backward()
            running_loss += loss.item()
        
            # UPDATE
            optim.step()

        # TRAINING LOG
        avg_loss = running_loss / len(train_loader)
        log = criterion.get_log()
        log['avg_loss'] = avg_loss
        log['epoch'] = e

        with open(str(lpath / f'trainlog{e}.json'), 'w') as f:
            json.dump(log, f)

        criterion.clear_log()


        # VALIDATION (every 5 epochs)
        #if e==0 or e % 1 == 0:
        eet = time.time() # epoch ending time
        print(f'Average loss at Epoch {e}: {avg_loss}')
        print(f'Epoch {e} run time: {(eet-est)/60}')

        model.eval()
        running_vloss=0.0
        if weight_penalty:
            vweights = model.get_weights()
        else:
            vweights = None

        # VALIDATION LOOP
        with torch.no_grad():
            for vdata in tqdm(val_loader):
                vinputs= vdata['features'].to(device)

                # ENCODE
                vencoding = model.encode(vinputs)

                if filter_loss:
                    vencoding = filter_encoding(vencoding, device=device, f_matrix=conv_matrix, c=filter_cutoff, ntaps=ntaps).to(device)
                    #vencoding = vencoding.to(device)
                # DECODE 
                voutputs = model.decode(vencoding)

                # LOSS
                vloss = criterion(decoded=voutputs, dec_target=vinputs, encoded=vencoding, weights=vweights)
                running_vloss += vloss.item()
            
        # VALIDATION LOG
        avg_vloss = running_vloss / len(val_loader)
        vlog = criterion.get_log()
        vlog['avg_loss'] = avg_vloss
        vlog['epoch'] = e
        
        with open(str(lpath / f'vallog{e}.json'), 'w') as f:
            json.dump(vlog, f)
        
        criterion.clear_log()


        # EARLY STOPPING
        if early_stop:
            early_stopping(avg_vloss, model, e)
            if early_stopping.early_stop and not alpha_update:
                print("Early stopping. Switching to alpha update.") 
                best_model, best_epoch, best_score = early_stopping.get_best_model()
                torch.save(best_model.state_dict(), str(mpath / f'{best_model.get_type()}_bestmodel_a{criterion.alpha}e{best_epoch}.pth'))
                if update:
                    alpha_update=True
                    new_epoch_counter = 0
                else:
                    break

        print(f'Average Validation Loss at Epoch {e}: {avg_vloss}')

        if (e == 0) or ((e+1) % 5 == 0):
            print('Saving epoch...')
            path = mpath / f'{model.get_type()}_e{e+1}.pth'
            torch.save(model.state_dict(), str(path))
        # CHANGE ALPHA
        # CASE 1: if model has stopped training early and update is True, will run for alpha_epochs
        # CASE 2: no early stopping, update is True, will run for epochs (ideally only do this if model is already trained)
        if alpha_update and update:
            print(f'Updating alpha....epoch {new_epoch_counter+1}')
            criterion.step()
            new_epoch_counter += 1
            if new_epoch_counter == alpha_epochs:
                break
        
        if lr_scheduler is not None:
            lr_scheduler.step()
    
    end_time = time.time() #model training end time

    if early_stop:
        #GET BEST MODEL AGAIN
        best_model,best_epoch, best_score = early_stopping.get_best_model()
        print(f'Best epoch: {best_epoch}')
        print(f'Best score: {best_score}')
        path = mpath / f'{best_model.get_type()}_bestmodel_e{best_epoch+1}.pth'
        torch.save(best_model.state_dict(), str(path))
        return best_model

    print(f'Model trained in {(end_time-start_time)/60} minutes.')
    return model

