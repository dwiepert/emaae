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
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
##local
from emaae.models import SparseLoss, CNNAutoEncoder, EarlyStopping

def set_up_train(model:Union[CNNAutoEncoder], optim_type:str='adamw', lr:float=0.0001, 
                 loss1_type:str='mse', loss2_type:str='l1', alpha:float=0.1, weight_penalty:bool=False,
                 penalty_scheduler:str='step', **kwargs) -> tuple[Union[torch.optim.AdamW, torch.optim.Adam], SparseLoss] :
    """
    Set up optimizer and and loss functions

    :param model: initialized model class (for getting parameters)
    :param optim_type: str, type of optimizer to initialize (default = adamw)
    :param lr: float, learning rate (default = 0.0001)
    :param loss1_type: str, basic autoencoder loss type (default = mse)
    :param loss2_type: str, sparsity loss type (default = l1)
    :param alpha: float, alpha for loss function (default=0.1)
    :param weight_penalty: boolean, indicate whether weight penalty is being added to loss (default = False)
    :param penalty_scheduler: str, type of penalty scheduler (default = step)
    :param **kwargs: any additional penalty scheduler parameters
    :return optim: torch.optim, initialized optimizer
    :return loss_fn: initialized SparseLoss 
    """

    # SET UP OPTIMIZER
    if optim_type=='adamw':
        optim = torch.optim.AdamW(params=model.parameters(),lr=lr)
    elif optim_type=='adam':
        optim = torch.optim.Adam(params=model.parameters(),lr=lr)
    else:
        return NotImplementedError(f'{optim_type} not implemented.')

    # SET UP LOSS
    loss_fn = SparseLoss(loss1_type=loss1_type, loss2_type=loss2_type, alpha=alpha,
                          weight_penalty=weight_penalty, penalty_scheduler=penalty_scheduler, **kwargs)

    return optim, loss_fn

def train(train_loader:DataLoader, val_loader:DataLoader, model:Union[CNNAutoEncoder], device, 
          optim:Union[torch.optim.AdamW, torch.optim.Adam], criterion:SparseLoss, save_path:Union[str, Path],
          epochs:int=500, alpha_epochs:int=15, update:bool=False, early_stop:bool=True, patience:int=5, weight_penalty:bool=False) -> Union[CNNAutoEncoder]:
    """
    Train a model

    :param train_loader: DataLoader, train dataloader
    :param val_loader: DataLoader, validation dataloader
    :param model: initialized autoencoder model
    :param device: torch device
    :param optim: torch.optim, initialized optimizer
    :param criterion: initialized SparseLoss 
    :param save_path: str/Path, path to save logs and best model to
    :param epochs: int, number of epochs to train for
    :param alpha_epochs: int, number of epochs to run alpha update for
    :param update: boolean, indicate whether to allow alpha update
    :param weight_penalty: boolean, indicate whether weight penalty is being added to loss (default=False)
    :return model: trained autoencoder model
    """

    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(patience=patience)
    start_time = time.time() #model start time
    alpha_update = False
    if update and not early_stop:
        alpha_update = True
        new_epoch_counter = 0

    # START TRAINING
    for e in range(epochs):
        print('EPOCH {}:'.format(e + 1))
        est = time.time() #epoch start time

        model.train(True)
        running_loss = 0.
        
        # TRAINING LOOP
        for data in tqdm(train_loader):
            inputs = data.to(device)
            optim.zero_grad()

            # ENCODE
            encoding = model.encode(inputs)
            enc_target = torch.zeros(encoding.shape)
            enc_target = enc_target.to(device)

            # DECODE
            outputs = model.decode(encoding)
            
            # LOSS
            if weight_penalty:
                weights = model.get_weights()
            else:
                weights = None

            loss = criterion(decoded=outputs, dec_target=inputs, encoded=encoding, enc_target=enc_target,weights=weights)
            loss.backward()
            running_loss += loss.item()

            # UPDATE
            optim.step()

        # TRAINING LOG
        avg_loss = running_loss / len(train_loader)
        log = criterion.get_log()
        log['avg_loss'] = avg_loss
        log['epoch'] = e

        with open(str(save_path / f'trainlog{e}.json'), 'w') as f:
            json.dump(log, f)

        criterion.clear_log()


        # VALIDATION (every 5 epochs)
        if e==0 or e % 1 == 0:
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
                    vinputs= vdata.to(device)

                    # ENCODE
                    vencoding = model.encode(vinputs)
                    venc_target = torch.zeros(vencoding.shape)
                    venc_target = venc_target.to(device)

                    # DECODE 
                    voutputs = model.decode(vencoding)

                    # LOSS
                    vloss = criterion(decoded=voutputs, dec_target=vinputs, encoded=vencoding,enc_target=venc_target, weights=vweights)
                    running_vloss += vloss.item()
            
            # VALIDATION LOG
            avg_vloss = running_vloss / len(val_loader)
            vlog = criterion.get_log()
            vlog['avg_loss'] = avg_vloss
            vlog['epoch'] = e
            
            with open(str(save_path / f'vallog{e}.json'), 'w') as f:
                json.dump(vlog, f)
            
            criterion.clear_log()


            # EARLY STOPPING
            if early_stop:
                early_stopping(avg_vloss, model, e)
                if early_stopping.early_stop and not alpha_update:
                    print("Early stopping. Switching to alpha update.") 
                    best_model,best_epoch = early_stopping.get_best_model()
                    torch.save(best_model.state_dict(), str(save_path / f'{best_model.get_type()}_bestmodel_a{criterion.alpha}e{best_epoch}.pth'))
                    alpha_update=True
                    new_epoch_counter = 0

            print(f'Average Validation Loss at Epoch {e}: {avg_vloss}')

        # CHANGE ALPHA
        # CASE 1: if model has stopped training early and update is True, will run for alpha_epochs
        # CASE 2: no early stopping, update is True, will run for epochs (ideally only do this if model is already trained)
        if alpha_update and update:
            print(f'Updating alpha....epoch {new_epoch_counter+1}')
            criterion.step()
            new_epoch_counter += 1
            if new_epoch_counter == alpha_epochs:
                break
    
    end_time = time.time() #model training end time

    if early_stopping:
        #GET BEST MODEL AGAIN
        path = str(save_path / f'{best_model.get_type()}_bestmodel_a{criterion.alpha}e{best_epoch}.pth')
        if not path.exists():
            best_model,best_epoch = early_stopping.get_best_model()
            torch.save(best_model.state_dict(), str(save_path / f'{best_model.get_type()}_bestmodel_a{criterion.alpha}e{best_epoch}.pth'))

    print(f'Model trained in {(end_time-start_time)/60} minutes.')
    return model

