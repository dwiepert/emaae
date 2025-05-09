"""
Evaluate a trained model

Author(s): Daniela Wiepert
Last modified: 04/13/2025
"""
#IMPORTS
##built-in
import json
import math
from pathlib import Path
from typing import Union, Dict, List
##third party
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
##local
from emaae.models import CNNAutoEncoder
from emaae.utils import fro_norm3d, filter_encoding, filter_matrix, add_white_noise

def evaluate(test_loader:DataLoader, maxt:int, model:Union[CNNAutoEncoder], save_path:Union[str,Path], device, 
             encode:bool=True, decode:bool=True, n_filters:int=20, ntaps:int=51) -> Dict[str,List[float]]:
    """
    Evaluate mean squared error and sparsity (number of zero entries)

    :param test_loader: test dataloader object
    :param maxt: int, max number of time points in the test dataset
    :param model: trained model
    :param save_path: str/Path, path to save metrics to
    :param device: torch device
    :param encode: bool, indicate whether to save encodings (default = True)
    :param decode: bool, indicate whether to save decodings (default = True)
    :param n_filters: int, number of filters to sweep for evaluation (default = 20)
    :param ntaps: int, filter size (default = 51)
    :return metrics: Dictionary of metrics
    """
    if encode:
        print('Saving encoding')
    save_path = Path(save_path)
    mse = []
    filtered_mse = []
    baseline_filtered = []
    noisy_filtered = []

    ###TODO - WHITENOISE FILTERED!!!!
    model.eval()

    cutoffs = np.linspace((1/n_filters),1,n_filters, endpoint=False)
    conv_matrices = []
    for i in range(len(cutoffs)):
        conv_matrices.append(filter_matrix(t=maxt, ntaps=ntaps, c=cutoffs[i]))

    # LOOK AT WEIGHTS
    weights = model.get_weights()
    norms = []
    for w in weights:
        norms.append(float(fro_norm3d(w.cpu().numpy())))
    
    if encode:
        print(save_path)
        epath = save_path /'encodings'
        epath.mkdir(exist_ok=True)
        print(epath)
    if decode:
        dpath = save_path /'decodings'
        dpath.mkdir(exist_ok=True)
    # EVALUATION LOOP    
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs= data['features'].to(device)
            encoded = model.encode(inputs)
            
            # SAVE ENCODINGS
            fname = Path(data['files'][0]).with_suffix('.npz').name
            #print(fname)
            if encode:
                np.savez(epath /fname ,encoded.cpu().numpy())
                #print('saved')
                #torch.save(encoded.cpu(), path)

            outputs = model.decode(encoded)
            
            # SAVE DECODINGS
            if decode:
                torch.save(outputs.cpu(),dpath/f'{fname}.pt')

            targets = np.squeeze(inputs.cpu().numpy())
            outputs = np.squeeze(outputs.cpu().numpy())
            mse.append(mean_squared_error(targets, outputs))

            ### PSD and low pass filtering 
            fm = sweep_filters(encoded=encoded, targets=targets, conv_matrices=conv_matrices, cutoffs=cutoffs, model=model, device=device)
            filtered_mse.append(fm)
            
            #print('baseline filtering')
            bfm = sweep_filters(inputs, targets, conv_matrices, cutoffs, model=None, device=device)
            baseline_filtered.append(bfm)

            #NOISY BASELINE
            noisy_inputs = add_white_noise(inputs, 0, 1, device=device)
            nfm = sweep_filters(noisy_inputs, targets, conv_matrices, cutoffs, model=None, device=device)
            #diff = abs(fm[0] - nfm[0])
            #if nfm[0] < fm[0]:
            noisy_filtered.append(nfm)


    # SAVE METRICS
    filtered_mse = np.asarray(filtered_mse)
    baseline_filtered = np.asarray(baseline_filtered)
    noisy_filtered = np.asarray(noisy_filtered)
    avg_filtered_mse = np.mean(filtered_mse, axis=0)
    avg_baseline_filtered = np.mean(baseline_filtered, axis=0)
    avg_noisy_filtered = np.mean(noisy_filtered, axis=0)
    metrics = {'mse': float(np.mean(mse)), 'weight_norms':norms, 'cutoffs':list(cutoffs), 'avg_filtered_mse': avg_filtered_mse.tolist(), 'filtered_mse': filtered_mse.tolist(), 'avg_baseline_filtered':avg_baseline_filtered.tolist(), 'baseline_filtered':baseline_filtered.tolist(), 'avg_noisy_filtered':avg_noisy_filtered.tolist(), 'noisy_filtered':noisy_filtered.tolist()}
    with open(str(save_path /'metrics.json'), 'w') as f:
        json.dump(metrics,f)

    return metrics

def sweep_filters(encoded:torch.tensor, targets:np.ndarray,conv_matrices:List[np.ndarray], cutoffs:List[float], 
                  model:CNNAutoEncoder, device, ntaps:int=51) -> List[float]:
    """
    Sweep filters over the encoding

    :param encoded: tensor, encoded input
    :param targets: np.ndarray, array of target values for reconstruction
    :param conv_matrices: list of numpy arrays, contains the matrices for filter convolution
    :param cutoffs: list of floats, list of all frequencies to sweep
    :param model: trained model
    :param device: torch device
    :param ntaps: int, filter size (default = 51)
    :return mse: np.ndarray of the mse for each filter
    """
    #encoded = np.expand_dims(encoded, axis=0)
    mse = []
    for i in range(len(conv_matrices)):
        new_encoded = filter_encoding(encoded.to(device), device=device, f_matrix=conv_matrices[i].to(device), c=cutoffs[i], ntaps=ntaps)
        if model is not None:
            new_encoded = new_encoded.to(device)
            outputs = model.decode(new_encoded)
            outputs = np.squeeze(outputs.cpu().numpy())
        else: 
            outputs = np.squeeze(new_encoded.cpu().numpy())

        
        mse.append(mean_squared_error(targets, outputs))

    return np.asarray(mse)

