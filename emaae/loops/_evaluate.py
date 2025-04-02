"""
Evaluate a trained model

Author(s): Daniela Wiepert
Last modified: 02/10/2025
"""
#IMPORTS
##built-in
import json
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
from emaae.utils import fro_norm3d, calc_sparsity, get_filters, filter_encoding, filter_matrix

def evaluate(test_loader:DataLoader, maxt:int, model:Union[CNNAutoEncoder], save_path:Union[str,Path], device, 
             encode:bool=True, decode:bool=True, n_filters:int=20, ntaps:int=51) -> Dict[str,List[float]]:
    """
    Evaluate mean squared error and sparsity (number of zero entries)

    :param test_loader: test dataloader object
    :param model: trained model
    :param save_path: str/Path, path to save metrics to
    :param device: torch device
    :param encode: bool, indicate whether to save encodings
    :param decode: bool, indicate whether to save decodings
    :return metrics: Dictionary of metrics
    """
    save_path = Path(save_path)
    mse = []
    filtered_mse = []
    baseline_filtered = []
    sparsity = []
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
        epath = save_path /'encodings'
        epath.mkdir(exist_ok=True)
    if decode:
        dpath = save_path /'decodings'
        dpath.mkdir(exist_ok=True)
    # EVALUATION LOOP    
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs= data['features'].to(device)
            encoded = model.encode(inputs)
            
            # SAVE ENCODINGS
            fname = data['files'][0]
            if encode:
                torch.save(encoded.cpu(),epath/f'{fname}.pt')

            outputs = model.decode(encoded)
            
            if decode:
                #print(fname)
                torch.save(outputs.cpu(),dpath/f'{fname}.pt')
                #print(f'saved to {dpath}')

            targets = np.squeeze(inputs.cpu().numpy())
            outputs = np.squeeze(outputs.cpu().numpy())
            mse.append(mean_squared_error(targets, outputs))

            # encoded = np.squeeze(encoded.cpu().numpy())
            # #encodings.append(encoded)
            # sparsity.append(calc_sparsity(encoded))

            ### PSD and low pass filtering 
            fm = sweep_filters(encoded=encoded, targets=targets, conv_matrices=conv_matrices, cutoffs=cutoffs, model=model, device=device)
            filtered_mse.append(fm)
            
            print('baseline filtering')
            bfm = sweep_filters(inputs, targets, conv_matrices, cutoffs, model=None, device=None)
            baseline_filtered.append(bfm)
            #frequencies, psd = welch(encoded, 50)
            #freqs.append(frequencies)
            #psd.append(psd)
    
    # SAVE METRICS
    filtered_mse = np.asarray(filtered_mse)
    baseline_filtered = np.asarray(baseline_filtered)
    avg_filtered_mse = np.mean(filtered_mse, axis=0)
    avg_baseline_filtered = np.mean(baseline_filtered, axis=0)
    metrics = {'mse': float(np.mean(mse)), 'weight_norms':norms, 'cutoffs':list(cutoffs), 'avg_filtered_mse': avg_filtered_mse.tolist(), 'filtered_mse': filtered_mse.tolist(), 'avg_baseline_filtered':avg_baseline_filtered.tolist(), 'baseline_filtered':baseline_filtered.tolist()}
    with open(str(save_path /'metrics.json'), 'w') as f:
        json.dump(metrics,f)

    return metrics

def sweep_filters(encoded:torch.tensor, targets:np.ndarray,conv_matrices:List[np.ndarray], cutoffs:List[float], model:CNNAutoEncoder, device, ntaps:int=51) -> List[float]:
    """"""
    #encoded = np.expand_dims(encoded, axis=0)
    mse = []
    for i in range(len(conv_matrices)):
        new_encoded = filter_encoding(encoded.to(device), device=device, f_matrix=conv_matrices[i].to(device), c=cutoffs[i], ntaps=ntaps)
        if model is not None:
            new_encoded = new_encoded.to(device)
            outputs = model.decode(new_encoded)
            outputs = np.squeeze(outputs.cpu().numpy())
        else: 
            outputs = np.squeeze(new_encoded.numpy())

        
        mse.append(mean_squared_error(targets, outputs))

    return np.asarray(mse)

