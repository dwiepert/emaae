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
from emaae.utils import fro_norm3d, calc_sparsity, get_filters, filter_encoding

def evaluate(test_loader:DataLoader, model:Union[CNNAutoEncoder], save_path:Union[str,Path], device, 
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

    filters, cutoffs = get_filters(n_filters=n_filters, ntaps=ntaps)

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

            encoded = np.squeeze(encoded.cpu().numpy())
            #encodings.append(encoded)
            sparsity.append(calc_sparsity(encoded))

            ### PSD and low pass filtering 
            fm = sweep_filters(encoded, targets, filters, model, device)
            filtered_mse.append(fm)
            
            print('baseline filtering')
            bfm = sweep_filters(targets, targets, filters, model=None, device=None)
            baseline_filtered.append(bfm)
            #frequencies, psd = welch(encoded, 50)
            #freqs.append(frequencies)
            #psd.append(psd)
    
    # SAVE METRICS
    filtered_mse = np.asarray(filtered_mse)
    baseline_filtered = np.asarray(baseline_filtered)
    #print(filtered_mse.shape)
    avg_filtered_mse = np.mean(filtered_mse, axis=0)
    avg_baseline_filtered = np.mean(baseline_filtered, axis=0)
    #print(avg_filtered_mse.shape)
    metrics = {'mse': float(np.mean(mse)), 'sparsity':float(np.mean(sparsity)), 'weight_norms':norms, 'cutoffs':list(cutoffs), 'avg_filtered_mse': avg_filtered_mse.tolist(), 'filtered_mse': filtered_mse.tolist(), 'avg_baseline_filtered':avg_baseline_filtered.tolist(), 'baseline_filtered':baseline_filtered.tolist()}
    with open(str(save_path /'metrics.json'), 'w') as f:
        json.dump(metrics,f)

    return metrics

def sweep_filters(encoded:np.ndarray, targets:np.ndarray,filters:List[np.ndarray], model:CNNAutoEncoder, device) -> List[float]:
    """"""
    encoded = np.expand_dims(encoded, axis=0)
    mse = []
    for f in filters:
        new_encoded = filter_encoding(encoded, f=f)
        if model is not None:
            new_encoded = torch.from_numpy(new_encoded).to(torch.float).to(device)
            outputs = model.decode(new_encoded)
            outputs = np.squeeze(outputs.cpu().numpy())
        else: 
            outputs = np.squeeze(new_encoded)
        
        # print(f'filter: {f.shape}')
        # print(f'encoded: {encoded.shape}')
        #print(f'new_encoded: {new_encoded.shape}')
        # print(f'output: {outputs.shape}')
        # print(f'target: {targets.shape}')
        
        mse.append(mean_squared_error(targets, outputs))

    return np.asarray(mse)


#### scipy.signal.firwin - convole w your signals
### num taps (how long the filter is), cutoff friequency - in units of nyquist frequency, probably easier to ignore our sampling rate, 20 linearly spaced cutoffs between 0-1, this gives a filter 
### np.convolve with time series , filter, mode='same'
### 51 vs. 151, orange one sharper, longer filter starts to get edge artifacts , should be high enough (51)
### plt psd