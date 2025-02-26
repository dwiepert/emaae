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
from scipy.signal import firwin
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
##local
from emaae.models import CNNAutoEncoder
from emaae.utils import fro_norm3d, calc_sparsity

def evaluate(test_loader:DataLoader, model:Union[CNNAutoEncoder], save_path:Union[str,Path], device, encode:bool=True, decode:bool=True) -> Dict[str,List[float]]:
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
    sparsity = []
    model.eval()

    filters, cutoffs = get_filters()

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
            fm = sweep_filters(encoded, targets, model, device, filters)
            filtered_mse.append(fm)
            #frequencies, psd = welch(encoded, 50)
            #freqs.append(frequencies)
            #psd.append(psd)
    


    # SAVE METRICS
    filtered_mse = np.asarray(filtered_mse)
    print(filtered_mse.shape)
    avg_filtered_mse = np.mean(filtered_mse, dim=1)
    print(avg_filtered_mse.shape)
    metrics = {'mse': float(np.mean(mse)), 'sparsity':float(np.mean(sparsity)), 'weight_norms':norms, 'cutoffs':list(cutoffs), 'avg_filtered_mse': avg_filtered_mse.tolist(), 'filtered_mse': filtered_mse.tolist()}
    with open(str(save_path /'metrics.json'), 'w') as f:
        json.dump(metrics,f)

    return metrics


def get_filters():
    """
    """
    filters = []
    cutoffs = np.linspace((1/20),1,20, endpoint=False)
    for c in cutoffs:
        filters.append(firwin(numtaps=51,cutoff=c))
    return filters, cutoffs


def sweep_filters(encoded:np.ndarray, targets:np.ndarray, model:CNNAutoEncoder, device, filters:List[np.ndarray]) -> List[float]:
    """"""
    mse = []

    for f in filters:
        convolved_signal = np.empty_like(encoded)
        for i in range(encoded.shape[0]):
            e = np.squeeze(encoded[i,:])
            convolved_signal[i,:] = np.convolve(e, f, mode='same')
        
        convolved_signal = np.expand_dims(convolved_signal, 0)
        convolved_signal = torch.from_numpy(convolved_signal)
        convolved_signal = convolved_signal.to(device)
        outputs = model.decode(convolved_signal)
        outputs = np.squeeze(outputs.cpu().numpy())

        mse.append(mean_squared_error(targets, outputs))

    return np.asarray(mse)


#### scipy.signal.firwin - convole w your signals
### num taps (how long the filter is), cutoff friequency - in units of nyquist frequency, probably easier to ignore our sampling rate, 20 linearly spaced cutoffs between 0-1, this gives a filter 
### np.convolve with time series , filter, mode='same'
### 51 vs. 151, orange one sharper, longer filter starts to get edge artifacts , should be high enough (51)
### plt psd