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
from emaae.utils import fro_norm3d

def evaluate(test_loader:DataLoader, model:Union[CNNAutoEncoder], save_path:Union[str,Path], device) -> Dict[str,List[float]]:
    """
    Evaluate mean squared error and sparsity (number of zero entries)

    :param test_loader: test dataloader object
    :param model: trained model
    :param save_path: str/Path, path to save metrics to
    :param device: torch device
    :return metrics: Dictionary of metrics
    """
    save_path = Path(save_path)
    mse = []
    sparsity = []
    model.eval()

    # LOOK AT WEIGHTS
    weights = model.get_weights()
    norms = []
    for w in weights:
        norms.append(float(fro_norm3d(w.cpu().numpy())))

    encodings = []
    # EVALUATION LOOP    
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs= data.to(device)
            encoded = model.encode(inputs)
            outputs = model.decode(encoded)

            targets = np.squeeze(inputs.cpu().numpy())
            outputs = np.squeeze(outputs.cpu().numpy())
            mse.append(mean_squared_error(targets, outputs))
            
            encodings.append(encoded.cpu())
            encoded = np.squeeze(encoded.cpu().numpy())
            sparsity.append(np.count_nonzero(encoded==0))
    
    # SAVE METRICS
    metrics = {'mse': float(np.mean(mse)), 'sparsity':float(np.mean(sparsity)), 'weight_norms':norms}
    encodings = torch.stack(encodings, dim=0)
    torch.save(encodings, save_path/'encodings.pt')
    with open(str(save_path /'metrics.json'), 'w') as f:
        json.dump(metrics,f)

    return metrics


