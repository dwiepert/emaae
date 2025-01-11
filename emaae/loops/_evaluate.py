"""
Evaluate a trained model

Author(s): Daniela Wiepert
Last modified: 01/10/2025
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

def evaluate(test_loader:DataLoader, model:Union[CNNAutoEncoder], save_path:Union[str,Path]) -> Dict[str,List[float]]:
    """
    Evaluate mean squared error and sparsity (number of zero entries)

    :param test_loader: test dataloader
    :param model: trained model
    :param save_path: str/Path, path to save metrics to
    :return metrics: Dictionary of metrics
    """
    save_path = Path(save_path)
    mse = []
    sparsity = []
    model.eval()

    with torch.no_grad():
        for i,data in tqdm(test_loader):
            inputs, targets = data
            outputs = model(inputs)
            encoded = model.encode(inputs)

            targets = np.squeeze(targets.numpy())
            outputs = np.squeeze(outputs.numpy())
            mse.append(mean_squared_error(targets, outputs))
            
            encoded = np.squeeze(encoded.numpy())
            sparsity.append(np.count_nonzero(encoded==0))
    
    metrics = {'mse': mse, 'sparsity':sparsity}

    with open(str(save_path /'metrics.json'), 'w') as f:
        json.dumps(metrics)

    return metrics


