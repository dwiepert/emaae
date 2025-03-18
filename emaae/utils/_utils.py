"""
Some util functions for this package

Author(s): Daniela Wiepert
Last modified: 02/13/2025
"""
# IMPORTS
##built-in
from typing import List, Union
##third-party
import numpy as np
import torch
from scipy.signal import firwin

def fro_norm3d(mat:np.ndarray) -> float:
    """
    3D frobenius norm

    :param mat: np.ndarray, matrix
    :return: calculated frobenius norm
    """
    sq_norms = []
    for i in range(mat.shape[-1]):
        sq_norms.append(np.square(np.linalg.norm(mat[:,:,i], ord='fro')))

    return np.sqrt(np.sum(sq_norms))

def fro_norm3d_list(mat_list:List[np.ndarray]) -> float:
    """
    3D frobenius norm across a list

    :param mat_list: list of np.ndarray, matrix
    :return: calculated frobenius norm
    """

    sq_norms = []
    
    for mat in mat_list:
        sub_sq_norms = []
        for i in range(len(mat.shape[-1])):
            sub_sq_norms.append(np.square(np.linalg.norm(mat[:,:,i], ord='fro')))

        sq_norms.append(np.square(np.sum(sub_sq_norms)))
    
    return np.sqrt(np.sum(sq_norms))


def calc_sparsity(encoding:Union[np.ndarray, torch.tensor]):
    """
    Calculate proportion of 0s in an encoding
    :param encoding: numpy array of encoding
    :return sparsity: calculated sparsity
    """
    if isinstance(encoding, np.ndarray):
        zero_count = np.count_nonzero(encoding==0)
        sparsity = zero_count/np.size(encoding)
    else:
        te = torch.numel(encoding)
        nz = torch.count_nonzero(encoding)
        sparsity = (te-nz)/te
        sparsity = sparsity.item()
    return sparsity

def filter_encoding(batch_encoded:Union[np.ndarray, torch.tensor], f:np.ndarray=None, c:float=0.2, ntaps:int=51, to_torch:bool=True) -> List[float]:
    """
    """
    if f is None:
        f = firwin(numtaps=ntaps,cutoff=c)
    convolved_batch = []
    if not isinstance(batch_encoded, np.ndarray):
        batch_encoded = batch_encoded.cpu().numpy()
    for b in range(batch_encoded.shape[0]):
        encoded = np.squeeze(batch_encoded[b,:,:])
        convolved_signal = np.empty_like(encoded)
        for i in range(encoded.shape[0]):
            e = np.squeeze(encoded[i,:])
            convolved_signal[i,:] = np.convolve(e, f, mode='same')
        convolved_batch.append(convolved_signal)

    convolved_batch = np.stack(convolved_batch)
    
    if to_torch:
        convolved_batch = torch.from_numpy(convolved_batch)

    return convolved_batch

def get_filters(n_filters:int=20, ntaps:int=51):
    """
    """
    filters = []
    cutoffs = np.linspace((1/n_filters),1,n_filters, endpoint=False)
    for c in cutoffs:
        filters.append(firwin(numtaps=ntaps,cutoff=c))
    return filters, cutoffs
