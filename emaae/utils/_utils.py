"""
Some util functions for this package

Author(s): Daniela Wiepert
Last modified: 04/13/2025
"""
# IMPORTS
##built-in
from typing import List, Union, Tuple
##third-party
import numpy as np
import torch
from scipy.signal import firwin
from scipy.linalg import toeplitz

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

def filter_encoding(batch_encoded:torch.tensor, device=None, f_matrix:torch.tensor=None, c:float=0.2, ntaps:int=51, to_numpy:bool=False) -> Union[torch.tensor, np.ndarray]:
    """
    Filter an encoding

    :param batch_encoded: tensor, batch of encodings (B, E, T)
    :param device: torch device (default = None)
    :param f_matrix: matrix for convolution (default = None)
    :param c: float, filter cutoff for convolution (default = 0.2)
    :param ntaps: int, size of filter (default = 51)
    :param to_numpy: bool, True if returning a numpy array of the filtered encoding (default = False)
    :return filtered: torch tensor or numpy array of the filtered encoding (B, E, T)
    """
    t = batch_encoded.shape[-1]
    if f_matrix is None:
        f_matrix = filter_matrix(int(t), ntaps=ntaps, c=c)
    else:
        f_matrix = f_matrix[:t,:t]
    #batch_encoded.to(device)
    f_matrix = f_matrix.to(device)
    filtered = torch.matmul(batch_encoded, f_matrix)
    if to_numpy:
        return filtered.cpu().numpy()
    else:
        return filtered


def filter_matrix(t:int, f:np.ndarray=None, ntaps:int=51, c:float=0.2, to_torch:bool=True) -> Union[torch.tensor, np.ndarray]:
    """
    Create a filter matrix to do convolution quickly with matrix multiplications 
    :param t: int, number of time points
    :param f: numpy array, existing filter
    :param c: float, filter cutoff for convolution (default = 0.2)
    :param ntaps: int, size of filter (default = 51)
    :param to_torch: bool, True if returning filter matrix as a torch matrix (default = True)
    :return m: torch tensor/numpy array, filter matrix
    """

    if f is None:
        f = firwin(numtaps=ntaps,cutoff=c)
    center = int(ntaps/2)
    row1 = np.zeros((t), dtype=np.float32)
    col1 = np.zeros((t), dtype=np.float32)
    row1[:center+1] = f[center:]
    col1[:center+1] = np.flip(f[:center+1])
    m = toeplitz(c=col1, r=row1)

    if to_torch:
        return torch.from_numpy(m)
    else:
        return m
    
def get_filters(n_filters:int=20, ntaps:int=51) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get filters for an array of evenly spaced filters between 0-1
    :param n_filters: int, number of filters to make
    :param ntaps: int, size of filter (default = 51)
    :return filters: list of filters
    :param cutoffs: list of cutoff frequencies 
    """
    filters = []
    cutoffs = np.linspace((1/n_filters),1,n_filters, endpoint=False)
    for c in cutoffs:
        filters.append(firwin(numtaps=ntaps,cutoff=c))
    return filters, cutoffs
