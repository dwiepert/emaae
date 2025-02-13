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