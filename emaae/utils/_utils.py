import numpy as np
from typing import List

def fro_norm3d(mat:np.ndarray) -> float:
    """
    3D frobenius norm

    :param m]at: np.ndarray, matrix
    :retrn: calculated frobenius norm
    """
    sq_norms = []
    for i in range(mat.shape[-1]):
        sq_norms.append(np.square(np.linalg.norm(mat[:,:,i], ord='fro')))

    return np.sqrt(np.sum(sq_norms))

def fro_norm3d_list(mat_list:List[np.ndarray]) -> float:
    """
    3D frobenius norm across a list

    :param mat_list: list of np.ndarray, matrix
    :retrn: calculated frobenius norm
    """

    sq_norms = []
    
    for mat in mat_list:
        sub_sq_norms = []
        for i in range(len(mat.shape[-1])):
            sub_sq_norms.append(np.square(np.linalg.norm(mat[:,:,i], ord='fro')))

        sq_norms.append(np.square(np.sum(sub_sq_norms)))
    
    return np.sqrt(np.sum(sq_norms))

