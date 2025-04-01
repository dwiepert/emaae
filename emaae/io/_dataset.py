"""
Custom EMA dataset

Author(s): Daniela Wiepert
Last modified: 01/10/2025
"""
#IMPORTS
##built-in
from pathlib import Path
from typing import Dict, Union, List
import warnings
##third party
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
##local
from ._load_feats import load_features

class ProcessEMA():
    """
    EMA processing transform - removes loudness information
    """
    def __init__(self):
        self.mask = np.ones(14, dtype=bool)
        self.mask[[12]] = False
    
    def __call__(self, sample:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Transform sample
        :param sample: dict, sample
        :return sample: dict, transformed sample
        """
        temp = sample['features']
        masked = temp[:, self.mask]
        norm_ema = np.empty((masked.shape), dtype=masked.dtype)
        for j in range(masked.shape[1]):
            jcol = masked[:,j]
            norm_ema[:,j] = 2 * np.divide((jcol - np.min(jcol)), (np.max(jcol)-np.min(jcol)))
        sample['features'] = norm_ema
        return sample
    
class ToTensor():
    """
    Convert sample features/times from numpy to tensors
    """
    def __call__(self, sample:Dict[str,np.ndarray]) -> Dict[str,torch.Tensor]:
        sample['features'] = torch.from_numpy(sample['features'])
        sample['times'] = torch.from_numpy(sample['times'])
        return sample

class EMADataset(Dataset):
    """
    Custom EMA Dataset
    :param root_dir: str/Path, root directory with feature files
    :param recursive: bool, boolean for whether to load features recursively (default = False)
    :param cci_features: cc interface (default = None)
    """
    def __init__(self, root_dir:Union[str,Path], recursive:bool=False, cci_features=None):
        super().__init__()
        self.root_dir = root_dir
        self.cci_features = cci_features
        self.recursive=recursive
        self._load_data()
        self.files = list(self.features.keys())
        self.transform = torchvision.transforms.Compose([ProcessEMA(), ToTensor()])

    def _load_data(self):
        """
        Load features
        """
        self.features, self.maxt = load_features(feature_dir=self.root_dir, cci_features=self.cci_features,
                          recursive=self.recursive,ignore_str='times')
        self.times,_ = load_features(feature_dir=self.root_dir, cci_features=self.cci_features,
                            recursive=self.recursive,search_str='times')
        #data = align_times(features, times)
    
    def __len__(self) -> int:
        """
        :return: int, length of data
        """
        return len(self.features)

    def __getitem__(self, idx:int) -> Dict[str,np.ndarray]:
        """
        Get item
        
        :param idx: int/List of ints/tensor of indices
        :return: dict, transformed sample
        """
        f = self.files[idx]
        sample = {'files':f, 'features': self.features[f], 'times':self.times[f]}

        return self.transform(sample)

def custom_collatefn(batch) -> torch.tensor:
    """
    Custom collate function to put batch together 

    :param batch: batch from DataLoader object
    :return: collated batch in proper format
    """
    warnings.filterwarnings("ignore")

    feat_list = []
    file_list = []
    max_t = 0
    for b in batch:
        f = torch.transpose(b['features'],0,1)
        if f.shape[-1] > max_t:
            max_t = f.shape[-1]
        feat_list.append(torch.transpose(b['features'],0,1))
        file_list.append(b['files'])

    for i in range(len(feat_list)):
        f = feat_list[i]
        if f.shape[-1] != max_t:
            new_f = torch.nn.functional.pad(f,(0,max_t-f.shape[-1]), mode="constant", value=0)
            feat_list[i] = new_f

    return {'features':torch.stack(feat_list, 0), 'files':file_list}
    

        
