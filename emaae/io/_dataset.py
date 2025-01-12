"""
Custom EMA dataset

Author(s): Daniela Wiepert
Last modified: 01/10/2025
"""
#IMPORTS
##built-in
from typing import Dict, Union, List
from pathlib import Path
##third party
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

##local
from ._load_feats import load_features

class ProcessEMA():
    """
    EMA processing transform
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
        sample['features'] = temp[:,self.mask]
        return sample
    
class ToTensor():
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
        self.features = load_features(feature_dir=self.root_dir, cci_features=self.cci_features,
                          recursive=self.recursive,ignore_str='times')
        self.times = load_features(feature_dir=self.root_dir, cci_features=self.cci_features,
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
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        
        #if not isinstance(idx,list):
        #    idx = [idx]
        
        f = self.files[idx]
        sample = {'features': self.features[f], 'times':self.times[f]}

        return self.transform(sample)

def custom_collatefn(batch):
    feat_list = []
    time_list = []

    for b in batch:
        feat_list.append(torch.transpose(b['features'],0,1))
        time_list.append(torch.transpose(b['times'],0,1))

    return torch.stack(feat_list, 0),torch.stack(feat_list,0)
    

        
