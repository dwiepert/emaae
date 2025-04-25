"""
CNN Autoencoder

Author(s): Daniela Wiepert
Last modified: 04/19/2025
"""
#IMPORTS
##built-in
from collections import OrderedDict
from typing import List, Dict
##third party
import torch
import torch.nn as nn
##local

class CNNAutoEncoder(nn.Module):
    """
    Convolutional neural net based autoencoder

    :param input_dim: int, size of input dimension (default = 13)
    :param n_encoder: int, number of encoder blocks (default = 5)
    :param n_decoder: int, number of decoder blocks (default = 5)
    :param inner_size: int, size of encoded representations (default = 1024)
    :param batchnorm_first: bool, True if batchnorm should come before CNN layer (default=True)
    :param final_tanh: bool, True if using tanh activation instead of no activation in the final layer of encoder/decoder (default=False)
    :param initial_ekernel: int, specify size of the first kernel in encoder (default = 5)
    :param initial_dkernel: int, specify size of the first kernel in decoder (default = 5)
    :param exclude_final_norm: bool, True if excluding normalization from final CNN block in encoder and decoder (default=False)
    :param exclude_all_norm: bool, True if excluding all normalization from the model (default=True)
    """
    def __init__(self, input_dim:int=13, n_encoder:int=2, n_decoder:int=2, inner_size:int=1024, batchnorm_first:bool=True, 
                 final_tanh:bool=False, initial_ekernel:int=5, initial_dkernel:int=5, exclude_final_norm:bool=False, exclude_all_norm:bool=False):
        super(CNNAutoEncoder, self).__init__()
        print(f'{n_encoder} encoder layers, {n_decoder} decoder layers, {inner_size} inner dims.')
        self.initial_ekernel=initial_ekernel
        self.initial_dkernel=initial_dkernel
        self.input_dim = input_dim
        if self.input_dim not in [13, 25, 50, 200, 768, 1408]:
            raise NotImplementedError(f'Model not compatible with {self.input_dim} dimensional features.')
        self.n_encoder = n_encoder
        if self.n_encoder not in [2,3,4,5]:
            raise NotImplementedError(f'Model not compatible with {self.n_encoder} encoder blocks.')
        self.n_decoder = n_decoder
        if self.n_decoder not in [2,3,4,5]:
            raise NotImplementedError(f'Model not compatible with {self.n_decoder} decoder blocks.')
        self.inner_size = inner_size
        if self.inner_size not in [512,768,1024,1408, 2048]:
            raise NotImplementedError(f'Model not compatible with an inner dimension of {self.inner_size}.')

        self.batchnorm_first = batchnorm_first
        self.final_tanh = final_tanh
        self.exclude_final_norm = exclude_final_norm
        self.exclude_all_norm = exclude_all_norm
        self.encoder_params = self._encoder_block_options()
        self.decoder_params = self._decoder_block_options()
        self.model = nn.Sequential(OrderedDict([('encoder',self._generate_sequence(params=self.encoder_params, exclude_final_norm=self.exclude_final_norm, batchnorm_first=self.batchnorm_first, final_tanh=False, exclude_all_norm=self.exclude_all_norm)), 
                                                ('decoder',self._generate_sequence(params=self.decoder_params, exclude_final_norm=True, batchnorm_first=self.batchnorm_first, final_tanh=self.final_tanh, exclude_all_norm=self.exclude_all_norm)) ]))

    def _encoder_block_options(self):
        """
        Present parameters for encoder blocks - add as necessary
        https://asiltureli.github.io/Convolution-Layer-Calculator/
        """
        ###1024 dim encoders
        if self.n_encoder == 3 and self.input_dim < 128 and self.inner_size > 512 and self.inner_size < 1024:
            return {'in_size': [self.input_dim, 128, 512],
                      'out_size': [128,512,self.inner_size],
                      'kernel_size':[self.initial_ekernel,3,3]} 
        elif self.n_encoder == 2 and self.input_dim <= 512 and self.inner_size > 512 and self.inner_size < 1024:
            return {'in_size': [self.input_dim, 512],
                      'out_size': [512,self.inner_size],
                      'kernel_size':[self.initial_ekernel,3]}
        elif self.n_encoder == 2 and self.input_dim <= 256 and self.inner_size <= 512 and self.inner_size >= 256:
            return {'in_size': [self.input_dim, 256],
                      'out_size': [256,self.inner_size],
                      'kernel_size':[self.initial_ekernel,3]}
        
        ### VIDEO FEATURES
        elif self.n_encoder == 3 and self.input_dim == 768 and self.inner_size == 768:
             return {'in_size': [self.input_dim, self.inner_size, self.inner_size],
                      'out_size': [self.inner_size,self.inner_size, self.inner_size],
                      'kernel_size':[self.initial_ekernel,3,3]}
        elif self.n_encoder == 3 and self.input_dim == 768 and self.inner_size == 2048:
             return {'in_size': [self.input_dim, 1280, 1536],
                      'out_size': [1280,1536, self.inner_size],
                      'kernel_size':[self.initial_ekernel,3,3]}
        elif self.n_encoder == 2 and self.input_dim == 768 and self.inner_size == 768:
             return {'in_size': [self.input_dim, self.inner_size],
                      'out_size': [self.inner_size, self.inner_size],
                      'kernel_size':[self.initial_ekernel,3]}
        elif self.n_encoder == 2 and self.input_dim == 768 and self.inner_size == 2048:
             return {'in_size': [self.input_dim,1536],
                      'out_size': [1536, self.inner_size],
                      'kernel_size':[self.initial_ekernel,3]}
        

        ### VIDEO FEATURES TAKE 2
        elif self.n_encoder == 3 and self.input_dim == 1408 and self.inner_size == 1408:
            return {'in_size': [self.input_dim, self.inner_size, self.inner_size],
                    'out_size': [self.inner_size,self.inner_size, self.inner_size],
                    'kernel_size':[self.initial_ekernel,3,3]}
        elif self.n_encoder == 2 and self.input_dim == 1408 and self.inner_size == 1408:
             return {'in_size': [self.input_dim, self.inner_size],
                      'out_size': [self.inner_size, self.inner_size],
                      'kernel_size':[self.initial_ekernel,3]}
        elif self.n_encoder == 2 and self.input_dim == 1408 and self.inner_size == 2048:
             return {'in_size': [self.input_dim,1700],
                      'out_size': [1700, self.inner_size],
                      'kernel_size':[self.initial_ekernel,3]}
        elif self.n_encoder == 2 and self.input_dim <=200 and self.inner_size == 1408:
             return {'in_size': [self.input_dim, 700],
                      'out_size': [700, self.inner_size],
                      'kernel_size':[self.initial_ekernel,3]}
        elif self.n_encoder == 3 and self.input_dim <=200 and self.inner_size == 1408:
            return {'in_size': [self.input_dim, 500, 1000],
                    'out_size': [500,1000, self.inner_size],
                    'kernel_size':[self.initial_ekernel,3,3]}
    
    def _decoder_block_options(self):
        """
        Present parameters for decoder blocks - add as necessary
        https://asiltureli.github.io/Convolution-Layer-Calculator/
        """
        ###1024 dim decoders
        
        if self.n_decoder == 3 and self.inner_size>=512 and self.input_dim<=128:
            return {'in_size': [self.inner_size, 512, 128],
                      'out_size': [512, 128, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3,3]}
        elif self.n_decoder == 2 and self.inner_size>512 and self.input_dim<=512:
            return {'in_size': [self.inner_size, 512],
                      'out_size': [512, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3]}
        elif self.n_decoder == 2 and self.inner_size<=512 and self.inner_size >= 256 and self.input_dim<=256:
            return {'in_size': [self.inner_size, 256],
                      'out_size': [256, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3]}
        
        ### VIDEO FEATURES
        elif self.n_decoder == 2 and self.input_dim == 768 and self.inner_size == 768:
            return {'in_size': [self.inner_size, self.inner_size],
                      'out_size': [self.inner_size, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3]}
        elif self.n_decoder == 3 and self.input_dim == 768 and self.inner_size == 768:
            return {'in_size': [self.inner_size, self.inner_size, self.inner_size],
                      'out_size': [self.inner_size, self.inner_size, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3, 3]}
        elif self.n_decoder == 2 and self.input_dim == 768 and self.inner_size == 2048:
            return {'in_size': [self.inner_size,1536],
                      'out_size': [1536, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3]}
        elif self.n_decoder == 3 and self.input_dim == 768 and self.inner_size == 1024:
            return {'in_size': [self.inner_size, 1536, 1280],
                      'out_size': [1536, 1280, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3, 3]}
        
        ### VIDEO FEATURES TAKE 2
        elif self.n_decoder == 2 and self.input_dim == 1408 and self.inner_size == 1408:
            return {'in_size': [self.inner_size, self.inner_size],
                      'out_size': [self.inner_size, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3]}
        elif self.n_decoder == 2 and self.input_dim <=200 and self.inner_size == 1408:
            return {'in_size': [self.inner_size, 700],
                      'out_size': [700, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3]}
        elif self.n_decoder == 3 and self.input_dim == 1408 and self.inner_size == 1408:
            return {'in_size': [self.inner_size, self.inner_size, self.inner_size],
                      'out_size': [self.inner_size, self.inner_size, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3, 3]}
        elif self.n_decoder == 3 and self.input_dim <= 200 and self.inner_size == 1408:
            return {'in_size': [self.inner_size, 1000, 500],
                      'out_size': [1000, 500, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3, 3]}
        elif self.n_decoder == 2 and self.input_dim == 1408 and self.inner_size == 2048:
            return {'in_size': [self.inner_size,1700],
                      'out_size': [1700, self.input_dim],
                      'kernel_size':[self.initial_dkernel,3]}

        
        
    def _generate_sequence(self, params:Dict[str, List[int]], exclude_final_norm:bool=False, 
                           batchnorm_first:bool=True, final_tanh:bool=False, exclude_all_norm:bool=False) -> nn.Sequential:
        """
        Generate a sequence of layers

        :param params: dictionary of model parameters
        :param batchnorm_first: bool, True if batchnorm should come before CNN layer (default=True)
        :param final_tanh: bool, True if using tanh activation instead of no activation in the final layer of encoder/decoder (default=False)
        :param exclude_final_norm: bool, True if excluding normalization from final CNN block in encoder and decoder (default=False)
        :param exclude_all_norm: bool, True if excluding all normalization from the model (default=True)
        :return: nn.Sequential of layers
        """
        sequence = OrderedDict()
    
        for n in range(len(params['in_size'])):
            block = OrderedDict()

            if (n < len(params['in_size']) - 1):
                if batchnorm_first and (not exclude_all_norm):
                    block['batchnorm'] = nn.BatchNorm1d(num_features=params['in_size'][n])
                
                block['conv'] = nn.Conv1d(in_channels=params['in_size'][n],out_channels=params['out_size'][n], kernel_size=params['kernel_size'][n], stride=1, padding="same")

                block['relu'] = nn.ReLU()
                if not batchnorm_first and (not exclude_all_norm):
                    block['batchnorm'] = nn.BatchNorm1d(num_features=params['out_size'][n])
            
            else:
                if batchnorm_first and (not exclude_final_norm) and (not exclude_all_norm):
                    block['batchnorm'] = nn.BatchNorm1d(num_features=params['in_size'][n])
                    #block['instancenorm'] = nn.InstanceNorm1d(num_features=params['out_size'][n])
                
                block['conv'] = nn.Conv1d(in_channels=params['in_size'][n],out_channels=params['out_size'][n], kernel_size=params['kernel_size'][n], stride=1, padding="same")
                
                if final_tanh:
                    block['tanh'] = nn.Tanh()

                if not batchnorm_first and (not exclude_final_norm) and (not exclude_all_norm):
                    block['batchnorm'] = nn.BatchNorm1d(num_features=params['out_size'][n])
                    #block['instancenorm'] = nn.InstanceNorm1d(num_features=params['out_size'][n])
                #block['batchnorm'] = nn.BatchNorm1d(num_features=params['out_size'][n])
              
            sequence[f'block{n}'] = nn.Sequential(block)
        print(sequence)
        return nn.Sequential(sequence)

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """
        Encode an input sequence

        :param x: tensor, input (batch_size, feature_dim, time)
        :return: tensor, encoded input (batch_size, encoding_dim, time)
        """
        return self.model.encoder(x)
    
    def decode(self, x:torch.Tensor) -> torch.Tensor:
        """
        Decode an input sequence

        :param x: tensor, input (batch_size, encoding_dim, time)
        :return: tensor, decoded input (batch_size, feature_dim, time)
        """
        return self.model.decoder(x)
    
    def get_type(self) -> str:
        """
        Return the model type
        :return: str, model type
        """
        return f'CNN_Autoencoder_ne{self.n_encoder}_nd{self.n_decoder}_innersz{self.inner_size}_iek{self.initial_ekernel}_idk{self.initial_dkernel}'
    
    def get_weights(self) -> List[torch.Tensor]:
        """
        :return weights: List of model weights for both encoder and decoder
        """
        weights = []

        for i in range(len(self.encoder_params["in_size"])):
            weights.append(self.model.encoder[i].conv.weight.detach())

        for i in range(len(self.decoder_params["in_size"])):
            weights.append(self.model.decoder[i].conv.weight.detach())

        return weights
        






