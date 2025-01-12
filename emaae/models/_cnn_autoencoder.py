"""
CNN Autoencoder

Author(s): Daniela Wiepert
Last modified: 01/11/2025
"""
#IMPORTS
##built-in
from typing import Tuple, Union, List
##third party
import numpy as np
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
    """
    def __init__(self, input_dim:int=13, n_encoder:int=5, n_decoder:int=5, inner_size:int=1024):
        super(CNNAutoEncoder, self).__init__()
        self.input_dim = input_dim
        if self.input_dim not in [13]:
            raise NotImplementedError(f'Model not compatible with {self.input_dim} dimensional features.')
        self.n_encoder = n_encoder
        if self.n_encoder not in [3,4,5]:
            raise NotImplementedError(f'Model not compatible with {self.n_encoder} encoder blocks.')
        self.n_decoder = n_decoder
        if self.n_decoder not in [3,4,5]:
            raise NotImplementedError(f'Model not compatible with {self.n_decoder} decoder blocks.')
        self.inner_size = inner_size
        if self.inner_size not in [768,1024]:
            raise NotImplementedError(f'Model not compatible with an inner dimension of {self.inner_size}.')

        self.encoder_params = self._encoder_block_options()
        self.decoder_params = self._decoder_block_options()
        self.teste,self.encoder = self._generate_sequence(function=self._encoder_block, params = self.encoder_params)
        self.testd,self.decoder = self._generate_sequence(function=self._decoder_block, params=self.decoder_params)

    def _encoder_block_options(self):
        """
        Present parameters for encoder blocks - add as necessary
        https://asiltureli.github.io/Convolution-Layer-Calculator/
        """
        ###1024 dim encoders
        if self.n_encoder == 5 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [self.input_dim, self.input_dim,128, 256, 512],
                      'out_size': [self.input_dim, 128, 256, 512,1024],
                      'kernel_size':[5,8,2,2,2],
                      'stride':[1,10,2,2,2],
                      'padding':[2,1313,146,146,146]}
        
        if self.n_encoder == 4 and self.inner_size==1024 and self.input_dim==13:
           return {'in_size': [self.input_dim,128, 256, 512],
                     'out_size': [128, 256, 512,1024],
                     'kernel_size':[8,2,2,2],
                     'stride': [10,2,2,2],
                     'padding':[1313,146,146,146]}
        
        if self.n_encoder == 3 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [self.input_dim, 128, 512],
                      'out_size': [128,512,1024],
                      'kernel_size':[8,4,2],
                      'stride': [10,4,2],
                      'padding':[1313,438,146]}
        
        ###768 dim encoders
        if self.n_encoder == 5 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [self.input_dim, self.input_dim,128, 256, 256],
                      'out_size': [self.input_dim, 128, 256, 256, 768],
                      'kernel_size':[5,8,2,3,3],
                      'stride': [1,10,2,1,3],
                      'padding':[2,1313,146,1,292]}
        
        if self.n_encoder == 4 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [self.input_dim,128, 256, 256],
                      'out_size': [128, 256, 256,768],
                      'stride': [11,2,1,3],
                      'kernel_size':[4,4,3,3],
                      'padding':[4,1,1,1]}
        
        if self.n_encoder == 3 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [self.input_dim, 128, 256],
                      'out_size': [128,256,768],
                      'stride': [11,2,3],
                      'kernel_size':[4,4,3],
                      'padding':[4,1,1]}
        
    def _calculate_padding(self, encode=True):
        if encode:
            return 
    
    def _decoder_block_options(self):
        """
        Present parameters for decoder blocks - add as necessary
        https://asiltureli.github.io/Convolution-Layer-Calculator/
        """
        ###1024 dim decoders
        if self.n_decoder == 5 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [1024, 1024, 512, 256, 128],
                      'out_size': [1024,512,256,128,self.input_dim], 
                      'kernel_size':[5,3,3,3,3],
                      'stride':[1,1,1,1,1],
                      'padding':[1,1,1,1,1]}
        if self.n_decoder == 4 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [1024, 512, 256, 128],
                      'out_size': [512, 256, 128, self.input_dim],
                      'stride': [2,2,2,10],
                      'kernel_size':[4,4,4,3],
                      'padding':[1,1,1,0]}
        
        if self.n_decoder == 3 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [1024, 340, 168],
                      'out_size': [340, 168, self.input_dim],
                      'kernel_size':[5,5,3],
                      'stride': [3,2,13],
                      'padding':[293,148,1747]}
        
        ###768 dim decoders
        if self.n_decoder == 5 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [768, 768, 256, 256, 128],
                      'out_size': [768,256,256,128,self.input_dim], 
                      'stride':[1,3,1,2,10],
                      'kernel_size':[5,5,3,4,3],
                      'padding':[2,2,1,1,0],
                      'dilation':[0,0,0,0,0]}
        if self.n_decoder == 4 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [768, 256, 256, 128],
                      'out_size': [256,256,128,self.input_dim], 
                      'stride':[3,1,2,10],
                      'kernel_size':[5,3,4,3],
                      'padding':[2,1,1,0]}
        if self.n_decoder == 3 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [768, 256, 128],
                      'out_size': [256,128,self.input_dim], 
                      'stride':[3,2,10],
                      'kernel_size':[5,4,3],
                      'padding':[3,1,0]}


    def _generate_sequence(self, function, params) -> nn.Sequential:
        """
        Generate a sequence of layers

        :param function: function for generating an encoder/decoder block
        :param n_blocks: int, number of blocks
        :param in_size: int, starting dimensions
        :param out_size: int, ending dimensions
        :return: nn.Sequential layers
        """
        sequence = []
    
        for n in range(len(params['in_size'])):
            sequence.extend(function(in_size=params['in_size'][n], out_size=params['out_size'][n], k=params['kernel_size'][n], s=params['stride'][n], p=params['padding'][n]))
        
        return sequence,nn.Sequential(*sequence)
        #return nn.Sequential(*sequence)


    def _encoder_block(self,in_size:int, out_size:int, k:int, s:Union[int, Tuple], p:int) -> List[nn.Module]:
        """
        Generate an encoder block

        :param in_size: int, starting dimensions
        :param out_size: int, ending dimension,
        :param k: int, kernel size
        :param s: int/tuple, stride
        :param p: int, padding
        :param d: int, dilation
        :return: list of layers
        """
        return [nn.ConvTranspose1d(in_channels=in_size, out_channels=out_size, kernel_size=k, stride=s, padding=p),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=out_size)]
    
    def _decoder_block(self, in_size:int, out_size:int, k:int, s:Union[int, Tuple], p:int) -> List[nn.Module]:
        """
        Generate a decoder block

        :param in_size: int, starting dimensions
        :param out_size: int, ending dimension,
        :param k: int, kernel size
        :param s: int/tuple, stride
        :param p: int, padding
        :return: list of layers
        """
        return [nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=k, stride=s, padding=p),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=out_size)]

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """
        Encode an input sequence

        :param x: tensor, input
        :return: tensor, encoded input
        """
        return self.encoder(x)
    
    def decode(self, x:torch.Tensor) -> torch.Tensor:
        """
        Decode an input sequence

        :param x: tensor, input
        :return: tensor, decoded input
        """
        return self.decoder(x)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward function of model

        :param x: tensor, input
        :return decoded: tensor, output
        """
        test = x
        for t in self.encoder:
            test = t(test)
            print(test.shape)
        encoded = self.encoder(x)
        test = encoded
        for t in self.decoder:
            test = t(test)
            print(test.shape)
        decoded = self.decoder(encoded)
        return decoded
        
    def get_type(self) -> str:
        """
        Return the model type
        :return: str, model type
        """
        return f'CNN_Autoencoder_ne{self.n_encoder}_nd{self.n_decoder}_innersz{self.inner_size}'


