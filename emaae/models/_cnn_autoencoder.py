"""
CNN Autoencoder

Author(s): Daniela Wiepert
Last modified: 01/10/2025
"""
#IMPORTS
##built-in
from typing import Tuple, Union, List
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
    :param inner_size: int, size of encoded representations
    """

    def __init__(self, input_dim:int=13, n_encoder:int=5, n_decoder:int=5, inner_size:int=1024):
        self.input_dim = input_dim
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.inner_size = inner_size

        self.encoder = self._generate_sequence(self, function=self._encoder_block, n_blocks=self.n_encoder, in_size=self.input_dim, out_size=self.inner_size)
        self.decoder = self._generate_sequence(self, function=self._decoder_block, n_blocks=self.n_decoder, in_size=self.inner_size, out_size=self.input_dim)

    def _calculate_parameters(self, n_blocks:int, in_size:int, out_size:int):
        """
        """
        return []

    def _generate_sequence(self, function, n_blocks:int, in_size:int, out_size:int) -> nn.Sequential:
        """
        Generate a sequence of layers

        :param function: function for generating an encoder/decoder block
        :param n_blocks: int, number of blocks
        :param in_size: int, starting dimensions
        :param out_size: int, ending dimensions
        :return: nn.Sequential layers
        """
        sequence = []
        params = self._calculate_parameters(n_blocks=n_blocks, in_size=in_size, out_size=out_size)

        for n in range(n_blocks):
            sequence.extend(function(in_size=params['in_size'][n], out_size=params['out_size'][n], k=params['k'][n], s=params['s'][n], p=params['p'][n]))
        
        return nn.Sequential(*sequence)


    def _encoder_block(self,in_size:int, out_size:int, k:int, s:Union[int, Tuple], p:int) -> List[nn.module]:
        """
        Generate an encoder block

        :param in_size: int, starting dimensions
        :param out_size: int, ending dimension,
        :param k: int, kernel size
        :param s: int/tuple, stride
        :param p: int, padding
        :return: list of layers
        """
        return [nn.ConvTranspose1d(in_channels=in_size, out_channels=out_size, kernel_size=k, stride=s, padding=p),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=out_size)]
    
    def _decoder_block(self, in_size:int, out_size:int, k:int, s:Union[int, Tuple], p:int) -> List[nn.module]:
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
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward function of model

        :param x: tensor, input
        :return decoded: tensor, output
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def get_type(self) -> str:
        """
        Return the model type
        :return: str, model type
        """
        return f'CNN_Autoencoder_ne{self.n_encoder}_nd{self.n_decoder}_innersz{self.inner_size}'


