"""
CNN Autoencoder

Author(s): Daniela Wiepert
Last modified: 02/10/2025
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
    """
    def __init__(self, input_dim:int=13, n_encoder:int=5, n_decoder:int=5, inner_size:int=1024, batchnorm_first:bool=True, 
                 final_tanh:bool=False, initial_ekernel:int=5, initial_dkernel:int=5, exclude_final_norm:bool=False):
        super(CNNAutoEncoder, self).__init__()
        print(f'{n_encoder} encoder layers, {n_decoder} decoder layers, {inner_size} inner dims.')
        self.initial_ekernel=initial_ekernel
        self.initial_dkernel=initial_dkernel
        self.input_dim = input_dim
        if self.input_dim not in [13]:
            raise NotImplementedError(f'Model not compatible with {self.input_dim} dimensional features.')
        self.n_encoder = n_encoder
        if self.n_encoder not in [2,3,4,5]:
            raise NotImplementedError(f'Model not compatible with {self.n_encoder} encoder blocks.')
        self.n_decoder = n_decoder
        if self.n_decoder not in [2,3,4,5]:
            raise NotImplementedError(f'Model not compatible with {self.n_decoder} decoder blocks.')
        self.inner_size = inner_size
        if self.inner_size not in [768,1024]:
            raise NotImplementedError(f'Model not compatible with an inner dimension of {self.inner_size}.')

        self.batchnorm_first = batchnorm_first
        self.final_tanh = final_tanh
        self.exclude_final_norm = exclude_final_norm
        self.encoder_params = self._encoder_block_options()
        self.decoder_params = self._decoder_block_options()
        self.model = nn.Sequential(OrderedDict([('encoder',self._generate_sequence(params=self.encoder_params, exclude_final_relu=True, exclude_final_norm=self.exclude_final_norm, batchnorm_first=self.batchnorm_first, final_tanh=False)), ('decoder',self._generate_sequence(params=self.decoder_params, exclude_final_relu=True, exclude_final_norm=True, batchnorm_first=self.batchnorm_first, final_tanh=self.final_tanh)) ]))

    def _encoder_block_options(self):
        """
        Present parameters for encoder blocks - add as necessary
        https://asiltureli.github.io/Convolution-Layer-Calculator/
        """
        ###1024 dim encoders
        if self.n_encoder == 5 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [self.input_dim, self.input_dim,128, 256, 512],
                      'out_size': [self.input_dim, 128, 256, 512,1024],
                      'kernel_size':[5,3,3,3,3]}
        
        if self.n_encoder == 4 and self.inner_size==1024 and self.input_dim==13:
           return {'in_size': [self.input_dim,128, 256, 512],
                     'out_size': [128, 256, 512,1024],
                     'kernel_size':[5,3,3,3]}
        
        if self.n_encoder == 3 and self.inner_size==1024 and self.input_dim==13 and self.initial_ekernel==11:
            return {'in_size': [self.input_dim, 128, 512],
                      'out_size': [128,512,1024],
                      'kernel_size':[11,3,3]} 
        
        if self.n_encoder == 3 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [self.input_dim, 128, 512],
                      'out_size': [128,512,1024],
                      'kernel_size':[5,3,3]} 
        
        # if self.n_encoder == 3 and self.inner_size==1024 and self.input_dim==13:
        #     return {'in_size': [self.input_dim, 128, 512],
        #               'out_size': [128,512,1024],
        #               'kernel_size':[5,2,2]}
        
        if self.n_encoder == 2 and self.inner_size==1024 and self.input_dim==13 and self.initial_ekernel==11:
            return {'in_size': [self.input_dim, 512],
                      'out_size': [512,1024],
                      'kernel_size':[11,3]}
        
        if self.n_encoder == 2 and self.inner_size==512 and self.input_dim==13:
            return {'in_size': [self.input_dim, 512],
                      'out_size': [256,512],
                      'kernel_size':[11,3]}
        
        # if self.n_encoder == 2 and self.inner_size==1024 and self.input_dim==13:
        #     return {'in_size': [self.input_dim, 512],
        #               'out_size': [512,1024],
        #               'kernel_size':[10,2]}
        
        ###768 dim encoders
        if self.n_encoder == 5 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [self.input_dim, self.input_dim,128, 256, 256],
                      'out_size': [self.input_dim, 128, 256, 256, 768],
                      'kernel_size':[5,3,3,3,3]}
        
        if self.n_encoder == 4 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [self.input_dim,128, 256, 256],
                      'out_size': [128, 256, 256,768],
                      'stride': [5,3,3,3]}
        
        if self.n_encoder == 3 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [self.input_dim, 128, 256],
                      'out_size': [128,256,768],
                      'stride': [5,3,3]}
    
    def _decoder_block_options(self):
        """
        Present parameters for decoder blocks - add as necessary
        https://asiltureli.github.io/Convolution-Layer-Calculator/
        """
        ###1024 dim decoders
        if self.n_decoder == 5 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [1024, 1024, 512, 256, 128],
                      'out_size': [1024,512,256,128,self.input_dim], 
                      'kernel_size':[5,3,3,3,3]}
        if self.n_decoder == 4 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [1024, 512, 256, 128],
                      'out_size': [512, 256, 128, self.input_dim],
                      'kernel_size':[5,3,3,3]}
        
        if self.n_decoder == 3 and self.inner_size==1024 and self.input_dim==13 and self.initial_dkernel==5:
            return {'in_size': [1024, 512, 128],
                      'out_size': [512, 128, self.input_dim],
                      'kernel_size':[5,3,3]}
        if self.n_decoder == 2 and self.inner_size==1024 and self.input_dim==13 and self.initial_dkernel==5:
            return {'in_size': [1024, 512],
                      'out_size': [512, self.input_dim],
                      'kernel_size':[5,3]}
        
        if self.n_decoder == 2 and self.inner_size==512 and self.input_dim==13 and self.initial_dkernel==5:
            return {'in_size': [512, 256],
                      'out_size': [256, self.input_dim],
                      'kernel_size':[5,3]}
        
        if self.n_decoder == 2 and self.inner_size==1024 and self.input_dim==13:
            return {'in_size': [1024, 512],
                      'out_size': [512, self.input_dim],
                      'kernel_size':[3,3]}
        
        ###768 dim decoders
        if self.n_decoder == 5 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [768, 768, 256, 256, 128],
                      'out_size': [768,256,256,128,self.input_dim],
                      'kernel_size':[5,3,3,3,3]}
        if self.n_decoder == 4 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [768, 256, 256, 128],
                      'out_size': [256,256,128,self.input_dim],
                      'kernel_size':[5,3,3,3]}
        if self.n_decoder == 3 and self.inner_size==768 and self.input_dim==13:
            return {'in_size': [768, 256, 128],
                      'out_size': [256,128,self.input_dim],
                      'kernel_size':[5,3,3]}


    def _generate_sequence(self, params:Dict[str, List[int]], exclude_final_relu:bool=False, exclude_final_norm:bool=False, batchnorm_first:bool=True, final_tanh:bool=False) -> nn.Sequential:
        """
        Generate a sequence of layers

        :param params: dictionary of model parameters
        :param exclude_final_relu: boolean, indicate whether to exclude the final relu layer
        :return: nn.Sequential layers
        """
        sequence = OrderedDict()
    
        for n in range(len(params['in_size'])):
            block = OrderedDict()
            block['conv'] = nn.Conv1d(in_channels=params['in_size'][n],out_channels=params['out_size'][n], kernel_size=params['kernel_size'][n], stride=1, padding="same")

            if (n == (len((params['in_size'])) - 1)) and exclude_final_relu and (not exclude_final_norm):
                block['instancenorm'] = nn.InstanceNorm1d(num_features=params['out_size'][n])
                #block['batchnorm'] = nn.BatchNorm1d(num_features=params['out_size'][n])
            
            if (n == (len((params['in_size'])) - 1)) and final_tanh:
                #case 1 - we are at the final layer and we want to have tanh - don't include batchnorm, just have tanh (ONLY FOR DECODER SO THAT'S WHY NO BATCHNORM)
                block['tanh'] = nn.Tanh()
            elif (n < len(params['in_size']) - 1) or not exclude_final_relu:
                #case 2 - we are either not at the final layer OR we aren't excluding final relu (that is, we're building the encoder)
                if batchnorm_first:
                    block['batchnorm'] = nn.BatchNorm1d(num_features=params['out_size'][n])

                block['relu'] = nn.ReLU()
                if not batchnorm_first:
                    block['batchnorm'] = nn.BatchNorm1d(num_features=params['out_size'][n])
            
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
    
    # def forward(self, x:torch.Tensor, debug:bool=False) -> torch.Tensor:
    #     """
    #     Forward function of model

    #     :param x: tensor, input
    #     :return decoded: tensor, output
    #     """
    #     if debug:
    #         print(f'Initial size: {x.shape}')
    #         encoded = self.model.encoder(x)
    #         print(f'Size after encoding: {encoded.shape}')
    #         decoded = self.model.decoder(encoded)
    #         print(f'Size after decoding: {decoded.shape}')
    #         return decoded
    #     else:
    #         return self.model(x)
        
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
        






