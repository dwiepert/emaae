import torch
import torch.nn as nn
import argparse
from typing import Union,List,Tuple
import json
import os
from pathlib import Path
from tqdm import tqdm
##third-party
import cottoncandy as cc
import torch
from torch.utils.data import DataLoader
##local
from emaae.io import EMADataset, custom_collatefn
from emaae.models import CNNAutoEncoder
from emaae.loops import train, set_up_train, evaluate

#### FUNCTIONS
def _generate_sequence(function, params) -> nn.Sequential:
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


def _encoder_block(in_size:int, out_size:int, k:int, s:Union[int, Tuple], p:int) -> List[nn.Module]:
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
    return [nn.ConvTranspose1d(in_channels=in_size, out_channels=out_size, kernel_size=k, stride=s, padding=p)]
    
def _decoder_block(in_size:int, out_size:int, k:int, s:Union[int, Tuple], p:int) -> List[nn.Module]:
    """
    Generate a decoder block

    :param in_size: int, starting dimensions
    :param out_size: int, ending dimension,
    :param k: int, kernel size
    :param s: int/tuple, stride
    :param p: int, padding
    :return: list of layers
    """
    return [nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=k, stride=s, padding=p)]


def calculate_padding(l, stride, dilation, kernel_size, output_padding=0,transpose=False):
    if transpose:
        #https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
        padding = (l - ((l-1)*stride) - (dilation*(kernel_size-1)) - output_padding - 1) / -2
        return padding 

    else:
        #https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        padding = (((l -1)*stride) + 1 + (dilation*(kernel_size-1)) - l)/2
        return padding 
    
#### DEBUG CODE

input1 = './debug/'
debug_dataset = EMADataset(root_dir=input1, recursive=True)
debug_loader = DataLoader(debug_dataset, batch_size=1, shuffle=False,collate_fn=custom_collatefn)

encoder_params = {'in_size': [13,128, 256, 512],
                     'out_size': [128, 256, 512,1024],
                     'kernel_size':[8,2,2,2],
                     'stride': [10,2,2,2],
                     'padding':[1313,146,146,146]}
decoder_params =  {'in_size': [1024, 340, 168],
                      'out_size': [340, 168, 13],
                      'kernel_size':[5,5,3],
                      'stride': [3,2,13],
                      'padding':[293,148,1747]}


encoder, seq_encoder = _generate_sequence(_encoder_block, encoder_params)
decoder, seq_decoder = _generate_sequence(_decoder_block, decoder_params)


for data in tqdm(debug_loader):
    inputs = data[0]

    #check calculated_padding:
    test = inputs
    print('-----ENCODER-----')
    print(f'Initial shape: {test.shape}')
    error = False
    for i in range(len(encoder)):
        padding = calculate_padding(l=inputs.shape[-1], stride=encoder_params['stride'][i], kernel_size=encoder_params['kernel_size'][i], transpose=True, dilation=1, output_padding=0)
        print(f'Required padding: {padding}')
        ap = encoder_params['padding'][i]
        print(f'Actual padding: {ap}')

        t=encoder[i]
        try:
            test = t(test)
            print(f'Shape after encoder layer {i}: {test.shape}')
        except:
            print('Runtime Error occurred. Input size < 292.\n')
            error = True

    if not error:
        print('-----DECODER-----')
        for i in range(len(decoder)):
            padding = calculate_padding(l=inputs.shape[-1], stride=decoder_params['stride'][i], kernel_size=decoder_params['kernel_size'][i], transpose=False, dilation=1, output_padding=0)
            print(f'Required padding: {padding}')
            ap = decoder_params['padding'][i]
            print(f'Actual padding: {ap}')

            t = decoder[i]
            try:
                test = t(test)
                print(f'Shape after decoder layer {i}: {test.shape}\n')
            except:
                print('Error')


