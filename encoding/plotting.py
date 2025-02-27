"""
"""

#IMPORTS
##built-in
import json
import re
import math
from pathlib import Path
from typing import Union
##third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import torch
##local
from emaae.io import EMADataset


#### plotting functions
def plot_logs(root:Union[str, Path], loss_type='tvl2', loss_label='Total Variation Loss (TVL2)'):
    """
    """
    root = Path(root)
    log_root = root / 'logs'
    save = root / 'plots'
    save = save / 'logs'
    save.mkdir(exist_ok=True)
    tf = log_root.glob('train*.json')
    vf = log_root.glob('val*.json')

    train_logs = pd.DataFrame(read_logs(tf))
    val_logs = pd.DataFrame(read_logs(vf))

    train_logs['type'] = 'train'
    val_logs['type'] = 'val'
    logs = pd.concat([train_logs, val_logs])

    ## COMBINED LOSS
    loss = logs[['epoch', 'avg_loss', 'type']]
    loss = loss.drop_duplicates()
    loss = loss.sort_values(by='epoch')
    #mean_loss = loss.groupby(['type']).mean()

    plt.style.use('ggplot')
    tl = loss[loss['type']=='train']
    vl = loss[loss['type']=='val']
    plt.plot(tl['epoch'].tolist(), tl['avg_loss'].tolist(),color='black', label='Train')
    plt.plot(vl['epoch'].tolist(), vl['avg_loss'].tolist(), color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Combined loss across epochs', loc='center')
    plt.savefig(str(save / 'combined_loss.png'), dpi=300)
    plt.clf()
    #plt.show()

    plt.plot(tl['epoch'].tolist()[1:], tl['avg_loss'].tolist()[1:],color='black', label='Train')
    plt.plot(vl['epoch'].tolist()[1:], vl['avg_loss'].tolist()[1:], color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Combined loss across epochs', loc='center')
    plt.savefig(str(save / 'combined_loss_e1.png'), dpi=300)
    plt.clf()

    ### MSE only
    loss = logs[['epoch', 'mse', 'type']]
    loss = loss.groupby(['epoch','type']).mean()
    #loss = loss.sort_values(by='epoch')
    loss = loss.reset_index()

    tl = loss[loss['type']=='train']
    vl = loss[loss['type']=='val']
    plt.plot(tl['epoch'].tolist(), tl['mse'].tolist(),color='black', label='Train')
    plt.plot(vl['epoch'].tolist(), vl['mse'].tolist(), color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE across epochs', loc='center')
    plt.savefig(str(save / 'mse.png'), dpi=300)
    plt.clf()
    #plt.show()

    plt.plot(tl['epoch'].tolist()[1:], tl['mse'].tolist()[1:],color='black', label='Train')
    plt.plot(vl['epoch'].tolist()[1:], vl['mse'].tolist()[1:], color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE across epochs', loc='center')
    plt.savefig(str(save / 'mse_e1.png'), dpi=300)
    plt.clf()

    ### Sparsity loss only
    loss = logs[['epoch', loss_type, 'type']]
    loss = loss.groupby(['epoch','type']).mean()
    #loss = loss.sort_values(by='epoch')
    loss = loss.reset_index()

    tl = loss[loss['type']=='train']
    vl = loss[loss['type']=='val']
    plt.plot(tl['epoch'].tolist(), tl[loss_type].tolist(),color='black', label='Train')
    plt.plot(vl['epoch'].tolist(), vl[loss_type].tolist(), color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(loss_label)
    plt.legend()
    plt.title(f'{loss_label} across epochs', loc='center')
    plt.savefig(str(save / f'{loss_type}.png'), dpi=300)
    plt.clf()
    #plt.show()

    plt.plot(tl['epoch'].tolist()[1:], tl[loss_type].tolist()[1:],color='black', label='Train')
    plt.plot(vl['epoch'].tolist()[1:], vl[loss_type].tolist()[1:], color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(loss_label)
    plt.legend()
    plt.title(f'{loss_label} across epochs', loc='center')
    plt.savefig(str(save / f'{loss_type}_e1.png'), dpi=300)
    plt.clf()

    ### Plot sparsity
    sparsity = logs[['epoch', 'sparsity', 'type']]
    sparsity = sparsity.groupby(['epoch','type']).mean()
    #loss = loss.sort_values(by='epoch')
    sparsity = sparsity.reset_index()
    
    tl = sparsity[sparsity['type']=='train']
    vl = sparsity[sparsity['type']=='val']
    plt.plot(tl['epoch'].tolist(), tl['sparsity'].tolist(),color='black', label='Train')
    plt.plot(vl['epoch'].tolist(), vl['sparsity'].tolist(), color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Sparsity (%)')
    plt.legend()
    plt.title(f'Sparsity across epochs', loc='center')
    plt.savefig(str(save / f'sparsity.png'), dpi=300)
    plt.clf()
    plt.close()
    

def read_logs(files):
    logs = {}
    for f in files:
        with open(str(f), "rb") as file:
            data = json.load(file)
        
        length = len(data['alpha'])

        for k in data.keys():
            d = data[k]
            if not isinstance(d, list):
                new_d = [d]*length
                d = new_d
            if k not in logs:
                logs[k] = d 
            else:
                temp = logs[k]
                temp.extend(d)
                logs[k] = temp
    
    return logs

def plot_reconstructions(reconstructed_root:Union[str,Path], original:Union[str,Path]):
    reconstructed_root = Path(reconstructed_root)
    decodings = reconstructed_root / 'decodings'
    rpaths = decodings.glob('*.pt')
    save_path = reconstructed_root / 'plots'
    save_path = save_path / 'reconstruction'
    save_path.mkdir(exist_ok=True)

    rfeats = {}
    for r in rpaths:
        name = r.name.replace('.pt','')
        rfeats[name] = np.transpose(np.squeeze(torch.load(r).numpy()))

    test_dataset = EMADataset(root_dir=original, recursive=True, cci_features=None)
    test_feats = test_dataset.features
    files = list(test_feats.keys())

    mask = np.ones(14, dtype=bool)
    mask[[12]] = False
    order = ['TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY','Pitch']
    colors = ['teal','teal', 'purple', 'purple', 'green', 'green', 'steelblue','steelblue', 'gold','gold', 'crimson', 'crimson', 'darkorange']
    rlist = np.arange(13)*2.1
    
    for i in range(len(files)):
        orig = test_feats[files[i]]
        orig = orig[:,mask]
        pred = rfeats[files[i]]

        #plt.style.use('ggplot')
        for j in range(orig.shape[1]):
            jcol = pred[:,j]
            norm_j = 2 * np.divide((jcol - np.min(jcol)), (np.max(jcol)-np.min(jcol)))
            norm_j += rlist[j]
            plt.plot(norm_j, color='dimgray')

            jcol = orig[:,j]
            norm_j = 2 * np.divide((jcol - np.min(jcol)), (np.max(jcol)-np.min(jcol)))
            norm_j += rlist[j]
            plt.plot(norm_j, color=colors[j])

        plt.yticks(rlist+1, order)
        plt.xlabel('Time')
        plt.title('Original vs. reconstructed EMA features')
        plt.savefig(str(save_path / f'reconstruction{i}'), dpi=300)
        plt.close()

        #print('pause')

def plot_activations(root, original):
    root = Path(root)
    enc_root = root / 'encodings'
    enc_files = enc_root.glob('*.pt')
    save_path = root / 'plots'


    k = []
    baselinek = []
    encodings = []
    encodings_flattened = []
    mask = np.ones(14, dtype=bool)
    mask[[12]] = False

    test_dataset = EMADataset(root_dir=original, recursive=True, cci_features=None)
    test_feats = test_dataset.features
    files = list(test_feats.keys())

    for f in enc_files:
        encoding = torch.load(f).numpy()
        enc_flattened = encoding.flatten()
        k.append(kurtosis(enc_flattened))
        encodings.append(encoding)
        encodings_flattened.append(enc_flattened)

        name = f.name.replace('.pt','')
        feat = test_feats[name]
        feat = feat[:,mask]
        padded = np.pad(feat, (0,(1024-13)), mode='constant', constant_values=0)
        baselinek.append(kurtosis(padded.flatten()))

    avg_k = np.mean(k)
    avg_bk = np.mean(baselinek)

    encodings_cat = np.concatenate(encodings_flattened)

    save_path = save_path / 'encodings'
    save_path.mkdir(exist_ok=True)

    plt.style.use('ggplot')
    plt.hist(encodings_cat, bins=100, log=True)
    plt.title('Log distribution of activations')
    plt.ylabel('Frequency')
    plt.xlabel('Activation')
    plt.text(0.6,10**8, f'Avg kurtosis: {round_sig(avg_k, 4)}')
    plt.text(0.6,10**7 + ((10**8 - 10**7)/3), f'Baseline: {round_sig(avg_bk,6)}')
    plt.savefig(str(save_path/'0loghist.png'),dpi=300)
    plt.clf()
    plt.close()


    for i in range(len(encodings)):
        plt.style.use('default')
        e = np.squeeze(encodings[i])
        plt.imshow(e)
        plt.ylabel('Encoding dimension')
        plt.xlabel('Time')
        plt.title('Encoding visualization')
        plt.savefig(str(save_path / f'activation{i}.png'),dpi=300)
        plt.clf()
        plt.close()

        plt.style.use('ggplot')
        for j in range(e.shape[0]):
            plt.plot(e[j,:])
        plt.xlabel('Time')
        plt.ylabel('Activation')
        plt.title('Activation over time across encoding dimensions')
        plt.savefig(str(save_path / f'activationvtime{i}.png'),dpi=300)
        plt.clf()
        plt.close()

def round_sig(x, digits):
    if x == 0:
        return 0
    
    logval = math.log10(abs(x))
    decimal_places = -int(logval) + (digits - 1)
    
    return round(x, decimal_places)

def plot_weights(root):
    pass

def plot_psd(root):
    root=Path(root)
    enc_root = root / 'encodings'
    enc_files = enc_root.glob('*.pt')
    save_path = root / 'plots'
    save_path = save_path / 'psd'
    save_path.mkdir(exist_ok=True)
    plt.style.use('ggplot')
    i = 0
    for f in enc_files:
        encoding = np.squeeze(torch.load(f).numpy())
        for j in range(encoding.shape[0]):
            plt.psd(encoding[j,:])
        plt.xlabel('Frequency')
        plt.ylabel('PSD (db)')
        plt.title('PSD', loc='center')
        plt.savefig(str(save_path/f'psd{i}.png'), dpi=300)
        i += 1
        plt.clf()


def plot_filtermse(root):
    root = Path(root)
    with open(str(root / 'metrics.json'), "rb") as file:
        metrics = json.load(file)
    
    fmse = np.asarray(metrics['filtered_mse'])
    cutoffs = metrics['cutoffs']

    fmse_df = pd.DataFrame(fmse, columns=cutoffs)
    fmse_df['encoding'] = fmse_df.index
    fmse_melted = pd.melt(fmse_df, id_vars=['encoding'], value_vars=cutoffs, var_name='cutoff', value_name='mse')
  
    plt.style.use('ggplot')
    for i in range(fmse.shape[0]):
        filtered = fmse_melted[fmse_melted['encoding'] == i]
        plt.plot(filtered['cutoff'].tolist(),filtered['mse'].tolist())



    



root = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e50bs16_adamw_mse_tvl2_a0.25_earlystop_bnf'
test_ema = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/librispeech/test/sparc'

#plot_logs(root)
#plot_reconstructions(root, test_ema)
#plot_psd(root)
#plot_activations(root, test_ema)
plot_filtermse(root)





    



