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
    save.mkdir(exist_ok=True)
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

    ### LOG
    plt.style.use('ggplot')
    tl = loss[loss['type']=='train']
    vl = loss[loss['type']=='val']
    plt.plot(tl['epoch'].tolist(), [np.log(i) for i in tl['avg_loss'].tolist()],color='black', label='Train')
    plt.plot(vl['epoch'].tolist(),[np.log(i) for i in vl['avg_loss'].tolist()], color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log)')
    plt.legend()
    plt.title('Combined loss across epochs (log)', loc='center')
    plt.savefig(str(save / 'combined_loss_log.png'), dpi=300)
    plt.clf()
    #plt.show()

    plt.plot(tl['epoch'].tolist()[1:], [np.log(i) for i in tl['avg_loss'].tolist()][1:],color='black', label='Train')
    plt.plot(vl['epoch'].tolist()[1:], [np.log(i) for i in vl['avg_loss'].tolist()][1:], color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log)')
    plt.legend()
    plt.title('Combined loss across epochs (log)', loc='center')
    plt.savefig(str(save / 'combined_loss_e1_log.png'), dpi=300)
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
    plt.title('MSE across epochs ', loc='center')
    plt.savefig(str(save / 'mse.png'), dpi=300)
    plt.clf()

    tl = loss[loss['type']=='train']
    vl = loss[loss['type']=='val']
    plt.plot(tl['epoch'].tolist(), [np.log(i) for i in tl['mse'].tolist()],color='black', label='Train')
    plt.plot(vl['epoch'].tolist(), [np.log(i) for i in vl['mse'].tolist()], color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Log MSE')
    plt.legend()
    plt.title('Log MSE across epochs ', loc='center')
    plt.savefig(str(save / 'log_mse.png'), dpi=300)
    plt.clf()
    #plt.show()

    plt.plot(tl['epoch'].tolist()[50:], tl['mse'].tolist()[50:],color='black', label='Train')
    plt.plot(vl['epoch'].tolist()[50:], vl['mse'].tolist()[50:], color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE across epochs', loc='center')
    plt.savefig(str(save / 'mse_e50.png'), dpi=300)
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
    plt.ylabel(f'{loss_label}')
    plt.legend()
    plt.title(f'{loss_label} across epochs', loc='center')
    plt.savefig(str(save / f'{loss_type}.png'), dpi=300)
    plt.clf()

    tl = loss[loss['type']=='train']
    vl = loss[loss['type']=='val']
    plt.plot(tl['epoch'].tolist(), [np.log(i) for i in tl[loss_type].tolist()],color='black', label='Train')
    plt.plot(vl['epoch'].tolist(), [np.log(i) for i in vl[loss_type].tolist()], color='crimson', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(f'Log {loss_label}')
    plt.legend()
    plt.title(f'Log {loss_label} across epochs', loc='center')
    plt.savefig(str(save / f'log_{loss_type}.png'), dpi=300)
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
    plt.style.use('default')
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
    d = []
    for i in range(len(files)):
        if i < 10:
            d.append(files[i])
            orig = test_feats[files[i]]
            orig = orig[:,mask]
            pred = rfeats[files[i]]

            #plt.style.use('ggplot')
            for j in range(orig.shape[1]):
                jcol = orig[:,j]
                norm_j = 2 * np.divide((jcol - np.min(jcol)), (np.max(jcol)-np.min(jcol)))
                norm_j += rlist[j]
                if j == 0:
                    plt.plot(norm_j, color='dimgray', label='Original')
                else:
                    plt.plot(norm_j, color='dimgray')
                

                jcol = pred[:,j]
                norm_j = 2 * np.divide((jcol - np.min(jcol)), (np.max(jcol)-np.min(jcol)))
                norm_j += rlist[j]
                plt.plot(norm_j, color=colors[j])

            plt.yticks(rlist+1, order)
            plt.legend()
            plt.xlabel('Time')
            plt.title('Original vs. reconstructed EMA features')
            plt.savefig(str(save_path / f'reconstruction{i}'), dpi=300)
            plt.close()
    return d
        #print('pause')

def plot_activations(root, original, upper_text=10**8, lower_text=10**7, x_text=0.1, flist=[]):
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

    rfeats = {}
    for r in enc_files:
        name = r.name.replace('.pt','')
        rfeats[name] = np.transpose(np.squeeze(torch.load(r).numpy()))

    if flist == []:
        files = list(test_feats.keys())
    else:
        files = flist

    for f in list(rfeats.keys()):
        encoding = rfeats[f]
        enc_flattened = encoding.flatten()
        k.append(kurtosis(enc_flattened))
        encodings.append(encoding)
        encodings_flattened.append(enc_flattened)

        feat = test_feats[f]
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
    plt.text(x_text,upper_text, f'Avg kurtosis: {round_sig(avg_k, 4)}')
    plt.text(x_text,lower_text + ((upper_text- lower_text)/3), f'Baseline: {round_sig(avg_bk,6)}')
    plt.savefig(str(save_path/'0loghist.png'),dpi=300)
    plt.clf()
    plt.close()

    i = 0
    for f in files:
        if i < 10:
            new_save = save_path / 'imshow'
            new_save.mkdir(exist_ok=True)
            plt.style.use('default')
            e = np.transpose(np.squeeze(rfeats[f]))
            plt.imshow(e)
            plt.ylabel('Encoding dimension')
            plt.xlabel('Time')
            plt.title('Encoding visualization')
            plt.savefig(str(new_save / f'activation{i}.png'),dpi=300)
            plt.clf()
            plt.close()

            new_save = save_path / 'spy'
            new_save.mkdir(exist_ok=True)
            plt.style.use('default')
            plt.spy(e)
            plt.ylabel('Encoding dimension')
            plt.xlabel('Time')
            plt.title('Encoding visualization')
            plt.savefig(str(new_save / f'activation{i}.png'),dpi=300)
            plt.clf()
            plt.close()

            new_save = save_path / 'time'
            new_save.mkdir(exist_ok=True)
            plt.style.use('ggplot')
            for j in range(e.shape[0]):
                plt.plot(e[j,:])
            plt.xlabel('Time')
            plt.ylabel('Activation')
            plt.title('Activation over time across encoding dimensions')
            plt.savefig(str(new_save / f'activationvtime{i}.png'),dpi=300)
            plt.clf()
            plt.close()
        i += 1

def round_sig(x, digits):
    if x == 0:
        return 0
    
    logval = math.log10(abs(x))
    decimal_places = -int(logval) + (digits - 1)
    
    return round(x, decimal_places)

def plot_weights(root):
    pass

def plot_psd(root, flist=[]):
    root=Path(root)
    enc_root = root / 'encodings'
    enc_files = enc_root.glob('*.pt')
    save_path = root / 'plots'
    save_path = save_path / 'psd'
    save_path.mkdir(exist_ok=True)
    plt.style.use('ggplot')
    i = 0
    avg_psd = None
    avg_freq = None

    rfeats = {}
    for r in enc_files:
        name = r.name.replace('.pt','')
        rfeats[name] = np.transpose(np.squeeze(torch.load(r).numpy()))
    if flist != []:
        files = flist
    else:
        files = list(rfeats.keys())
    for f in files:
        if i < 10:
            encoding = rfeats[f]
            x_vals = []
            y_vals = None
            for j in range(encoding.shape[0]):
                x,y = plt.psd(encoding[j,:])
                x_vals.append(x)

                if y_vals is None:
                    y_vals = y

            plt.xlabel('Frequency')
            plt.ylabel('PSD (db)')
            plt.title('PSD', loc='center')
            plt.savefig(str(save_path/f'psd{i}.png'), dpi=300)
            plt.clf()
            plt.close()

            #x_vals = np.stack(x_vals,0)
            plt.clf()
            #if avg_psd is None:
            #    avg_psd = x_vals
            #    avg_freq = y_vals
            #else:
            #    avg_psd = np.add(avg_psd, x_vals)

        i += 1
    
    #avg_psd = avg_psd / len(enc_files)
    #plt.plot(avg_psd, y_vals)
        
    #plt.xlabel('Frequency')
    #plt.ylabel('PSD (db)')
    #plt.title('Average PSD', loc='center')
    #plt.savefig(str(save_path/f'avgpsd.png'), dpi=300)


def plot_filtermse(root):
    root = Path(root)
    save_path = root / 'plots'
    save_path = save_path / 'filters'
    save_path.mkdir(exist_ok=True)

    with open(str(root / 'metrics.json'), "rb") as file:
        metrics = json.load(file)
    
    fmse = np.asarray(metrics['filtered_mse'])
    bmse = np.asarray(metrics['baseline_filtered'])
    cutoffs = metrics['cutoffs']

    fmse_df = pd.DataFrame(fmse, columns=cutoffs)
    fmse_df['encoding'] = fmse_df.index
    fmse_melted = pd.melt(fmse_df, id_vars=['encoding'], value_vars=cutoffs, var_name='cutoff', value_name='mse')

    bmse_df = pd.DataFrame(bmse, columns=cutoffs)
    bmse_df['encoding'] = bmse_df.index
    bmse_melted = pd.melt(bmse_df, id_vars=['encoding'], value_vars=cutoffs, var_name='cutoff', value_name='mse')
  
    plt.style.use('ggplot')
    for i in range(fmse.shape[0]):
        filtered = fmse_melted[fmse_melted['encoding'] == i]
        plt.plot(filtered['cutoff'].tolist(),filtered['mse'].tolist())
    
    plt.gca().set_ylim(bottom=0)
    plt.xlabel('Filter Cutoff Frequency')
    plt.ylabel('MSE')
    plt.title('MSE after filtering')
    plt.savefig(str(save_path / f'allfilters.png'),dpi=300)
    plt.clf()

    avg_filtered = fmse_melted.groupby('cutoff').mean()
    avg_baseline = bmse_melted.groupby('cutoff').mean()
    plt.plot(avg_filtered.index.tolist(),avg_filtered['mse'].tolist(), color='black', label='Encodings')
    plt.plot(avg_baseline.index.tolist(),avg_baseline['mse'].tolist(), color='crimson', linestyle='--', label='Baseline')
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.xlabel('Filter Cutoff Frequency')
    plt.ylabel('MSE')
    plt.title('Average MSE after filtering')
    plt.savefig(str(save_path / f'avgfilters.png'),dpi=300)
    plt.close()


root = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_e3_iek5_d3_idk5_lr0.0001e500bs16_adamw_mse_tvl2_a0.0001_earlystop_bnf'
test_ema = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/librispeech/test/sparc'

plot_logs(root)
plot_filtermse(root)
d = plot_reconstructions(root, test_ema)
#filtermse_baseline(test_ema)
plot_psd(root, flist=d)
plot_activations(root, test_ema, upper_text=10**8, lower_text=10**7, flist=d)





    



