from emaae.models import CNNAutoEncoder
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


model_config_path = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/model_config.json'
checkpoint_path = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/models/CNN_Autoencoder_ne2_nd2_innersz1024_bestmodel_e13.pth'
save_path =  Path('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/plots')
save_path.mkdir(exist_ok=True)

with open(model_config_path, "rb") as f:
    model_config = json.load(f)
   


model = CNNAutoEncoder(input_dim=model_config['input_dim'], n_encoder=model_config['n_encoder'], n_decoder=model_config['n_decoder'], inner_size=model_config['inner_size'])

checkpoint = torch.load(checkpoint_path, map_location='cpu')
        #print(checkpoint.keys())
        #print(model.state_dict().keys())
model.load_state_dict(checkpoint)

weights = model.get_weights()
conv1 = weights[0].numpy()

def get_plots(weights, norm_function, save_path, prefix):
    norm_weights = norm_function(weights)

    #get strongest activation
    strongest_activation = np.linalg.norm(np.asarray(norm_weights), axis=(1,2))
    m = np.argmax(strongest_activation)
    
    plt.imshow(norm_weights[m,:,:])
    plt.title(f'Filter {m}')
    plt.ylabel('EMA dim')
    plt.xlabel('Time')
    plt.savefig(str(save_path / f'{prefix}.png'), transparent=True, bbox_inches='tight', pad_inches=0)
    plt.clf()

    for i in range(norm_weights.shape[1]):
        singledim = np.squeeze(norm_weights[:,i,:])
        t = np.linalg.norm(singledim, axis=(1))
        ti = np.argmax(t)
        plt.imshow(norm_weights[ti,:,:])
        plt.title(f'Filter {ti}, ema {i}')
        plt.ylabel('EMA dim')
        plt.xlabel('Time')
        plt.savefig(str(save_path / f'{prefix}_{i}.png'), transparent=True, bbox_inches='tight', pad_inches=0)
        plt.clf()

max_norm = lambda weights: (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
std_norm = lambda weights: (weights - np.mean(weights)) / np.std(weights)

def pair(weights, norm_function=max_norm):
    flattened = np.transpose(weights.reshape(weights.shape[0],weights.shape[1]*weights.shape[2]))
    flattened = norm_function(flattened)
    paired_feats = [weights[:,:2,:], weights[:,2:4,:], weights[:,4:6,:], weights[:,6:8,:], weights[:,8:10,:],weights[:,10:12,:]]
    paired_feats = [np.reshape(t,(t.shape[0],t.shape[1]*t.shape[2])) for t in paired_feats]
    pf = np.stack(paired_feats, axis=1)
    return pf

get_plots(conv1, max_norm, save_path, 'maxmin')
get_plots(conv1, std_norm, save_path, 'std')
#pf = pair(conv1)
#print(pf)
get_plots(conv1, pair, save_path, 'paired_maxmin')

print('pause')
#cpc = PCA(n_components=13)
#conv1_pca = cpc.fit_transform(conv1_flattened)



# for i in range(conv1_pca.shape[1]):
#     data = conv1_pca[:,i].reshape(13,3)
#     plt.imshow(data)
#     plt.title(f'PCA {i}')
#     plt.savefig(str(save_path / f'pcafilter_{i}.png'))
#     #plt.show()
#     plt.clf()

# to print('pause')


""" max_norm = -np.inf
max_v = None

#create a visualization for max
for i in range(conv1.shape[0]):
    norm = np.linalg.norm(conv1[i,:,:])
    if norm > max_norm:
        max_v = i

plt.imshow(conv1[max_v,:,:])
plt.title(f'Max magnitude filter {max_v}')
plt.savefig(str(save_path / 'max_mag_filter.png'))
#plt.show()
plt.clf()

# create a visualization for the max of each thing?
for i in range(conv1.shape[1]):
    max_avg = -np.inf
    max_v = None
    for j in range(conv1.shape[0]):
        avg = np.sum(conv1[j,i])/conv1.shape[2]
        if avg > max_avg:
            max_v = j
    
    if max_v != 511:
        print('pause')
    plt.imshow(conv1[max_v,:,:])
    plt.title(f'Max magnitude filter {max_v} for feature {i}')
    plt.savefig(str(save_path / f'max_mag_filter_{i}.png'))
    #plt.show()
    plt.clf()


print('pause')
#weight = conv1.weight.data.numpy() """