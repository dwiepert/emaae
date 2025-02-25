"""
"""
#IMPORTS
#built-in
import glob 
from pathlib import Path

#third-party
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis
import torch

#local
from emaae.io import EMADataset

original = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/librispeech/test/sparc'
test_dataset = EMADataset(root_dir=original, recursive=True, cci_features=None)
test_feats = test_dataset.features

enc_dir = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/encodings'
enc_files = glob.glob('*.pt', root_dir=enc_dir, recursive=False)

encodings = []
encodings_flattened = []
k = []
baseline_k = []

mask = np.ones(14, dtype=bool)
mask[[12]] = False

for f in enc_files:
    encoding = torch.load(Path(enc_dir)/f).numpy()
    enc_flattened = encoding.flatten()
    k.append(kurtosis(enc_flattened))
    encodings.append(encoding)
    encodings_flattened.append(enc_flattened)

    #baseline
    bs = f.split("_")[0]
    feat = test_feats[bs]
    feat = feat[:,mask]
    padded = np.pad(feat, (0,(1024-13)), mode='constant', constant_values=0)
    baseline_k.append(kurtosis(padded.flatten()))
    #padded = np.pad(feat, (0,(1024-13)), mode='constant', constant_values=0)

print(f'Avg kurtosis: {np.mean(k)}')
print(f'Avg kurtosis baseline: {np.mean(baseline_k)}')
encodings_cat = np.concatenate(encodings_flattened)
#TRY
#encodings_cat = np.log(encodings_cat[encodings_cat != 0])

#logbins = np.geomspace(encodings_cat.min(), encodings_cat.max(),10plt.figure(figsize=(8, 6))
plt.hist(encodings_cat, bins=100, log=True)
#plt.xscale('log')
plt.title('Log distribution of activations')
plt.ylabel('Frequency')
plt.show()
plt.savefig('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/plots/activations_dist.png')
plt.clf()


#TRY

e = np.squeeze(encodings[100])
plt.figure(figsize=(5,5))
plt.imshow(e)
plt.ylabel('Encoding dimension')
plt.xlabel('Time')
plt.show()
plt.savefig('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/plots/visualization1.png')
plt.clf()

plt.spy(e)
plt.ylabel('Encoding dimension')
plt.xlabel('Time')
plt.show()
plt.savefig('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/plots/visualization2.png')
plt.clf()

for i in range(e.shape[0]):
    plt.plot(e[i,:])
plt.ylabel('Activation')
plt.xlabel('Time')
plt.show()
plt.savefig('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/plots/visualization3.png')

