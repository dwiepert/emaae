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

enc_dir = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e50bs16_adamw_mse_l1_a0.25_earlystop/encodings'
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
    padded = np.resize(feat,(feat.shape[0],1024))
    baseline_k.append(kurtosis(padded.flatten()))
    #padded = np.pad(feat, (0,(1024-13)), mode='constant', constant_values=0)

encodings_cat = np.concatenate(encodings_flattened)
#TRY
plt.hist(encodings, bins=100)
plt.show()


#TRY
e = encodings[5]
# plt.imshow(e)
# plt.show()

# plt.spy(e)
# plt.show()

for i in range(e.shape[1]):
    plt.plot(e[:,i])

plt.show()

