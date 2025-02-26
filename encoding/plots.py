"""
"""
from pathlib import Path
import glob
import torch
import numpy as np
from emaae.io import EMADataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import kurtosis
#reconstruction vs. original features
original = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/librispeech/test/sparc'
reconstructed = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e50bs16_adamw_mse_tvl2_a0.25_earlystop_bnf/decodings'
#encodings = '/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e50bs16_adamw_mse_l1_a0.25_earlystop/encodings'
#ep = glob.glob('*.pt', root_dir=encodings, recursive=False)

# encs = []
# ks = []

# i = 0
# for e in ep:
#     enc = torch.load(Path(encodings) / e).numpy()
#     new_name = Path(encodings)/e.replace(".pt",".npy")
#     np.save(new_name, enc)
#     encs.append(enc.flatten())
#     ks.append(kurtosis(enc.flatten()))

# encs2 = np.concatenate(encs)
# k = kurtosis(encs2)
# k100 = kurtosis(encs[100])
# sns.kdeplot(encs[100])
# plt.xlim(-1,5)
# plt.show()
# print('pause')
test_dataset = EMADataset(root_dir=original, recursive=True, cci_features=None)
test_feats = test_dataset.features
files = list(test_feats.keys())
reconstructed_files = glob.glob('*.pt', root_dir=reconstructed, recursive=False)
reconstructed = Path(reconstructed)
rfeats = {}

for r in reconstructed_files:
    f = r.replace('.pt', '')
    rfeats[f] = np.transpose(np.squeeze(torch.load(reconstructed/r).numpy()))

mask = np.ones(14, dtype=bool)
mask[[12]] = False

mse = []
for i in range(len(files)):
    if i%10==0:
        print(f'{i}/{len(files)}')
    orig = test_feats[files[i]]
    orig = orig[:,mask]
    #omin = np.min(orig, axis=0)
    #omax = np.max(orig, axis=0)
    #orig = np.divide(np.subtract(orig, omin), np.subtract(omax,omin))
    pred = rfeats[files[i]]
    
    #pmin = np.min(pred, axis=0)
    #pmax = np.max(pred, axis=0)
    #pred = np.divide(np.subtract(pred, pmin), np.subtract(pmax,pmin))
    
    if i < 1:
        print('pause')
        #orig = test_feats[files[i]]
        #pred = rfeats[files[i]]

        rlist = np.arange(13)*5
        orig2 = np.add(orig, rlist) 
        pred2 = np.add(pred, rlist) 
        
        #figure, axis = plt.subplots(13,1)


        plt.plot(list(range(orig.shape[0])), orig2[:,:12], label='Original')
        plt.plot(list(range(orig.shape[0])), pred2[:,:12], label='Reconstructed')
        plt.yticks([])
        plt.xlabel('Time')
        plt.savefig('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e50bs16_adamw_mse_tvl2_a0.25_earlystop_bnf/plots/reconstructed_ema.png')
        plt.clf()

        plt.plot(list(range(orig.shape[0])), orig[:,12], label='Original')
        plt.plot(list(range(orig.shape[0])), pred[:,12], label='Reconstructed')
        plt.yticks([])
        plt.xlabel('Time')
        plt.savefig('/Users/dwiepert/Documents/SCHOOL/Grad_School/Huth/data/emaae/model_lr0.0003e50bs16_adamw_mse_tvl2_a0.25_earlystop_bnf/plots/reconstructed_pitch.png')
        plt.clf()


        print('pause')

    for j in range(orig.shape[0]):
        zero_inds = np.where(pred[j,:])
        mask_zeros = np.ones(orig.shape[1],dtype=bool)
        mask_zeros[zero_inds] = False
        morig = orig[j,mask_zeros]
        mpred = pred[j,mask_zeros]
        if morig.size != 0:
            mse.append(mean_squared_error(orig[j,mask_zeros], pred[j,mask_zeros]))
        #else:
            #mse.append(mean_squared_error(orig[j,:], pred[j,:]))
    #for j in range(13):
        #axis[i,0].plot(list(range(orig.shape[0])), orig[:,])
 
print(np.mean(mse)) 
    #print('pause')

    
        


print('pause')
