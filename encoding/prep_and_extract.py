import subprocess
import glob
from pathlib import Path
from emaae.io import EMADataset
from .downsample import downsample_vectors

stimulus_dir = Path('/mnt/data/dwiepert/data/processed_stimuli')
model_dir = Path('/mnt/data/dwiepert/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/models')
model_config = Path('/mnt/data/dwiepert/data/emaae/model_lr0.0003e51bs16_adamw_mse_l1_a0.25_earlystop/model_config.json')
out_dir = Path('/mnt/data/dwiepert/data/emaae/stimulus_features/')

path_to_stim_py = 'stimulus_features.py'
path_to_emaae_py = 'emaee.py'

# 1 - extract sparc features
sparc_out_dir = out_dir / 'sparc'
if not sparc_out_dir.exists():
    cmd = ['python', path_to_stim_py, f'--stimulus_dir={str(stimulus_dir)}', f'--outdir={str(sparc_out_dir)}',
            '--model_name=en', '--feature_type=sparc', '--return_numpy', '--recursive', 
            '--skip_window', '--keep_all']

    subprocess.run(cmd)

# 2 - extract encodings for stimulus - each model
models = glob.glob(str(model_dir / '*.pth'))
feature_dirs = []
for m in models:
    mn = models.split("/")[-1].replace('.pth','')
    new_out_dir = out_dir / mn
    enc_dir = new_out_dir / 'encodings'
    if not enc_dir.exists():
        cmd = ['python3', path_to_emaae_py, f'--train_dir={str(sparc_out_dir)}', f'--val_dir={str(sparc_out_dir)}', f'--test_dir={str(sparc_out_dir)}',
            f'--out_dir={str(new_out_dir)}', '--recursive', '--eval_only', f'--model_config={str(model_config)}', f'--checkpoint={m}']
        subprocess.run(cmd)

# 3 - load in features
ema = EMADataset(root_dir=sparc_out_dir, recursive=False)

# 3 - downsample features
ema_data = ema.features
downsampled_data = 

