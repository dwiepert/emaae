# EMA Autoencoder 
Autoencoder meant to learn semi-shift invariant representations of EMA features. 

## Setup
In order to install all components easily, it is best to have a conda environment (some packages installed with conda and not setup.py)

To install, use

```
$ git clone https://github.com/dwiepert/emaae.git
$ cd audio_features
$ pip install . 
```

## Data
This model is trained/validated/tested with [LibriSpeech ASR Corpus](https://www.openslr.org/12). We use the clean sets and 100 hours of training data (train-clean-100) and extract EMA features from this. 

### Extracting EMA from Audio
This autoencoder expects that 14-dimensional EMA features have been extracted (each audio is converted into a *Tx14* matrix where *T* is the number of timepoints). The EMA features can be extracted using [audio_features repo](https://github.com/dwiepert/audio_features.git), installing the proper requirements for that (including the SPARC model from Cho et al. 2024 that extracts EMA features from a waveform + database_utils (which you will need to be granted access to)), and running with the following parameters:

```
python stimulus_features.py --stimulus_dir=PATH_TO_AUDIO_FILES --out_dir=PATH_TO_SAVE --model_name=en --feature_type=sparc --return_numpy --recursive --skip_window --keep_all
```

This code also has more extensive options for windowing, but training/validation/testing is done with EMA features extracted from the entire audio file. 

## Training/Evaluating Model
The model can be trained and evaluated using [run_emaae.py](https://github.com/dwiepert/emaae/main/tree/run_emaae.py). The model is evaluated using mean squared error and quantifying sparsity as the number of zero elements in the encoded representation. 

Required/Important arguments:
* --train_dir, --test_dir, --val_dir: Path to feature directorys with train/test/validation data (can be cotton candy path)
* --out_dir: Local directory to save outputs

Feature loading arguments:
* --recursive: use to flag whether to recursively load features
* --bucket: cotton candy bucket name if loading using cotton candy

Model initialization arguments:
* --model_config: path to json with existing model arguments (used to initialize a model with the same parameters (#layers, input dimension, inner size))
* --model_type: type of model to initialize (OPTIONS = ['cnn'])
* --inner_size: dimension of encoded representations (OPTIONS = [1024, 768])
* --n_encoder: number of encoder blocks (OPTIONS = 2-5)
* --n_decoder: number of decoder blocks (OPTIONS = 2-5)
* --checkpoint: checkpoint path or name of checkpoint, expecting a .pth file containing the saved WEIGHTS

Training arguments:
* --eval_only: flag for skipping training (all following arguments become irrelevant)
* --early_stop: flag for early stopping
* --batch_sz: batch size (default = 32)
* --epochs: number of training epochs (default = 500)
* --lr: learning rate (default=0.001)
* --optimizer: type of optimizer to initialize (OPTIONS = ['adamw', 'adam'])
* --autoencoder_loss: type of autoencoder loss to use (OPTIONS = ['mse'])
* --sparse_loss: type of sparsity loss to use (OPTIONS = ['l1'])
* --weight_penalty: flag for adding a penalty based on the model weights
* --alpha: preset or starting alpha for loss function (default = 0.25)
* --update: flag to specify whether alpha is being updated 
* --alpha_epochs: number of epochs to update alpha for (default=15 if early stop is enabled, otherwise it is set to be equivalent to epochs)
* --penalty_scheduler: type of penalty scheduler (OPTIONS = ['step])

## Visualizations
Plots are all created using R ([Analysis.Rmd](https://github.com/dwiepert/emaae/main/tree/Analysis.Rmd)).

This has options to create the following plots:
* Training vs. Validation loss (Combined, Individual)
* MSE vs. Alpha
* Eval - Ground Truth EMA vs. Predicted EMA 
* Visualizing weights? https://gist.github.com/krishvishal/e6bebc0d809a31f56cbccf5e15f24016
* Visualizing encoding?

### TODO
1. Try with keeping loudness?