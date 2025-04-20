# EMA Autoencoder 
Autoencoder meant to learn semi-shift invariant representations of EMA features through slow feature analysis. 

## Setup
In order to install all components easily, it is best to have a conda environment (some packages installed with conda and not setup.py)

To install, use

```
$ git clone https://github.com/dwiepert/emaae.git
$ cd emaae
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

## Feature Dataset 
[FeatureDataset](https://github.com/dwiepert/emaae/main/tree/emaae/io/_dataset.py) is a simple torch dataset that loads .npz features. If loading EMA features it normalizes the features to -1 to 1 and removes the loudness dimension. [`custom_collatefn(batch)`](https://github.com/dwiepert/emaae/main/tree/emaae/io/_dataset.py) should be used with the `DataLoader`.

Parameters:
* `root_dir`: str/path-like, directory or bucket containing features
* `feature_type`: str to indicate which feature (default = 'ema')
* `recursive`: boolean, True if features are stored in sub-directories of root_dir
* `cci_features`: coral interface for loading features from a bucket (default = None)

## Model
The [CNN_Autoencoder](https://github.com/dwiepert/emaae/main/tree/emaae/models/_cnn_autoencoder.py) class contains the autoencoder architecture. The encoder and decoder can be flexibly built with different number of CNN blocks and different starting kernel sizes based on a few preset options. A basic CNN block contains a batch normalization layer, followed by 2D convolution, and ReLU activation. ReLU is left off of the final block in the encoder or decoder. 

* `input_dim`: int, specify the size of the input dimension. This influences the in_channels of the first encoder cnn block and the out_channels of the final decoder block (default = 13)
* `n_encoder`: int, specify the number of encoder blocks (default = 2)
* `n_decoder`: int, specify number of decoder blocks (default = 2)
* `inner_size`: int, specify the dimension of the encodings. This influences the out_channels of the final encoder block and the in_channels of the first decoder block (default = 2)
* `batchnorm_first`: bool, True if batchnorm should come before CNN layer (default=True)
* `final_tanh`: bool, True if using tanh activation instead of no activation in the final layer of encoder/decoder (default=False)
* `initial_ekernel`: int, specify size of the first kernel in encoder (default = 5)
* `initial_dkernel`: int, specify size of the first kernel in decoder (default = 5)
* `exclude_final_norm`: bool, True if excluding normalization from final CNN block in encoder and decoder (default=False)
* `exclude_all_norm`: bool, True if excluding all normalization from the model (default=True)

It is recommended to run the encoder and decoder separately: `model.encode(input)` and `model.decode(output)`

Other useful functions are `get_type()` which returns the 'name' of the model and `get_weights` which returns the model weights

## Custom loss
The [CustomLoss](https://github.com/dwiepert/emaae/main/tree/emaae/models/_loss.py) is an important part of training the model. It includes options for a few different kinds of training loss procedures. 
    * `loss1_type`: str, specifies the reconstruction loss. Currently only compatible with MSE (default = 'mse')
    * `loss2_type`: str, specifies the loss to use for the encoding. Compatible with L1 norm ('l1'), TVL2 ('tvl2'), and low-pass filtering ('filter'). (default = 'l1')
    * `alpha`: float, weight for weighting the encoding loss (default = 0.25)
    
There are a few other parameters that are not being actively used, but can be checked out in the function. 

### custom early stopping/scheduler
These are helper classes that aren't currently being used but can be implemented with parameters in [run_emaae.py](https://github.com/dwiepert/emaae/main/tree/run_emaae.py). They can be found in [models](https://github.com/dwiepert/emaae/main/tree/models/)

## Training/Evaluating Model
The model can be trained and evaluated using [run_emaae.py](https://github.com/dwiepert/emaae/main/tree/run_emaae.py). The model is evaluated using mean squared error and a sweep of low pass filters. 

`--early_stop` flag recommened for saving best model

Required/Important arguments:
* `--train_dir`, `--test_dir`, `--val_dir`: Path to feature directorys with train/test/validation data (can be cotton candy path)
* `--out_dir`: Local directory to save outputs
* `--encode`, `--decode`: flags for saving encodings and decodings from test dir
* `--checkpoint`: checkpoint path or name of checkpoint if loading from existing checkpoint, expecting a .pth file containing the saved WEIGHTS
* `--eval_only`: flag for skipping training (all following arguments become irrelevant)

Feature loading arguments:
* `--recursive`: use to flag whether to recursively load features
* `--bucket`: cotton candy bucket name if loading using cotton candy


Model initialization arguments:
* `--model_config`: path to json with existing model arguments (used to initialize a model with the same parameters (#layers, input dimension, inner size))
* `--model_type`: type of model to initialize (OPTIONS = ['cnn'])
* `--inner_size`: dimension of encoded representations (OPTIONS = [1024, 512])
* `--n_encoder`: number of encoder blocks (OPTIONS = 2-3)
* `--initial_ekernel`: size of initial kernel in encoder (default = 5)
* `--n_decoder`: number of decoder blocks (OPTIONS = 2-3)
* `--initial_dkernel`: size of initial kernel in decoder(default = 5)
* `--batchnorm_first`: indicate whether batchnorm layer comes before non-linearity
* `--exclude_final_norm`: indicate whether to exclude final normalization layers from encoder/decoder.
* `--exclude_all_norm`: indicate whether to exclude all normalization layers from the model
* `--final_tanh`: indicate whether to add a tanh activation at final decoder layer
* `--residual`: bool, add residual connection in training

Training arguments:
* `--batch_sz`: batch size (default = 32)
* `--epochs`: number of training epochs (default = 500)
* `--lr`: learning rate (default=0.001)
* `--optimizer`: type of optimizer to initialize (OPTIONS = ['adamw', 'adam'])
* `--reconstruction_loss`: type of reconstruction loss to use (OPTIONS = ['mse'])
* `--encoding_loss`: type of encoding loss to use (OPTIONS = ['l1', 'tvl2', 'filter'])
* `--alpha`: preset or starting alpha for loss function (default = 0.25). Should drop significantly in magnitude if using TVL2 loss (e.g. 0.0001)

Filtering parameters:
* `--cutoff_freq`: float, cutoff frequency for lowpass filtering during training (default = 0.2)
* `--n_taps`: int, filter size (default = 51)
* `--n_filters`: int, number of filters for evaluation filtering (default = 20)

Optional parameters:
* `--patience`: how many epochs to wait for early stopping (default = 500)
* `--weight_penalty`: flag for adding a penalty based on the model weights
* `--update`: flag to specify whether alpha is being updated 
* `--alpha_epochs`: number of epochs to update alpha for (default=15 if early stop is enabled, otherwise it is set to be equivalent to epochs)
* `--penalty_scheduler`: type of penalty scheduler (OPTIONS = ['step])
* `--lr_scheduler`: bool, toggle for learning rate scheduler
* `--end_lr`: float, specify goal end learning rate

Example model command:
```
python3 run_emaae.py --train_dir=<PATH_TO_DIR> 
--test_dir==<PATH_TO_DIR> --val_dir==<PATH_TO_DIR> 
--out_dir==<PATH_TO_DIR>  --recursive --inner_size=1024 --n_encoder=2 --initial_ekernel=5 --n_decoder=2 --initial_dkernel=5 --batch_sz=16 --epochs=250 --lr=0.0001 --alpha=0.000 --early_stop --encode --decode 
--reconstruction_loss=tvl2 --batchnorm_first
```
## Visualizations
Plots are all created using R ([plotting.py](https://github.com/dwiepert/emaae/main/tree/helpers/plotting.py)).


