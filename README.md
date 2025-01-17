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

You will also need to install:
* [database_utils repo](https://github.com/dwiepert/database_utils.git)

## Data
This model is trained/validated/tested with [LibriSpeech ASR Corpus](https://www.openslr.org/12). We use the clean sets and 100 hours of training data (train-clean-100) and extract EMA features from this. 

### Extracting EMA from Audio
This autoencoder expects that 14-dimensional EMA features have been extracted (each audio is converted into a *Tx14* matrix where *T* is the number of timepoints). The EMA features can be extracted using [audio_features repo](https://github.com/dwiepert/audio_features.git), installing the proper requirements for that (including the SPARC model from Cho et al. 2024 that extracts EMA features from a waveform), and running with the following parameters:

```
python stimulus_features.py --stimulus_dir=PATH_TO_AUDIO_FILES --out_dir=PATH_TO_SAVE --model_name=en --feature_type=sparc --return_numpy --recursive --skip_window --keep_all
```

This code also has more extensive options for windowing, but training/validation/testing is done with EMA features extracted from the entire audio file. 

## Training/Evaluating Model
The model can be trained and evaluated using [run_emaae.py](https://github.com/dwiepert/emaae/main/tree/run_emaae.py). The model is evaluated using mean squared error and quantifying sparsity as the number of zero elements in the encoded representation. 

Required/Important arguments:
* --train_dir, --test_dir, --val_dir: Path to feature directorys with train/test/validation data (can be cotton candy path)
* --out_dir: Local directory to save outputs
* --train: flag for training model (must be true to train)

Feature loading arguments:
* --recursive: use to flag whether to recursively load features
* --bucket: cotton candy bucket name if loading using cotton candy

Model initialization arguments:
* --model_config: path to json with model arguments
* --model_type: type of model to initialize
* --inner_size: dimension of encoded representations
* --n_encoder: number of encoder blocks
* --n_decoder: number of decoder blocks
* --checkpoint: checkpoint path or name of checkpoint

Training arguments:
* --batch_sz: batch size
* --epochs: number of training epochs
* --lr: learning rate
* --optimizer: type of optimizer to initialize
* --autoencoder_loss: type of autoencoder loss to use
* --sparse_loss: type of sparsity loss to use
* --penalty_scheduler: type of penalty scheduler
* --penalty_gamma: alpha update parameter for scheduler
* --alpha: preset alpha for loss function (don't use if using scheduler, default = None)



### TODO
1. Consider using conv-transpose still in the encoder part of the network?* last thing to try
2. same # of layers in encoder/decoder 
3. More layers in encoder/decoder
4. More epochs - Try 100 see what changes?
5. Try smaller batch size - see what changes? 1, 8, 16, 32
6. Try different learning rate? 0.0001, 0.001, 0.0005