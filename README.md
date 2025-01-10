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

You wil lalso need to install:
* [database_utils repo](https://github.com/dwiepert/database_utils.git)

## Extracting EMA from Audio
This autoencoder expects that 14-dimensional EMA features have been extracted (each audio is converted into a *Tx14* matrix where *T* is the number of timepoints). The EMA features can be extracted using [audio_features repo](https://github.com/dwiepert/audio_features.git), installing the proper requirements for that (including the SPARC model from Cho et al. 2024 that extracts EMA features from a waveform), and running with the following parameters:

```
python stimulus_features.py --stimulus_dir=PATH_TO_AUDIO_FILES --out_dir=PATH_TO_SAVE --model_name=en --feature_type=sparc --return_numpy --recursive --skip_window --keep_all
```

This code also has more extensive options for windowing, but training/validation/testing is done with EMA features extracted from the entire audio file. 

## Data
This model is trained/validated/tested with [LibriSpeech ASR Corpus](https://www.openslr.org/12). We use the clean sets and 100 hours of training data (train-clean-100). 

