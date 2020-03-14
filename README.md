Audio Source Separation with Deep Clustering
=============================================

Overview:
---------

Our project is focused on tackling the problem of audio source separation. To put it simply:

`given an audio file containing sounds from multiple different voices/instruments/sources, can a computer efficiently classify and separate a sound file into separate sound files of its constituents?`


## Introduction

Our repository currently is a boilerplate for reproducible and transparent computer audition research that leverages
[nussl](https://interactiveaudiolab.github.io/nussl/), a source separation library. This 
project provides boilerplate code for training neural network models to separate mixtures
containing multiple speakers, music, and environmental sounds. It is easy to add and
train new models or datasets (GPU sold separately). The goal of this project is to enable
further reproducibility within the source separation community.


## Features

The following models can be trained:

- Mask Inference
- Deep Clustering (The algorithm we are running to currently separate sounds)
- Chimera
- TasNet

on the following datasets:

- MUSDB18
- MIR-1k
- Slakh
- WSJ0-mix2
- Wham!
- RemixPacks600 (Currently in the works)

This project utilizes building block components from `nussl` for input/output 
(reading/writing audio, STFT/iSTFT, masking, etc.), and for neural network construction
(recurrent networks, convolutional networks, etc) to train models with minimal setup.
The main source separation library, `nussl`, contains many pre-trained models trained
using this code. See the [External File Zoo (EFZ)](http://nussl.ci.northwestern.edu/)
for

This project uses
[cookiecutter](https://cookiecutter.readthedocs.io/en/latest/readme.html).
Cookiecutter is a *logical, reasonably standardized, but flexible project structure
for doing and sharing research.* This project and `nussl` are both built upon
the [PyTorch](https://pytorch.org/) machine learning framework, as such, building new
components is as simple as adding new PyTorch code, though writing python is not required.


## Requirements

- Install `cookiecutter` command line: `pip install cookiecutter` (generates boilerplate 
code)

- Install [Anaconda or Miniconda](https://www.anaconda.com/distribution/)

- Install Poetry: https://poetry.eustace.io/docs/#installation (dependency management)

- In order to efficiently install all packages with dependencies, `make poetry` in a new Conda environment. This will install all of the necessary components in the right order.

## Usage

We have provided a completely configured `nussl` project to use the model we have trained. After cloning the project, run the following commands to set up the project. It is assumed this will be done a Linux System.

Install SoX if you haven't already:
```
sudo apt-get install sox
```
Run the following commands to setup the environment.
```
source setup/environment/ml_config.sh
conda create -n nussl python=3.7
conda activate nussl
conda install pytorch==1.4.0
make poetry
make install
```

## Class Project API

For this project, we will showcase our model separating snippets of songs from the musdb dataset. More information about this dataset can be found here: [https://sigsep.github.io/datasets/musdb.html]

In our repository, we have also provided a sample of musdb, containing 3000 4 second snippets of each song, each contianing only 2 sources. 

To randomly separate one of the snippets, run:
```
python -m scripts.visualize -p best_model/config.yml
```
After the script finishes, you will find the output in `output/viz/test`. The folder will be named a string of 8 numbers. The newest one is most likely the one generated from the visualize script eariler. In this folder, you will find:

 - `mixture.wav`, which is an audio file containing both of the sources. 
 - `source0.wav` and `source1.wav`, which are the audio files containing the separated source
 - `viz.png` contains graphical visualization for clustering, as well as the waveforms of the sources.

## Documentation

The documentation is [here](https://pseeth.github.io/cookiecutter-nussl/). It includes
guides for getting started, training models, creating datasets, and API documentation.


## Deep Clustering in Depth:

The deep clustring algorithm is a 2-step machine learning paradigm used to separate audio files. On a high level, the training steps are as follows:

1. Audio files are pieced into time-frequency blocks (ex: .2 seconds, 1000-1500 Hz) and labeled as a certain voice/instrument/environmental sound.

2. These "pieces" and their respective labels are fed into a traditional Deep Neural Net, in which the Deep Neural Net embeds the 2-dimensional audio pieces into a 20-dimensional embedding space.

3. Within this embedding space, the K-Means clustering algorithm is run in order to group together audio pieces that are most similar-- or in other words, are closest together in the embedding space.

4. The labels from the resulting K-Means clustering is compared to the ground truth labels, the loss/accuracy is calculated, and the resulting change in weights via gradient descent is backpropogated through the DNN, leading to a more accurate source separating model.

For an in-depth study of the model we used, check out the report: https://arxiv.org/abs/1508.04306

## Guiding Principles

The idea behind this project structure is to make it easy to use `nussl` to set up
source separation experiments. The functionality here is such that classes are taken
from `nussl` and can be extended and customized by your package code. For example, to
set up a new type of training scheme, you might subclass the Trainer class from 
`nussl` and then modify it by overriding functions from the original Trainer class
with your own implementation.

If you're trying new types of models, you can use the existing SeparationModel class but
add custom modules in `model/extras/`. These extra modules are handed to 
SeparationModel so it can resolve the model configuration. Note that models trained using
extra modules will need to be shipped with the accompanying code to be portable. For a new
model architecture to be shipped via nussl's external file zoo, the accompanying modules
must be pull requested to the main nussl repository and then deployed.

If you are implementing new separation algorithms, you can work in the `algorithms/`
folder. Implement your algorithm and then include it in `algorithms/__init__.py`. The 
base classes in nussl for all separation algorithms are included already: `SeparationBase`,
`MaskSeparationBase`, and `ClusteringSeparationBase`. 
Your new algorithm will now be accessible by the scripts via `.yml` files. Once your new
algorithm is implemented to your satisfaction, you can factor it out of the cookiecutter
and start a PR for nussl to contribute it to the main library.

All scripts, which are kept in the `scripts/` folder should take in a `.yml` file and 
function according to the parsed `.yml` file. This is to make sure every part of the
pipeline you create in your experiment is easily reproducible by processing a sequence
of `.yml` files with their associated scripts. This prevents "magic commands" with
mysterious and long forgotten command-line arguments that you ran one time 3 months ago 
from occurring. 


## License

This project is licensed under the terms of the [MIT License](/LICENSE)