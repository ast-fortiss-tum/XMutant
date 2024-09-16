# Test Input Generator for IMDB - Detailed Installation Guide #

## General Information ##
This folder contains the application of the XMutant approach to the IMDB sentiment analysis problem.
This tool is developed in Python. These instructions are for python 3.8.

## Dependencies ##

### 1. Environment ###
Install [conda](https://docs.conda.io/en/latest/miniconda.html#) and create an environment with `python 3.8`

[/XMutant/environment_MNIST_IMDB.yml](/XMutant/environment_MNIST_IMDB.yml)

```
conda create -n xmutant python=3.8
conda activate xmutant
```


## Usage ##

### Input ###

* A trained model in h5 format. The default one is in the folder `models`;
* Download nltk dataset and WordNet dataset.
```
python download.py
```
* `config.py` containing the configuration of the tool selected by the user.

### Output ###
When the run is finished, the tool produces the following outputs in the folder `runs`:
* the folder containing the generated inputs (in array format) and summaries (in csv format).

### Run the Tool ###
Run the command:
`python main.py`
