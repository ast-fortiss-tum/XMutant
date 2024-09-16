# Test Input Generator for MNIST - Detailed Installation Guide #

## General Information ##
This folder contains the application of the XMutant approach to the handwritten digit classification problem.
This tool is developed in Python. These instructions are for python 3.8.

## Dependencies ##

### 1. Environment ###
Install [conda](https://docs.conda.io/en/latest/miniconda.html#) and create an environment with `python 3.8`

[/XMutant/environment_MNIST_IMDB .yml](/XMutant/environment_MNIST_IMDB.yml)

```
conda create -n xmutant python=3.8
conda activate xmutant
```


## Usage ##

### Input ###

* A trained model in h5 format. The default one is in the folder `models`;
* A list of seeds used for the input generation. The default list is in the folder `original_dataset`;
* `config.py` containing the configuration of the tool selected by the user.

### Output ###
When the run is finished, the tool produces the following outputs in the folder `runs`:
* the folder containing the generated inputs (in array format) and summaries (in csv format).

### Run the Tool ###
Run the command:
`python main.py`
