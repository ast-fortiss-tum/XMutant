# Test Input Generator for MNIST - Detailed Installation Guide #

## General Information ##
This folder contains the application of the XMutant approach to the handwritten digit classification problem.
This tool is developed in Python. These instructions are for Ubuntu 18.04 (bionic) OS and python 3.8.

## Dependencies ##

### 1. Environment ###
Install [conda](https://docs.conda.io/en/latest/miniconda.html#) and create an environment with `python 3.8`

[/XMutant/environment_LK_ADS.yml](/XMutant/environment_LK_ADS.yml)

```
conda create -n xmutant-ads python=3.8
conda activate xmutant-ads
```

### Udacity Driving Simulator

The binary of the driving simulator in the `sim` directory. Precompiled binaries for Linux are available [XMutant-LK-ADS/sim/udacity_sim_linux/udacity_sim_linux.x86_64](XMutant-LK-ADS/sim/udacity_sim_linux/udacity_sim_linux.x86_64). 


## Usage ##

The following command starts the Udacity simulator with randomly generated roads with seed 0 that have 12 control points. The agent that drives is a random agent.

```
python main_comparison_test.py --seed 0 \
                                --mutation-type RANDOM \
                                --num-control-nodes 12 \
                                --mutation-method random
```

The following command runs a overall experiment via bash.
```
bash run_test.bash
```
## Troubleshooting ##

### First install requirements-macos.txt
Recommend to check the version of following packages
```
gym==0.21.0
numpy==1.24.2
Pillow==8.4.0
python-socketio==4.5.1
python-engineio==3.11.2
flask==2.0.0
Shapely==1.8.1
matplotlib==3.3.4
descartes==1.1.0
eventlet==0.33.3
pandas
natsort
tf-keras-vis
tqdm
```

### Error by installing gym
Solution: install setuptools and wheel
```
pip install setuptools==65.5.0
pip install wheel==0.38.4
```

### Run error with protobuf. 
Solution: Downgrade the protobuf package to 3.20.x or lower.
```
pip install protobuf=3.20.0
```

### AttributeError: module 'dns.rdtypes' has no attribute 'ANY'
Solution: upgrade  eventlet  
```
pip install eventlet==0.33.3
```

### Some packages with specific version
```
tensorflow==2.11.0
numpy==1.22.1
Werkzeug==2.2.2
```

### Canberra-gtk-module
```
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module -y
```

### Failed to create valid graphics context
Solution: Please ensure you meet the minimum requirements. E.g. OpenGL core profile 3.2 or later for OpenGL Core renderer
```
sudo apt install mesa-utils
sudo apt install freeglut3-dev
```
How to check:
```
glxinfo | grep "OpenGL version"
```

### libGL error: MESA-LOADER: failed to open iris
Solution: 
```
conda install -c conda-forge libstdcxx-ng
```
