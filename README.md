# Quest4FMR

## Introduction
**Multimodal Foundation Models for Robotics** This repository contains functions for extracting the spatial tokens of different Vision-Language foundation models.

## Installation

Only model supported for now is CLIP.

### Torch-multimodal for CLIP, Flava, and CoCA
1. Install conda environment

    ```
    conda create -n multimodal python=3.10
    conda activate multimodal

2. Install pytorch, torchvision, and torchaudio. See [PyTorch documentation](https://pytorch.org/get-started/locally/).

    ```
    # Tested with NVIDIA driver 530.30.02 and Cuda 12.1
    # Select the nightly Pytorch build, Linux as the OS, and conda. Pick the most recent CUDA version.
    conda install pytorch torchvision torchaudio pytorch-cuda=\<cuda_version\> -c pytorch-nightly -c nvidia

    # For CPU-only install
    conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
    ```

3. Build facebookresearch/multimodal

    ```
    cd multimodal
    python -m pip install torchmultimodal-nightly
    cd ..
    ```

4. Check out instantiated and modified CLIP: [vl_clip.py](https://github.com/sThalham/Quest4FMR/blob/main/vl_clip.py), Flava: [vl_flava.py](https://github.com/sThalham/Quest4FMR/blob/main/vl_flava.py), and CoCa: [vl_coca.py](https://github.com/sThalham/Quest4FMR/blob/main/vl_coca.py) 


### OV-Dino
1. Install conda environment

    '''
    conda create -n ov-dino python=3.10
    conda activate ov-dino
    '''

2. Install according to the Installation guide in GLIPv2: [OV-DINO](https://github.com/wanghao9610/OV-DINO)

    '''
    # modify the following lines
    export CUDA_HOME="/usr/local/cuda"
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y 
    python -m pip install git+https://github.com/facebookresearch/detectron2.git@717ab9f0aeca216a2f800e43d705766251ba3a55
    '''

### GLIP
1. Install conda environment

    ```
    conda create -n glip python=3.10
    conda activate glip
    ```

2. Install pytorch and other dependencies
    ```
    # Tested with NVIDIA driver 530.30.02 and Cuda 12.1
    # Select the nightly Pytorch build, Linux as the OS, and conda. Pick the most recent CUDA version.
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia

    # For CPU-only install
    conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
    ```

3. Install packages and build locally

    ```
    pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
    pip install transformers 
    python setup_glip.py build develop --user
    ```

3. Check out instantiated and modified GLIPv2: [vl_glipv2.py](https://github.com/sThalham/Quest4FMR/blob/main/vl_glipv2.py)


### Model Overview

TODO

| Backbone | Parameters | Size (MB) |
| :-----------------: | :---------: | :---------: |
| CLIP vit_b16 | 
|         | 
