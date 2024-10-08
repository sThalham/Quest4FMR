# Quest4FMR

## Introduction
**Multimodal Foundation Models for Robotics** This repository contains functions for extracting the spatial tokens of different Vision-Language foundation models.

## Installation

Only model supported for now is CLIP.

### Prerequisites
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

4. Check out instantiated and modified CLIP: [vl_clip.py](https://github.com/sThalham/Quest4FMR/blob/main/vl_clip.py)


### Model Overview

TODO

| Backbone | Parameters | Size (MB) |
| :-----------------: | :---------: | :---------: |
| CLIP vit_b16 | 
|         | 
