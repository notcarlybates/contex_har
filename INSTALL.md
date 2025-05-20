# Installation Guide

Create [Anaconda](https://www.anaconda.com/products/distribution) environment:

```
conda create -n context_har python==3.13.2
conda activate context_har
```

Install PyTorch distribution (CUDA 12.4):

```
pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

Install other packages:

```
pip install -r requirements.txt
```
