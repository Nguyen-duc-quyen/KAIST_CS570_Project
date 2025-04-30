# Classification Template

A Template for training, evaluating Diffusion Models


## Create conda environment
```
conda create -n DMs python=3.10 pip cuda-toolkit=12.1 gxx cudnn -c conda-forge
conda activate DMs
```

## Download PyTorch
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Install Requirements
```
pip install -r requirements.txt
```

### Questions:
**Why use Interpolation + Convolution instead of Transpose Convolution?**
- To avoid checkerboard effect introduced by transpose convolution, also to have more control.

**Why use GroupNorm instead of BatchNorm?**
- Diffusion models work with very large images and big models, thus the batchsize tends to be small (usually 1 or 2). Hence, BatchNorm becomes very noisy and meaningless. On the other hand, GroupNorm normalizes across channels within each individual sample, making it completely independent of the batchsize. Also, to have consistency between the training and inference processes.

**Why don't use LayerNorm**
- LayerNorm normalizes across all channels and spatial dimensions, while GroupNorm only normalizes across groups of channels for each spatial location. LayerNorm is great for Transformers but not for CNNs.

**Why use Norm->Act->Conv instead of traditional Conv->Act->Norm?**
- Firstly, pre-activation is smoother for optimization. Secondly, it is consistent with the attention block. 

**What is Scale-Shift Norm**
- Normalization + learned Scale and Shift from embedding. It is a way for diffusion models to inject information deeply into the network. In the U-Net architecture: 
```
scale, shift = torch.chunk(emb_out, 2, dim=1)
h = norm(h) * (1 + scale) + shift
```

