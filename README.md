# DominaultNet
> Faces super resolution, Multiscale Parsing Attention Module, Multichannel Detail Attention Module

## Framework
<div align=center><img width="760" height="340" src=img/img_1.jpg/></div>

## Prerequisites

- Windows 
- Python 3.10
- Pytorch 2.0.1
- NVIDIA GPU + CUDA


## Train
* Firstly, you need to download the CelebA dataset and crop the center of the CelebA image to 128*128 size.
* Secondly, after downsampling the image to 16*16 size, it will be used as ground truth and input image respectively.
* Finally, set the paths of ground truth and input image in the corresponding positions in (dataset.py/dataset_parsing.py/dataset_parsingnet.py).
  ### Instructions
