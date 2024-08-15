# Areal_prediction_framework
Automated download of raster data and acquisition dates from wms servers or geoportals then using UNet model for prediction target.

# Description
In this repository I used two submodule
[Data_acquisition](https://github.com/LUP-LuftbildUmweltPlanung/Data_acquisition): This repository contains several python scripts to download the image data as well as the acquisition dates from a specified wms server or geoportal, given one or multiple shape files. And  [UNet](https://arxiv.org/abs/1505.04597) this repository contains the code necessary to run a UNet based on the Dynamic Unet implementation of fastai. The implementation uses the PyTorch DeepLearning framework. UNet is used for image segmentation (pixel-wise classification). 

## Getting Started

### Dependencies
* GDAL, Pytorch-fast.ai, Scipy ... (see installation)
* Cuda-capable GPU ([overview here](https://developer.nvidia.com/cuda-gpus))
* Anaconda ([download here](https://www.anaconda.com/products/distribution))
* developed on Windows 10

# Installation
* clone the Stable UNet repository
* conda create --name UNet python==3.9.6
* conda activate UNet
* cd ../UNet/environment
* pip install -r requirements.txt

## Executing program
set parameters and run in Areal_predict_framework.py

## Authors
* [Benjamin Stöckigt](https://github.com/benjaminstoeckigt)
* [Shadi Ghantous](https://github.com/Shadiouss)

## Acknowledgments
Inspiration, code snippets, etc.

* [fastai](https://www.fast.ai/)
* [fastai documentation](https://docs.fast.ai/)
* [UNet tutorial by Deep Learning Berlin](https://deeplearning.berlin/satellite%20imagery/computer%20vision/fastai/2021/02/17/Building-Detection-SpaceNet7.html)
* [UNet adjustable input-channels tutorial by Navid Panchi](https://github.com/navidpanchi/N-Channeled-Input-UNet-Fastai/blob/master/N-Channeled-Input-UNet%20.ipynb)
* [UNet paper](https://arxiv.org/abs/1505.04597)
