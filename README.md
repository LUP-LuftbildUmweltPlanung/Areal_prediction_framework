# Areal Prediction Framework

An automated framework for downloading raster data and acquisition dates from WMS servers or geoportals, followed by prediction the mask over the WMS tiles using a UNet model.

## Description

This repository integrates two submodules:

1. **[Data Acquisition](https://github.com/LUP-LuftbildUmweltPlanung/Data_acquisition):** This module contains Python scripts to download image data and their corresponding acquisition dates from specified WMS servers or geoportals using one or more shapefiles.

2. **[UNet](https://arxiv.org/abs/1505.04597):** This module provides the code necessary to run a UNet model, based on the Dynamic UNet implementation from fastai, utilizing the PyTorch Deep Learning framework. The UNet model is used for image segmentation (pixel-wise classification).

   
* Finally the output will be three different folder: **"WMS_tiles", "Meta_files", "Predicted tiles"** or
Three different files **"WMS_merged.tif", "Meta_merged.tif", "Predicted_merged.tif"**.

## Getting Started

### Dependencies

- GDAL
- PyTorch-Fastai
- SciPy
- CUDA-capable GPU ([CUDA GPUs Overview](https://developer.nvidia.com/cuda-gpus))
- Anaconda ([Download Anaconda](https://www.anaconda.com/products/distribution))

**Note:** This project was developed on Windows 10.

### Installation

1. Clone the stable UNet repository.
2. Create a new environment:
    ```bash
    conda create --name UNet python=3.9.6
    ```
3. Activate the environment:
    ```bash
    conda activate UNet
    ```
4. Navigate to the environment folder:
    ```bash
    cd ../UNet/environment
    ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Executing the Program

Set the desired parameters in `Areal_predict.py` and run the script.

## Authors

- [Benjamin Stöckigt](https://github.com/benjaminstoeckigt)
- [Shadi Ghantous](https://github.com/Shadiouss)

## Acknowledgments

This project was inspired by and built upon the work of many contributors. Special thanks to:

- [fastai](https://www.fast.ai/)
- [fastai Documentation](https://docs.fast.ai/)
- [UNet Tutorial by Deep Learning Berlin](https://deeplearning.berlin/satellite%20imagery/computer%20vision/fastai/2021/02/17/Building-Detection-SpaceNet7.html)
- [Adjustable Input-Channels UNet Tutorial by Navid Panchi](https://github.com/navidpanchi/N-Channeled-Input-UNet-Fastai/blob/master/N-Channeled-Input-UNet%20.ipynb)
- [UNet Paper](https://arxiv.org/abs/1505.04597)

---

This revision provides a more structured, concise, and readable format for your README, which should help users quickly understand and navigate your project.
