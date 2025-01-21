# Areal Prediction Framework

An automated framework for downloading raster data and acquisition dates from WMS servers or geoportals, followed by predicting masks over the WMS tiles using various models, including UNet, SAM2 and RetinaNet.

## Description

This repository integrates these submodules:

1. **[Data Acquisition](https://github.com/LUP-LuftbildUmweltPlanung/Data_acquisition):** This module contains Python scripts to download image data and their corresponding acquisition dates from specified WMS servers or geoportals using one or more shapefiles.

2. **[UNet](https://arxiv.org/abs/1505.04597):** This module provides the code necessary to run a UNet model, based on the Dynamic UNet implementation from fastai, utilizing the PyTorch Deep Learning framework. The UNet model is used for image segmentation (pixel-wise classification).

3. **[SAM2_1_fine_tune](https://github.com/LUP-LuftbildUmweltPlanung/SAM2_1_fine_tune/tree/main):** This module contains the code required to fine-tune the pre-trained SAM 2 model on a custom dataset, enhancing its performance for defining a tree or canopy model. Like the UNet, it is also used for image segmentation.

4. **[RetinaNet](https://arxiv.org/abs/1708.02002):** This module is based on the [DeepForest](https://github.com/weecology/DeepForest) framework to detect objects within an image. Unlike UNet and SAM2, which perform image segmentation, RetinaNet is specifically used for object detection. The model is particularly suited for high-resolution aerial imagery, such as images with a resolution of approximately 20 cm. Users can apply RetinaNet either to the entire image or to smaller tiles created by splitting the image. Images can be downloaded using the `wms_servers.py` script, and the resulting tiles can be merged to create a single, large image for more cohesive object detection.

### Output Structure
The final output of the framework is organized into:
- **Folders:**
  - `WMS_tiles` 
  - `Meta_files` 
  - `Predicted tiles`
- **Files (if merging is enabled):**
  - `WMS_merged.tif`
  - `Meta_merged.tif`
  - `Predicted_merged.tif`

![Output Example](https://github.com/user-attachments/assets/bbb1a98e-c121-4a00-b561-e871cd316373)

## Getting Started

### Dependencies

- GDAL
- PyTorch-Fastai
- SciPy
- CUDA-capable GPU ([CUDA GPUs Overview](https://developer.nvidia.com/cuda-gpus))
- Anaconda ([Download Anaconda](https://www.anaconda.com/products/distribution))

**Note:** This project was developed on Windows 10.

### Installation

#### Installation for UNet
To use the UNet model for prediction processing, follow these installation steps:
```bash
conda create --name Areal_predict_unet python=3.9.6
conda activate Areal_predict_unet
cd ../Areal_prediction_framework/environment
pip install -r requirements.txt
```

#### Installation for SAM2
If you want to use SAM2 for predicting tiles, follow these steps for installation:
```bash
# 1. Create and activate the Conda environment
conda create -n Areal_predict_sam2_1 python=3.11
conda activate Areal_predict_sam2_1

# 2. Install PyTorch and its dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Go to the main folder where the script is located
cd ../Areal_prediction_framework

# 4. Clone the SAM2 repository and rename the folder to avoid conflicts
git clone https://github.com/facebookresearch/sam2.git
mv sam2 sam2_conf

# 5. Change into the 'sam2_conf' directory and copy the 'sam2' folder to the 'sam2_1_fine_tune-main' folder
cd sam2_conf
cp -r sam2 ../sam2/

# 6. Install the SAM2 package in editable mode
pip install -e .

# 7. Navigate to the 'checkpoints' folder and download model checkpoints
cd checkpoints && download_ckpts.sh
cd ../..
cd checkpoints_sam2 && download_ckpts.sh

# 8. Go two directories up and install additional dependencies
cd environment
pip install -r requirements_sam2.txt
```

#### Installation for deepforest
To use the RetinaNet model for object detection, follow these steps:
```bash
conda create -n Areal_predict_RetinaNet python=3.11
conda activate Areal_predict_RetinaNet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd ../RetinaNet/environment
pip install -r requirements.txt
```

## Executing the Program

Set the desired parameters in `Areal_predict.py` and run the script to process the data and generate predictions.

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
- [Deepforest](https://deepforest.readthedocs.io/en/latest/)

