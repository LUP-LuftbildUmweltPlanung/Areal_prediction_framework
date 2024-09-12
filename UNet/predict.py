import glob
import os
import warnings
import numpy as np
import torch
import time
from tqdm import tqdm
from pathlib import Path
from osgeo import gdal
from fastai.learner import load_learner
from sklearn.metrics import confusion_matrix, classification_report
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# save the predicted tiles
def store_tif(output_folder, output_array, dtype, geo_transform, geo_proj, nodata_value, class_zero=False):
    """Stores a tif file in a specified folder."""
    driver = gdal.GetDriverByName('GTiff')

    if len(output_array.shape) == 3:
        out_ds = driver.Create(str(output_folder), output_array.shape[2], output_array.shape[1], output_array.shape[0],
                               dtype)
    else:
        out_ds = driver.Create(str(output_folder), output_array.shape[1], output_array.shape[0], 1, dtype)
    out_ds.SetGeoTransform(geo_transform)

    out_ds.SetProjection(geo_proj)

    if class_zero:
        # Process the output array to handle class definitions
        processed_array = np.where(output_array == 0, nodata_value, output_array - 1)  # Class 0 as NaN and decrement other classes by 1
    else:
        processed_array = output_array


    if len(processed_array.shape) == 3:
        for b in range(processed_array.shape[0]):
            out_ds.GetRasterBand(b + 1).WriteArray(processed_array[b])
    else:
        out_ds.GetRasterBand(1).WriteArray(processed_array)

    # Loop through the image bands to set nodata
    if nodata_value is not None:
        for i in range(1, out_ds.RasterCount + 1):
            # Set the nodata value of the band
            out_ds.GetRasterBand(i).SetNoDataValue(nodata_value)

    out_ds.FlushCache()
    out_ds = None


def merge_tiles_using_vrt(input_folder, output_file):
    # Get a list of all .tif files in the output folder
    tiff_files = glob.glob(os.path.join(input_folder, "*.tif"))

    if not tiff_files:
        print("No TIFF files found in the directory.")
        return

    # Build a VRT (Virtual Raster Table) from the list of files
    vrt_options = gdal.BuildVRTOptions(separate=False)  # separate=False means no multi-banding
    vrt = gdal.BuildVRT("", tiff_files, options=vrt_options)

    if vrt is None:
        print("Failed to create VRT.")
        return

    # Translate the VRT to a GeoTIFF format with LZW compression and NoData value handling
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=["COMPRESS=LZW"],  # Apply LZW compression
        noData=None  # Set the NoData value to None 
    )

    gdal.Translate(output_file, vrt, options=translate_options)

    print(f"Merged {len(tiff_files)} tiles into {output_file} with LZW compression and NoData set to None.")


def process_and_merge_predictions(output_folder, merge=False):
    # Convert output_folder to a Path object if it isn't already
    output_folder = Path(output_folder)

    # Assuming you already have code that predicts and saves individual tiles in `output_folder`

    # After predictions are saved, check if merging is required
    if merge:
        print("Start merging the predicted tiles...")

        # Get the name of the folder (without the full path)
        folder_name = output_folder.name  # This gets the last part of the folder name

        # If the merge function expects a string, convert it explicitly
        folder_name_str = str(folder_name)  # Ensure it is a string

        # Define the output path for the merged file using the folder name
        merged_output_file = output_folder.parent / f"{folder_name_str}_merged_output.tif"

        # Convert merged_output_file to string if required by external functions
        merged_output_file_str = str(merged_output_file)  # Ensure it is a string

        # Call the merge function to combine all tiles into one .tif file
        merge_tiles_using_vrt(str(output_folder), merged_output_file_str)  # Ensure paths are passed as strings

        print(f"Merged tiles saved to: {merged_output_file_str}")


# create valid figures
def plot_valid_predict(output_folder, predict_path, regression=False, merge=False, class_zero=False):
    if merge:
        raise ValueError("It's not possible to calculate the confusion matrix with merged tiles")
    elif regression:
        raise ValueError("This function is just for classification problems")

    # Create a new folder to save the figures
    valid_path = os.path.join(output_folder, "Valid_figures")
    os.makedirs(valid_path, exist_ok=True)

    # Replace the last part of the truth_label path
    truth_label = predict_path.replace('img_tiles', 'mask_tiles')

    y_true = []
    y_pred = []

    for file_name in os.listdir(output_folder):
        if file_name.endswith('.tif'):
            pred_path = os.path.join(output_folder, file_name)
            true_path = os.path.join(truth_label, file_name)

            with rasterio.open(pred_path) as src_pred:
                pred_data = src_pred.read(1).astype(np.int64)  # Assuming single band for class labels

            with rasterio.open(true_path) as src_true:
                true_data = src_true.read(1).astype(np.int64)  # Assuming single band for class labels

            # Determine the most frequent class in the tile
            pred_class = np.argmax(np.bincount(pred_data.flatten()))
            true_class = np.argmax(np.bincount(true_data.flatten()))

            if class_zero:
                true_class = true_class[true_class != 0] - 1


            y_true.append(true_class)
            y_pred.append(pred_class)

    if not y_true or not y_pred:
        raise ValueError("No valid tiles found for evaluation")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, zero_division=1)

    # Plot the classification report and get class names
    report_data = []
    class_names = []
    lines = class_report.split('\n')
    for line in lines[2:-3]:  # Extract just the values
        row_data = line.split()
        if len(row_data) < 5:  # Check if the row_data has the expected number of elements
            continue
        class_names.append(row_data[0])
        row = {
            'class': row_data[0],
            'precision': float(row_data[1]),
            'recall': float(row_data[2]),
            'f1_score': float(row_data[3]),
            'support': int(float(row_data[4]))
        }
        report_data.append(row)

    dataframe = pd.DataFrame.from_dict(report_data)

    plt.figure(figsize=(10, 7))
    sns.heatmap(dataframe.set_index('class'), annot=True, fmt='.2f', cmap='crest')
    plt.title('Classification Report')
    classification_report_path = os.path.join(valid_path, "classification_report.png")
    plt.savefig(classification_report_path)
    plt.show()

    # Plot the confusion matrix with class names
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='crest', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(valid_path, "Confusion_Matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.show()

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(class_report)

    return cm, class_report


def save_predictions(predict_model, predict_path, regression, merge=False, all_classes=False, specific_class=None,
                     large_file=False, AOI=None, year=None, validation_vision=True, class_zero=False):
    """
    Runs a prediction on all tiles within a folder and stores predictions in the predict_tiles folder

    Parameters:
    -----------
        learn :             Unet learner containing a Unet prediction model
        path :              Path containing tiles for prediction
        regression :        If the prediction should output continuous values (else: classification)
        merge :             If predicted tiles should be merged to a single .tif file (default=False)
        all_classes :       If the prediction should contain all prediction values for all classes (default=False)
        specific_class :    Only prediction values for this specific class will be stored (default=None)
    """

    learn = load_learner(Path(predict_model))

    path = Path(predict_path)

    # Define the path as the current directory
    output_folder = path.parent / ('predicted_tiles_' + Path(predict_model).stem)

    model_name = os.path.basename(predict_model).split('.')[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    tiles = glob.glob(str(path) + "\\*.tif")

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(f'Started at: {current_time}')
    # for i in range(len(tiles)):
    for i in tqdm(range(len(tiles)), desc='Processing tiles'):
        # print(f'Current progress: {i}/{len(tiles)}')
        tile_preds = learn.predict(Path(tiles[i]), with_input=False)
        class_lst = []

        if regression:
            for cl in range(len(tile_preds[1])):
                class_lst.append(tile_preds[1][cl])
        else:
            for cl in range(len(tile_preds[2])):
                class_lst.append(tile_preds[2][cl])

        class_lst = torch.stack(class_lst)

       # else:
        if regression:
            pass
        elif all_classes:
            pass
        elif specific_class is None:
            # for decoded argmax value
            class_lst = class_lst.argmax(axis=0)
        else:
            # for probabilities of specific class [1] -> klasse 1
            class_lst = class_lst[specific_class]
        img_ds_proj = gdal.Open(str(tiles[i]))
        geotrans = img_ds_proj.GetGeoTransform()
        geoproj = img_ds_proj.GetProjection()

        if "float" in str(class_lst.dtype):
            dtype = gdal.GDT_Float32
        else:
            dtype = gdal.GDT_Byte

        if large_file and np.max(class_lst.numpy()) <= 1 and (all_classes or specific_class):
            class_lst = class_lst.numpy()
            class_lst *= ((128 / 4) - 1)
            class_lst = np.around(class_lst).astype(np.int8)
            dtype = gdal.GDT_Byte
            store_tif(str(output_folder) + "\\" + os.path.basename(tiles[i]), class_lst, dtype, geotrans, geoproj,
                      None, class_zero)
        else:
            store_tif(str(output_folder) + "\\" + os.path.basename(tiles[i]), class_lst.numpy(), dtype, geotrans,
                      geoproj, None, class_zero)
    if validation_vision:
        plot_valid_predict(output_folder, predict_path, regression, merge, class_zero)

    if merge:
        process_and_merge_predictions(output_folder, merge)

        print(f"Prediction stored in {output_folder}.")
