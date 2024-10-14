import os
import sys
import time
import importlib
import logging

# Add the paths to the system path so Python can find the modules
sys.path.append(os.path.abspath('Data_acquisition'))
sys.path.append(os.path.abspath('UNet'))

import wms_saveraster as wms_saveraster
import download_by_shape_functions as func
from create_tiles_unet import load_json_params

# Define the method
model_usage = "sam2"  # "UNet" or "sam2"

# Dynamically import based on model usage
if model_usage == "UNet":
    predict_module = importlib.import_module('predict')
    save_predictions = predict_module.save_predictions

    # Define the input parameters directly, including predict_model
    input_data = {
        'log_file': 'BB_log.txt',
        'directory_path': r'\\qnap3\projekte3b\MnD\projects\2024_09_18_Leipzig_CanopyCover',
        'predict_model': r'H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\model_path\Beschirmung_comb_Aug_data_less_propa_transform\Beschirmung_comb_Aug_data_less_propa_transform.pkl',
        'r_aufl': None,  # We will set this later after loading the JSON
        'meta_calc': True,
        'wms_calc': True,
        # 'wms_ad': 'https://sg.geodatenzentrum.de/wms_dop__14152289-bb6b-bcbb-93d7-74602cfa13d6?request=GetCapabilities&service=WMS&',
        # 'layer': 'rgb',
        'wms_ad': 'https://geodienste.sachsen.de/wms_geosn_dop-rgb/guest?REQUEST=GetCapabilities&SERVICE=WMS&',
        'layer': 'sn_dop_020',
        'layer2': 'None',
        # 'wms_ad_meta': 'http://sg.geodatenzentrum.de/wms_info?',
        # 'layer_meta': 'dop',
        'wms_ad_meta': 'https://geodienste.sachsen.de/wms_geosn_dop-rgb/guest?REQUEST=GetCapabilities&SERVICE=WMS&',
        'layer_meta': 'sn_dop_020_info',
        'state': False,
        'img_width': None,  # We will set this later after loading the JSON
        'img_height': None, # We will set this later after loading the JSON
        'batch_size':1000,
        'AOI': 'Leipzig',
        'year': '2024',
        'merge': True,
    }
    # Load the JSON parameters using predict_model from input_data
    json_path = input_data['predict_model'].replace('.pkl', '.json')
    params = load_json_params(json_path)

    # Update input_data with the values from JSON
    input_data['r_aufl'] = params["resolution"][0]
    input_data['img_width'] = params["patch_size"]
    input_data['img_height'] = params["patch_size"]
    # sam2 model process
elif model_usage == "sam2":
    predict_sam2_module = importlib.import_module('predict_sam2')
    predict_and_save_tiles = predict_sam2_module.predict_and_save_tiles

    input_data = {
        'log_file': 'BB_log.txt',
        'directory_path': r'shadi_test',
        'predict_model': r'N:\MnD\models\sam2_fine_tune_beschirmung_30_epochs\model_sam2_fine_tune_beschirmung_30_epochs_best.torch',
        'r_aufl': 0.5,  # We will set this later after loading the JSON
        'meta_calc': True,
        'wms_calc': True,
        'wms_ad': 'https://sg.geodatenzentrum.de/wms_dop__14152289-bb6b-bcbb-93d7-74602cfa13d6?request=GetCapabilities&service=WMS&',
        'layer': 'rgb',
        'layer2': 'None',
        'wms_ad_meta': 'http://sg.geodatenzentrum.de/wms_info?',
        'layer_meta': 'dop',
        'state': False,
        'img_width': 400,  # We will set this later after loading the JSON
        'img_height': 400, # We will set this later after loading the JSON
        'batch_size':1000,
        'AOI': 'Leipzig',
        'year': '2024',
        'merge': True,
    }

# Set up logging
log_file = "iterate_wms_log.txt"
main_log = func.config_logger("debug", log_file)

starttime = time.time()

try:
    # Process the WMS tiles
    wms_saveraster.main(input_data)

    # After processing, run the predictions
    output_wms_path = func.create_directory(input_data['directory_path'], "output_wms")

    # Define predict_path now that output_wms_path is available
    predict_path = os.path.join(output_wms_path, "dop")


    if model_usage == "UNet":
        # Run the prediction function
        save_predictions(
            predict_model=input_data['predict_model'],
            predict_path=predict_path,
            regression=False,  # Assuming default values for these parameters
            merge=input_data['merge'],
            all_classes=False,
            specific_class=None,
            large_file=False,
            AOI=input_data['AOI'],
            year=input_data['year'],
            validation_vision=False,
            class_zero=True
        )
    else:
        predict_and_save_tiles(
            input_folder=predict_path,
            model_path=input_data['predict_model'],
            merge=input_data['merge'],
            AOI=input_data['AOI'],
            year=input_data['year']
        )

except KeyError as ke:
    main_log.error(f"KeyError: {str(ke)}")
    print(f"KeyError: {str(ke)}")
except TypeError as te:
    main_log.error(f"TypeError: {str(te)}")
    print(f"TypeError: {str(te)}")
except Exception as e:
    main_log.error(f"Unexpected Error: {str(e)}")
    print(f"Unexpected Error: {str(e)}")

endtime = time.time()
main_log.info(f"All done! Execution time: {endtime - starttime} seconds")
print(f"All done! Execution time: {endtime - starttime} seconds")


