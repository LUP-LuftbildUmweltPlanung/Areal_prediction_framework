import os
import sys
import time
import logging

# Add the paths to the system path so Python can find the modules
sys.path.append(os.path.abspath('Data_acquisition'))
sys.path.append(os.path.abspath('UNet'))

import os
import wms_saveraster as wms_saveraster
import download_by_shape_functions as func
import time
from predict import save_predictions
from create_tiles_unet import load_json_params

# Define the input parameters directly, including predict_model
input_data = {
    'log_file': 'BB_log.txt',
    'directory_path': r'W:\2023_RailVitaliTree\Prozessierung\SubPixel_RLP\small_dataset',
    'predict_model': r'H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\model_path\Beschirmung_comb_Aug_data_less_propa_transform\Beschirmung_comb_Aug_data_less_propa_transform.pkl',
    'r_aufl': None,  # We will set this later after loading the JSON
    'meta_calc': True,
    'wms_calc': True,
    'wms_ad': 'https://sg.geodatenzentrum.de/wms_dop__14152289-bb6b-bcbb-93d7-74602cfa13d6?request=GetCapabilities&service=WMS&',
    'layer': 'rgb',
    'layer2': 'None',
    'wms_ad_meta': 'http://sg.geodatenzentrum.de/wms_info?',
    'layer_meta': 'dop',
    'state': False,
    'img_width': None,  # We will set this later after loading the JSON
    'img_height': None, # We will set this later after loading the JSON
    'batch_size':1000,
    'AOI': 'Potsdam',
    'year': '2018',
    'merge': True,
}

# Load the JSON parameters using predict_model from input_data
json_path = input_data['predict_model'].replace('.pkl', '.json')
params = load_json_params(json_path)

# Update input_data with the values from JSON
input_data['r_aufl'] = params["resolution"][0]
input_data['img_width'] = params["patch_size"]
input_data['img_height'] = params["patch_size"]

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
