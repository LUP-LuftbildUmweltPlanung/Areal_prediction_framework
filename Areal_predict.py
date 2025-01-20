import os
import sys
# Add the paths to the system path so Python can find the modules
sys.path.append(os.path.abspath('Data_acquisition'))
sys.path.append(os.path.abspath('UNet'))
import time
import importlib
import logging
import multiprocessing
import wms_saveraster as wms_saveraster
import download_by_shape_functions as func
from create_tiles_unet import load_json_params
from predict_deepforest import process_all_tif_files_in_folder
from deepforest import main
from PIL import Image
# Increase the pixel limit
Image.MAX_IMAGE_PIXELS = None

# Define the method
model_usage = "deepforest"  # "UNet" or "sam2"

# if the model is UNet set the params:
if model_usage == "UNet":
    predict_module = importlib.import_module('predict')
    save_predictions = predict_module.save_predictions

    # Define the input parameters directly, including predict_model
    input_data = {
        'log_file': 'BB_log.txt',
        'directory_path': r'PATH',
        'predict_model': r'PATH\Beschirmung_comb_Aug_data_less_propa_transform.pkl',
        'r_aufl': None,  #  loading from JSON
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
        'img_width': None,  # loading from JSON
        'img_height': None, # loading from JSON
        'batch_size':100,
        'AOI': 'Leipzig',
        'year': '2024',
        'merge': True,
    }
    # Load the JSON parameters using predict_model from input_data and update input_data
    json_path = input_data['predict_model'].replace('.pkl', '.json')
    params = load_json_params(json_path)
    input_data['r_aufl'] = params["resolution"][0]
    input_data['img_width'] = params["patch_size"]
    input_data['img_height'] = params["patch_size"]

# If the moswl is sam2 set the params:
elif model_usage == "sam2":
    predict_sam2_module = importlib.import_module('predict_sam2')
    predict_and_save_tiles = predict_sam2_module.predict_and_save_tiles

    input_data = {
        'log_file': 'BB_log.txt',
        'directory_path': r'PATH',
        'predict_model': r'N:\MnD\models\sam2_fine_tune_beschirmung_30_epochs\model_sam2_fine_tune_beschirmung_30_epochs_best.torch',
        'r_aufl': 0.5, 
        'meta_calc': True,
        'wms_calc': True,
        'wms_ad': 'https://sg.geodatenzentrum.de/wms_dop__14152289-bb6b-bcbb-93d7-74602cfa13d6?request=GetCapabilities&service=WMS&',
        'layer': 'rgb',
        'layer2': 'None',
        'wms_ad_meta': 'http://sg.geodatenzentrum.de/wms_info?',
        'layer_meta': 'dop',
        'state': False,
        'img_width': 400,  
        'img_height': 400, 
        'batch_size':100,
        'AOI': 'Leipzig',
        'year': '2024',
        'merge': True,
    }

# If the model is deepforest set the params:
elif model_usage == "deepforest":
    # Define input data for DeepForest
    input_data = {
        'log_file': 'BB_log.txt',
        'directory_path': r'PATH', # The folder where the shapefile located
        'predict_model': r'PATH\best_model-epoch11-val_classification0.1530.ckpt', # The path to the model usage
        'r_aufl': 0.2, # define the required resolution to download 
        'meta_calc': True, # if you want to save also meta data 
        'wms_calc': True, # if you want to save dop images
        'wms_ad': 'https://isk.geobasis-bb.de/mapproxy/dop20_2019_2021/service/wms?request=GetCapabilities&service=WMS', # WMS for whole Germany
        'layer': 'dop20_bebb_2019_2021_farbe', # the name of layer
        'layer2': 'None', # if there is another layer
        'wms_ad_meta': 'https://isk.geobasis-bb.de/ows/aktualitaeten_wms?', # Meta data
        'layer_meta': 'bb_dop-19-21_info', # layer name
        'state': False, # just for Brandenburg
        'img_width': 400, 
        'img_height': 400,
        'batch_size': 100, # using with merge function , define how many images can merge in batches, 'depending on how many tiles you have' 
        'AOI': 'Leipzig', # to give the name of saved predicted file
        'year': '2024', # to give the year of saved predicted file
        'merge': True, # if you want to merge the predicted files or keep them as tiles
        "folder_path": r"PATH", # the path to the shapefile which need to download the tiles and predict
        "savedir": r"PATH", # the path to save the predicted result
        "small_tiles": True, # if True, the prediction process will divide the whole image into small tiles then apply predict function.
        "patch_size": 400, # if ("small_tiles": True) then you have to define the size of the tiles "recommended: 400"
        "patch_overlap": 0.15,  #if ("small_tiles": True) then Overlap percentage between adjacent predicted tiles to merge all predicted tiles
        "thresh": 0.2 # Confidence score threshold for filtering predictions "recommended not more 0.4"
    }
    def run_predict(args_predict):
        model_path = args_predict["predict_model"]
        model = main.deepforest.load_from_checkpoint(model_path)  # Load the model
        # Force NMS threshold override
        model.config["nms_thresh"] = args_predict["thresh"]  # Overwrite config
        model.nms_thresh = args_predict["thresh"]  # Overwrite model attribute
        print(f"Forced NMS threshold: {model.nms_thresh}")
        process_all_tif_files_in_folder(
            model=model,
            folder_path=args_predict["folder_path"],
            savedir=args_predict["savedir"],
            small_tiles=args_predict["small_tiles"],
            patch_size=args_predict["patch_size"], 
            patch_overlap=args_predict["patch_overlap"], 
            thresh=args_predict["thresh"] 
        )

if __name__ == "__main__":
    # Set up logging
    log_file = "iterate_wms_log.txt"
    main_log = func.config_logger("debug", log_file)
    starttime = time.time()
    try:
        # Process WMS tiles
        wms_saveraster.main(input_data)
        # Define output WMS path
        output_wms_path = func.create_directory(input_data['directory_path'], "output_wms")
        predict_path = os.path.join(output_wms_path, "dop")

        if model_usage == "UNet":
            save_predictions(
                predict_model=input_data['predict_model'],
                predict_path=predict_path,
                merge=input_data['merge'],
                AOI=input_data['AOI'],
                year=input_data['year'],
                class_zero=True
            )
            
        elif model_usage == "sam2":
            predict_and_save_tiles(
                input_folder=predict_path,
                model_path=input_data['predict_model'],
                merge=input_data['merge'],
                AOI=input_data['AOI'],
                year=input_data['year']
            )
            
        elif model_usage == "deepforest":
            print("Start predict")
            run_predict(input_data)

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
