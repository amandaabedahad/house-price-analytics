"""
Main script that scrapes hemnet of data, and processes it into wanted format.
"""
import os
from os.path import exists
from tqdm import tqdm
import pandas as pd
from scrape_hemnet import main_scrape_hemnet
import data_process_functions
import logging
from ML_models import update_ml_model


if __name__ == "__main__":
    path_to_hemnet_data_raw = "hemnet_data/hemnet_house_data_raw.csv"
    path_to_hemnet_data_processed = "hemnet_data/hemnet_house_data_processed.csv"
    path_shp_file = "geospatial_data_polygons_areas/JUR_PRIMÄROMRÅDEN_XU_region.shp"

    path_log_file = "logs/logging_file.txt"
    if not exists(path_log_file):
        os.makedirs(path_log_file.split('/')[0], exist_ok=True)
        f = open(path_log_file, "w+")
    logging.basicConfig(filename=path_log_file, filemode="a", level=logging.INFO, format='%(asctime)s - %(levelname)s '
                                                                                         '- %(message)s')
    my_logger = logging.getLogger("my_logger")
    my_logger.info("\n Main script started")
    raw_hemnet_data = main_scrape_hemnet(path_to_hemnet_data_raw, my_logger)
    nr_samples_raw_data = raw_hemnet_data.shape[0]
    if not exists(path_to_hemnet_data_processed):
        data_processed = None
        nr_samples_data_processed = 0
    else:
        data_processed = pd.read_csv(path_to_hemnet_data_processed)
        nr_samples_data_processed = data_processed.shape[0]

    diff_raw_processed = nr_samples_raw_data - nr_samples_data_processed

    if diff_raw_processed == 0:
        print('No new samples in dataset raw compared to processed')
        my_logger.info("0 new listings that needs to be processed - script exited")
        exit()
    my_logger.info(f"{diff_raw_processed} new listings that needs to be processed")
    pbar = tqdm(total=diff_raw_processed)
    print(f'{diff_raw_processed} new samples to be processed')

    new_data = raw_hemnet_data.iloc[:diff_raw_processed].copy()
    new_data["region"] = new_data["region"].apply(data_process_functions.clean_region_sample)
    new_data["address"] = new_data["address"].apply(data_process_functions.clean_address_sample)

    location_info = new_data["address"].apply(lambda x: data_process_functions.get_long_lat(x, pbar=pbar))
    new_data["latitude"], new_data["longitude"], new_data["post_code"] = zip(*location_info)

    processed_new_data = data_process_functions.map_address_to_area(new_data, path_shp_file)

    # We don't want the address mapping to delete/add listings.
    assert processed_new_data.shape[0] == new_data.shape[0]

    hemnet_data = pd.concat([processed_new_data, data_processed], ignore_index=True)
    # update_ml_model(hemnet_data, my_logger)
    hemnet_data.to_csv("hemnet_data/hemnet_house_data_processed.csv", index=False)

    my_logger.info("Data scraping and processing finished.")

