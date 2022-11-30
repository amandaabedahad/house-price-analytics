"""
Main script that scrapes hemnet of data, and processes it into wanted format.
"""
import copy
# -*- coding: utf-8 -*-
import os
from os.path import exists
from tqdm import tqdm
import pandas as pd
from scrape_hemnet import main_scrape_hemnet
import data_process_functions
import logging
import re
import numpy as np
from ML_models import update_ml_model
from sql_queries import get_pandas_from_database, insert_to_database
from dotenv import load_dotenv


def get_nr_new_samples_since_last_ml_update(logger_path, nr_listings_trigger_ml_update, target_word):
    sum_new_listings = 0
    for line in reversed(list(open(logger_path))):
        match_nr_listings = re.search(f"\d+ {target_word}", line)
        match_updated_model = re.search("updating machine learning model", line)
        if match_nr_listings:
            nr_new_listings = re.split(' ', match_nr_listings[0])[0]
            sum_new_listings += int(nr_new_listings)
        elif match_updated_model:
            return sum_new_listings, False

        if sum_new_listings >= nr_listings_trigger_ml_update:
            return sum_new_listings, True

    # if whole logger file is empty, we come here
    return sum_new_listings, False


load_dotenv('.env')

if __name__ == "__main__":
    path_shp_file = "geospatial_data_polygons_areas/JUR_PRIMÄROMRÅDEN_XU_region.shp"

    path_log_file = "logs/logging_file.txt"
    if not exists(path_log_file):
        os.makedirs(path_log_file.split('/')[0], exist_ok=True)
        f = open(path_log_file, "w+")
    logging.basicConfig(filename=path_log_file, filemode="a", level=logging.INFO, format='%(asctime)s - %(levelname)s '
                                                                                         '- %(message)s')
    my_logger = logging.getLogger("my_logger")
    my_logger.info("\n Main script started")

    hemnet_data_raw = get_pandas_from_database("raw_data")

    new_listings_raw_data = main_scrape_hemnet(hemnet_data_raw, my_logger)

    new_listings_raw_data = new_listings_raw_data.sort_index()
    if new_listings_raw_data.empty:
        print('No new samples found when scraped web page')
        my_logger.info("0 new listings that needs to be processed - script exited")
        exit()

    nr_new_samples = new_listings_raw_data.shape[0]
    data_processed = get_pandas_from_database("processed_data")

    nr_samples_data_processed = data_processed.shape[0]

    my_logger.info(f"{nr_new_samples} new listings that needs to be processed")
    print(f'{nr_new_samples} new samples to be processed')
    pbar = tqdm(total=nr_new_samples)

    new_listings_to_be_processed = copy.deepcopy(new_listings_raw_data)
    new_listings_to_be_processed["region"] = new_listings_to_be_processed["region"].apply(
        data_process_functions.clean_region_sample)
    new_listings_to_be_processed["address"] = new_listings_to_be_processed["address"].apply(
        data_process_functions.clean_address_sample)

    location_info = new_listings_to_be_processed["address"].apply(
        lambda x: data_process_functions.get_long_lat(x, pbar=pbar))
    new_listings_to_be_processed["latitude"], new_listings_to_be_processed["longitude"], new_listings_to_be_processed[
        "post_code"] = zip(*location_info)

    processed_new_data = data_process_functions.map_address_to_area(new_listings_to_be_processed, path_shp_file)
    processed_new_data = processed_new_data.sort_index()

    # append rows with corresponding listing_id to listing_information table
    data_listing_information = get_pandas_from_database("listing_information")
    max_listing_id = data_listing_information["listing_id"].max()

    # Create new unique listing_ids
    new_listing_ids = pd.DataFrame(
        np.arange(max_listing_id + 1,
                  max_listing_id + nr_new_samples + 1), columns=["listing_id"], index=processed_new_data.index)

    # join on index --> assured to have same listing id for every index. Index is same for same listings
    new_listings_raw_data = new_listings_raw_data.join(new_listing_ids)
    processed_new_data = processed_new_data.join(new_listing_ids)

    # Checks if it is time to retrain the ML-model
    new_listings_since_last_ml_update, reached_trigger_amount = get_nr_new_samples_since_last_ml_update(
        path_log_file, nr_listings_trigger_ml_update=200, target_word="new listings"
    )

    my_logger.info(f"{new_listings_since_last_ml_update} listings since last ML-update")

    if reached_trigger_amount:
        # concat newly extracted data and data from data base. used
        hemnet_data_processed_all = pd.concat([processed_new_data, data_processed], ignore_index=True)
        update_ml_model(hemnet_data_processed_all, my_logger)

    # Insert new data into tables in database
    insert_to_database(new_listing_ids, "listing_information")
    insert_to_database(new_listings_raw_data, "raw_data")
    insert_to_database(processed_new_data, "processed_data")

    # since all the new listings have not been in training!!!!!!
    new_listing_ids["listing_in_train_set"] = 0
    insert_to_database(new_listing_ids, "listing_train_or_test_set")

    my_logger.info("Data scraping and processing finished.")
