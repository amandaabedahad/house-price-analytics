"""
Main script that scrapes hemnet of data, and processes it into wanted format.
"""
from os.path import exists
from tqdm import tqdm
import pandas as pd
from scrape_hemnet import main_scrape_hemnet
from process_data import process_data
from map_address_to_area import map_address_to_area

if __name__ == "__main__":
    path_to_hemnet_data_raw = "hemnet_data/hemnet_house_data_raw.csv"
    path_to_hemnet_data_processed = "hemnet_data/hemnet_house_data_processed.csv"
    path_shp_file = "geospatial_data_polygons_areas/JUR_PRIMÄROMRÅDEN_XU_region.shp"

    raw_hemnet_data = main_scrape_hemnet(path_to_hemnet_data_raw)
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
        exit()

    pbar = tqdm(total=diff_raw_processed)
    print(f'{diff_raw_processed} new samples to be processed')

    new_data = raw_hemnet_data.loc[: diff_raw_processed - 1, :]
    processed_new_data = process_data(new_data, pbar)

    processed_new_data = map_address_to_area(processed_new_data, path_shp_file)

    hemnet_data = pd.concat([processed_new_data, data_processed], ignore_index=True)

    hemnet_data.to_csv("hemnet_data/hemnet_house_data_processed.csv", index=False)



