"""
Main script that scrapes hemnet of data, and processes it into wanted format.
"""
from os.path import exists
from tqdm import tqdm
import pandas as pd
from scrape_hemnet import main_scrape_hemnet
import data_process_functions

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

    new_data = raw_hemnet_data.iloc[:diff_raw_processed].copy()
    # TODO: change these rows to new_data.loc[:, column]. At the moment, we get warning.
    new_data["region"] = new_data["region"].apply(data_process_functions.clean_region_sample)
    new_data.loc[:, "address"] = new_data["address"].apply(data_process_functions.clean_address_sample)

    location_info = new_data["address"].apply(lambda x: data_process_functions.get_long_lat(x, pbar=pbar))
    new_data["latitude"], new_data["longitude"], new_data["post_code"] = zip(*location_info)

    processed_new_data = data_process_functions.map_address_to_area(new_data, path_shp_file)

    assert processed_new_data.shape[0] == new_data.shape[0]

    hemnet_data = pd.concat([processed_new_data, data_processed], ignore_index=True)

    hemnet_data.to_csv("hemnet_data/hemnet_house_data_processed.csv", index=False)



