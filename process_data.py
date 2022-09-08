from os.path import exists

import pandas as pd
from fake_useragent import UserAgent
from geopy.geocoders import Nominatim
import re
from tqdm import tqdm

ua = UserAgent()
header = {
    "User-Agent": ua.random
}


def clean_address_sample(sample):
    match = re.search("([A-Ö|a-ö]+.)*(\d+)*(\w)*", sample)
    str_match = match[0]
    return str_match


def clean_region_sample(sample):
    match = re.split("\-", sample)[-1]
    return match


def get_long_lat(sample, city='Göteborgs kommun'):  # TODO: change this hardcoded city and find more efficient calcs
    address = sample + ', ' + city
    location = geolocator.geocode(address)
    pbar.update(1)
    if location is None:
        return None, None
    return location.latitude, location.longitude


if __name__ == '__main__':
    geolocator = Nominatim(user_agent=str(header))

    path_to_file_raw_data = "hemnet_data/hemnet_house_data_raw.csv"
    path_to_file_processed_data = "hemnet_data/hemnet_house_data_processed.csv"

    data_raw = pd.read_csv(path_to_file_raw_data)
    nr_samples_data_raw = data_raw.shape[0]
    if not exists(path_to_file_processed_data):
        data_processed = None
        nr_samples_data_processed = 0
    else:
        data_processed = pd.read_csv(path_to_file_processed_data)
        nr_samples_data_processed = data_processed.shape[0]

    diff_raw_processed = nr_samples_data_raw - nr_samples_data_processed
    pbar = tqdm(total=diff_raw_processed)
    if diff_raw_processed == 0:
        print('No new samples in dataset raw compared to processed')
        exit()
    else:
        print(f'{diff_raw_processed} new samples to be processed')

    data_to_be_processed = data_raw.loc[: diff_raw_processed - 1, :]
    data_to_be_processed["region"] = data_to_be_processed["region"].apply(clean_region_sample)
    data_to_be_processed["address"] = data_to_be_processed["address"].apply(clean_address_sample)

    coordinates = data_to_be_processed["address"].apply(get_long_lat)
    data_to_be_processed["latitude"], data_to_be_processed["longitude"] = zip(*coordinates)

    data = pd.concat([data_to_be_processed, data_processed], ignore_index=True)
    data.to_csv("hemnet_data/hemnet_house_data_processed.csv", index=False)
