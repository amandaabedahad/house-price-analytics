import pandas as pd
from geopy.geocoders import Nominatim
import numpy as np
import re
from tqdm import tqdm


def clean_address_sample(sample):

    match = re.search("([A-Ö|a-ö]+.)*(\d+)*(\w)*", sample)
    str_match = match[0]
    return str_match


def clean_region_sample(sample):
    match = re.split("\-", sample)[-1]
    return match


def get_long_lat(sample, city='Göteborgs kommun'):
    address = sample + ', ' + city
    location = geolocator.geocode(address)
    pbar.update(1)
    if location is None:
        return None, None
    return location.latitude, location.longitude


if __name__ == '__main__':
    geolocator = Nominatim(user_agent="my_request_1")
    path_to_file = "hemnet_data/hemnet_house_data_raw.csv"
    data = pd.read_csv(path_to_file)
    pbar = tqdm(total=data.shape[0])

    data["region"] = data["region"].apply(clean_region_sample)
    data["address"] = data["address"].apply(clean_address_sample)

    coordinates = data["address"].apply(get_long_lat)
    data["latitude"], data["longitude"] = zip(*coordinates)
    data.to_csv("hemnet/house_data_processed.csv", index=False)