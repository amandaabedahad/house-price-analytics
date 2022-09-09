from os.path import exists

import pandas as pd
from fake_useragent import UserAgent
from geopy.geocoders import Nominatim
import re
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

ua = UserAgent()
header = {
    "User-Agent": ua.random
}


def clean_address_sample(sample):
    # TODO: double check if it parses ex Hejhej 3 A correctly (space between 3 and A)
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


def dummy_encoding(column_to_be_encoded):
    dummy_enc = pd.get_dummies(column_to_be_encoded, prefix="Type")
    return dummy_enc


def interpolate_missing_data_KNN(dataframe, column):
    # interpolates by taking the average of the nearby houses, implemented using KNN regression
    # data_that_needs_interpolation =
    dataframe.dropna(subset=[column, 'longitude', "nr_rooms"], inplace=True)

    for housing_type in dataframe["housing_type"].unique():
        rslt_data = dataframe[dataframe["housing_type"] == housing_type]

        if housing_type == "Lägenhet":
            X = rslt_data[["sqr_meter", "nr_rooms", "rent_month", "final_price", "latitude", "longitude"]]
        elif housing_type == "Villa":
            X = rslt_data[["sqr_meter", "nr_rooms", "final_price", "land_area", "other_srq", "latitude", "longitude"]]
        else:
            continue
        param = {"n_neighbors": range(1, 20), "weights": ["uniform", "distance"]}
        neigh = GridSearchCV(KNeighborsRegressor(), param)
        # neigh = KNeighborsRegressor(n_neighbors=5, weights="distance")

        y = rslt_data[column]
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.33, random_state=42)
        neigh.fit(X_train, y_train)
        mse = mean_squared_error(y_test, neigh.predict(X_test))
        print(mse)
        print(neigh.best_params_, neigh.best_score_)


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

    new_data = data_raw.loc[: diff_raw_processed - 1, :]
    new_data["region"] = new_data["region"].apply(clean_region_sample)
    new_data["address"] = new_data["address"].apply(clean_address_sample)

    coordinates = new_data["address"].apply(get_long_lat)
    new_data["latitude"], new_data["longitude"] = zip(*coordinates)

    data = pd.concat([new_data, data_processed], ignore_index=True)
    
    # data = interpolate_missing_data_KNN(new_data, "price_increase")

    data.to_csv("hemnet_data/hemnet_house_data_processed.csv", index=False)
