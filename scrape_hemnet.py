# Import library
import pandas as pd
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
from tqdm import tqdm
from os.path import exists
from geopy.geocoders import Nominatim
from datetime import date
import locale
from datetime import datetime

ua = UserAgent()
header = {
    "User-Agent": ua.random
}
locale.setlocale(locale.LC_ALL, 'sv_SE')
PAGES_TO_SEARCH = 50
HOUSE_CARDS_PER_PAGE = 50
CHAR_TO_LOOK_FOR_ADDRESS = ['-', '/', 'Vån', 'vån', 'BV', 'lägenhet']
CHAR_TO_LOOK_FOR_REGION = ['-', '/']


def clean_single_data(data_to_clean, char_to_find):
    for char in char_to_find:
        if char in data_to_clean:
            find_index_char = data_to_clean.find(char)
            data_to_clean = data_to_clean[:find_index_char]
    return data_to_clean


def parse_html(row, loaded_hemnet_df):
    # All different housing types have these properties:
    address_raw = row.find("h2", class_="sold-property-listing__heading qa-selling-price-title").text.strip()
    address = clean_single_data(address_raw, CHAR_TO_LOOK_FOR_ADDRESS)
    housing_type = row.find("span", class_="svg-icon__fallback-text").text.strip()
    region_raw = row.find("div", class_="sold-property-listing__location").text.split('\n')[7].strip(' ,')
    region = clean_single_data(region_raw, CHAR_TO_LOOK_FOR_REGION)
    city = row.find("div", class_="sold-property-listing__location").text.split("\n")[8].strip()
    final_price = int(
        row.find("div", class_="sold-property-listing__price").text.split('\n')[2].strip(' Slutpris kr').replace(
            '\xa0', ''))
    sold_date = row.find("div", class_="sold-property-listing__price").text.split('\n')[5].strip(' Såld')

    sold_date = str(datetime.strptime(sold_date, "%d %B %Y").date())

    if housing_type == 'Övrigt':
        return None

    elif loaded_hemnet_df is not None and (loaded_hemnet_df["address"] == address).any():
        address_series = loaded_hemnet_df[["address", "sold_date"]]
        occurrences = address_series[address_series.values == address]
        if (occurrences["sold_date"] == sold_date).any():
            return None

    # These depend on each object. not included in all. Set to None if not included
    sqr_meter = row.find("div", class_="sold-property-listing__subheading sold-property-listing__area")
    rent_month = row.find("div", class_="sold-property-listing__fee")
    price_sqr_m = row.find("div", class_="sold-property-listing__price-per-m2")
    land_area = row.find("div", class_="sold-property-listing__land-area")
    nr_rooms = row.find("div", class_="sold-property-listing__subheading sold-property-listing__area")
    price_increase = row.find("div", class_="sold-property-listing__price-change")
    other_sqr_area = 0

    if land_area is not None and 'tomt' in land_area.text:
        land_area = float(land_area.text.strip(" \n m²tomt").replace('\xa0', '').replace(",", "."))
    else:
        land_area = 0

    if nr_rooms is not None and "rum" in nr_rooms.text:
        room_index = nr_rooms.text.split().index("rum") - 1
        nr_rooms = float(nr_rooms.text.split()[room_index].replace(",", "."))
    else:
        nr_rooms = None

    if sqr_meter is not None and "m²" in sqr_meter.text:
        if '+' in sqr_meter.text:
            other_sqr_area = float(sqr_meter.text.split()[2].replace(",", "."))
        sqr_meter = float(sqr_meter.text.split()[0].replace(",", "."))

    else:
        sqr_meter = None

    if rent_month is not None and "kr/mån" in rent_month.text:
        rent_month = float(rent_month.text.strip(" \n kr/mån").replace('\xa0', ''))
    else:
        rent_month = None

    if price_sqr_m is not None and "kr/" in price_sqr_m.text:
        price_sqr_m = float(price_sqr_m.text.strip(" \n kr/m²").replace('\xa0', ''))
    elif sqr_meter is not None:
        price_sqr_m = round(final_price / sqr_meter)
    else:
        price_sqr_m = None

    if price_increase is not None and "%" in price_increase.text:
        price_increase = float(price_increase.text.strip(" \n %").replace('\xa0', '').strip('±'))
    else:
        price_increase = None

    location = geolocator.geocode(address + ", " + city)

    if location is not None:
        latitude = location.latitude
        longitude = location.longitude
    else:
        latitude = None
        longitude = None

    data_series = {"address": address,
                   "housing_type": housing_type,
                   "region": region,
                   "city": city,
                   "sqr_meter": sqr_meter,
                   "nr_rooms": nr_rooms,
                   "rent_month": rent_month,
                   "final_price": final_price,
                   "sold_date": sold_date,
                   "price_increase": price_increase,
                   "price_sqr_m": price_sqr_m,
                   "land_area": land_area,
                   "other_srq": other_sqr_area,
                   "latitude": latitude,
                   "longitude": longitude
                   }

    return data_series


data = {}
path_to_hemnet_data = "hemnet_data/hemnet_house_data.csv"
nr_objects = 0
pbar = tqdm(total=PAGES_TO_SEARCH * HOUSE_CARDS_PER_PAGE)
geolocator = Nominatim(user_agent="my_request")

if not exists(path_to_hemnet_data):
    loaded_hemnet_data = None
    original_nr_objects = 0
else:
    loaded_hemnet_data = pd.read_csv(path_to_hemnet_data)
    original_nr_objects = loaded_hemnet_data.shape[0]

nr_consecutive_pages_already_documented = 0

for page in range(1, PAGES_TO_SEARCH + 1):
    # TODO: change municipality code.
    municipality_code = 17920  # for gothenburg
    url = f"https://www.hemnet.se/salda/bostader?location_ids%5B%5D={municipality_code}&page={page}"

    # if 5 pages in a row are filled with already documented objects, then finished.
    occurrence_counter = 0
    response = requests.get(url, headers=header).text
    soup = BeautifulSoup(response.encode("utf-8").decode('utf-8'), 'html.parser')
    for row in soup.find_all('a', class_="sold-property-link js-sold-property-card-link"):
        nr_objects += 1
        data_series = parse_html(row, loaded_hemnet_data)
        pbar.update(1)
        if data_series is None:
            occurrence_counter += 1
            if occurrence_counter == HOUSE_CARDS_PER_PAGE:
                nr_consecutive_pages_already_documented += 1
                break
            continue
        occurrence_counter = 0
        nr_consecutive_pages_already_documented = 0
        data[nr_objects] = data_series
    if nr_consecutive_pages_already_documented == 2:
        pbar.close()
        print("100 objects in a row already in file, assume that the remaining objects are also in file. ")
        break

pd_data_series = pd.DataFrame.from_dict(data, orient="index")

df_data = pd.concat([pd_data_series, loaded_hemnet_data], ignore_index=True)
df_data.to_csv(f"hemnet_house_data_{date.today()}.csv", index=False)

print(f"originally {original_nr_objects} objects, now {df_data.shape[0]} objects")