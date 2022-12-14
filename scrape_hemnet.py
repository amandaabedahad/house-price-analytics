# Import library
import os
import locale
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent
from tqdm import tqdm

ua = UserAgent()
header = {
    "User-Agent": ua.random
}
if os.name == 'nt':
    locale.setlocale(locale.LC_ALL, 'sv_SE')
else:
    locale.setlocale(locale.LC_ALL, 'sv_SE.utf8')

PAGES_TO_SEARCH = 50
HOUSE_CARDS_PER_PAGE = 50


def parse_html(row, loaded_hemnet_df):
    # All different housing types have these properties:
    address = row.find("h2",
                       class_="sold-property-listing__heading qa-selling-price-title").text.strip()
    housing_type = row.find("span", class_="svg-icon__fallback-text").text.strip()
    region = row.find("div", class_="sold-property-listing__location").text.split('\n')[7].strip(
        ' ,')
    city = row.find("div", class_="sold-property-listing__location").text.split("\n")[8].strip()
    final_price = int(
        row.find("div", class_="sold-property-listing__price").text.split('\n')[2].strip(
            ' Slutpris kr').replace(
            '\xa0', ''))
    sold_date_raw = row.find("div", class_="sold-property-listing__price").text.split('\n')[
        5].strip(' Såld')

    sold_date = datetime.strptime(sold_date_raw, "%d %B %Y").date()

    specific_info_listing = pd.DataFrame.from_dict({"address": address, "sold_date": sold_date,
                                                    "final_price": final_price}, orient="index").T
    # if sold_date, address and final price already included in data --> move on to next listing
    if (specific_info_listing.values ==
        loaded_hemnet_df[["address", "sold_date", "final_price"]].values).sum(axis=1).max() == 3:
        return None
    if housing_type == 'Övrigt':
        return None

    # These depend on each object. not included in all. Set to None if not included
    sqr_meter_raw = row.find("div",
                             class_="sold-property-listing__subheading sold-property-listing__area")
    rent_month_raw = row.find("div", class_="sold-property-listing__fee")
    price_sqr_m_raw = row.find("div", class_="sold-property-listing__price-per-m2")
    land_area_raw = row.find("div", class_="sold-property-listing__land-area")
    nr_rooms_raw = row.find("div",
                            class_="sold-property-listing__subheading sold-property-listing__area")
    # TODO: change to price change? in all the other files too
    price_increase_raw = row.find("div", class_="sold-property-listing__price-change")
    other_sqr_area = 0

    if land_area_raw is not None and 'tomt' in land_area_raw.text:
        land_area = float(land_area_raw.text.strip(" \n m²tomt").replace('\xa0', '')
                          .replace(",", "."))
    else:
        land_area = 0

    if nr_rooms_raw is not None and "rum" in nr_rooms_raw.text:
        room_index = nr_rooms_raw.text.split().index("rum") - 1
        nr_rooms = float(nr_rooms_raw.text.split()[room_index].replace(",", "."))
    else:
        nr_rooms = None

    if sqr_meter_raw is not None and "m²" in sqr_meter_raw.text:
        if '+' in sqr_meter_raw.text:
            other_sqr_area = float(sqr_meter_raw.text.split()[2].replace(",", "."))
        sqr_meter = float(sqr_meter_raw.text.split()[0].replace(",", "."))

    else:
        sqr_meter = None

    if rent_month_raw is not None and "kr/mån" in rent_month_raw.text:
        rent_month = float(rent_month_raw.text.strip(" \n kr/mån").replace('\xa0', ''))
    else:
        rent_month = None

    if price_sqr_m_raw is not None and "kr/" in price_sqr_m_raw.text:
        price_sqr_m = float(price_sqr_m_raw.text.strip(" \n kr/m²").replace('\xa0', ''))
    elif sqr_meter is not None:
        price_sqr_m = round(final_price / sqr_meter)
    else:
        price_sqr_m = None

    if price_increase_raw is not None and "%" in price_increase_raw.text:
        price_increase = float(price_increase_raw.text.strip(" \n %").replace('\xa0', '')
                               .strip('±'))
    else:
        price_increase = None

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
                   "other_srq": other_sqr_area
                   }

    return data_series


def main_scrape_hemnet(loaded_hemnet_data, logger):
    """
    Scrapes new data from Hemnet.se

    Parameters
    ----------
    loaded_hemnet_data: pandas.Dataframe
        containing the samples we already have scraped from before
    logger:
        Logger

    Returns
    -------
    pd_data_series: pandas.Dataframe
        New scraped data
    """
    data = {}
    nr_objects = 0
    pbar = tqdm(total=PAGES_TO_SEARCH * HOUSE_CARDS_PER_PAGE)

    original_nr_objects = loaded_hemnet_data.shape[0]
    nr_consecutive_pages_already_documented = 0

    for page in range(1, PAGES_TO_SEARCH + 1):
        municipality_code = 17920  # for gothenburg
        url = f"https://www.hemnet.se/salda/bostader?location_ids%5B%5D={municipality_code}" \
              f"&page={page}"

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
            print("100 consecutive listings already in database, assume that the remaining"
                  " listings are also in dataset")
            logger.info("100 consecutive listings already in database, assume that the remaining"
                        " listings are also in dataset")
            break

    pd_data_series = pd.DataFrame.from_dict(data, orient="index")
    new_samples = pd_data_series.shape[0]
    print(f"originally {original_nr_objects} objects, now {new_samples + original_nr_objects}"
          f" objects")
    logger.info(
        f"web scraping - originally {original_nr_objects} objects, now"
        f" {new_samples + original_nr_objects} objects")
    return pd_data_series
