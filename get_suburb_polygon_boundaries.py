import json
import requests
from fake_useragent import UserAgent
import pandas as pd
import re
import locale

locale.setlocale(locale.LC_ALL, 'sv_SE')
ua = UserAgent()
header = {"User-Agent": ua.random}
hemnet_data = pd.read_csv("hemnet_data/hemnet_house_data_processed.csv", index=False)
areas_gothenburg = pd.read_excel("C:/Users/aabedaha/Downloads/Områdeslista2022.xlsx")

suburbs = areas_gothenburg["Primärområde"]
dict_polygons = {}
for suburb in suburbs.unique():
    suburb = re.search("[A-Ö|a-ö]+", suburb)[0]
    if suburb not in dict_polygons:
        subrub_and_city = suburb + "Göteborg"
        url = f"https://nominatim.openstreetmap.org/search.php?q={suburb}&polygon_geojson=1&format=json"
        response = requests.get(url, headers=header).text
        dict_polygons[suburb] = response

with open("polygon_suburbs_gothenburg.txt", 'w') as polygon_file:
    polygon_file.write(json.dumps(dict_polygons))
