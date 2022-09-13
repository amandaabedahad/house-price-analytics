import pandas as pd
import re

hemnet_data = pd.read_csv("hemnet_data/hemnet_house_data_processed.csv")
addresses_gothenburg = pd.read_excel("C:/Users/aabedaha/Downloads/Adressregister+Göteborg+september+2022.xlsx",
                                     header=1)
areas_gothenburg = pd.read_excel("C:/Users/aabedaha/Downloads/Områdeslista2022.xlsx")
areas_gothenburg = areas_gothenburg.rename(columns={"Basområde (NYKO)": "Basområde"})

for i, address in enumerate(hemnet_data["address"]):
    if address.upper() in addresses_gothenburg["Adress"].apply(str.upper).values:
        base_area = addresses_gothenburg[addresses_gothenburg["Adress"].apply(str.upper) == address.upper()]
    else:
        continue

    primary_area_row = areas_gothenburg[areas_gothenburg["Basområde"] == base_area["Basområde"].values[0]]
    primary_area_raw = primary_area_row["Primärområde"].values[0]
    primary_area = re.sub('\d+\s', "", primary_area_raw)

    hemnet_data.loc[i, "region_NEW"] = primary_area

hemnet_data.to_csv("hemnet_data/hemnet_house_data_processed.csv", index=False)
