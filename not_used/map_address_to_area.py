import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

shp_file = "geospatial_data_polygons_areas/JUR_PRIMÄROMRÅDEN_XU_region.shp"


def map_address_to_area(hemnet_data, path_shp_file):
    geo_data_raw = gpd.read_file(path_shp_file)
    geo_data = geo_data_raw[["PRIMÄROMRÅ", "PRIMÄRNAMN", "geometry"]]
    geo_data = geo_data.rename(columns={"PRIMÄRNAMN": "region"})

    # shp file has other lat/long reference system.
    # https://gis.stackexchange.com/questions/302699/extracting-longitude-and-latitude-from-shapefile

    geo_data["geometry"] = geo_data["geometry"].to_crs(epsg=4326)

    hemnet_data["coordinates"] = list(zip(hemnet_data["longitude"], hemnet_data["latitude"]))
    hemnet_data["coordinates"] = hemnet_data["coordinates"].apply(Point)

    points = gpd.GeoDataFrame(hemnet_data[["coordinates", "address", "latitude", "longitude"]], geometry="coordinates")
    points_to_region_map = gpd.tools.sjoin(points, geo_data, predicate="within")

    # TODO: will delete this region_new when new processed datafile is extracted
    hemnet_data_nan_location = hemnet_data[hemnet_data["longitude"].isna()].drop("coordinates", axis=1)
    rows_with_location_data = points_to_region_map.index
    hemnet_data = hemnet_data.iloc[rows_with_location_data].drop(["coordinates", "region"], axis=1)

    hemnet_data = hemnet_data.merge(points_to_region_map[["region", "latitude", "longitude"]],
                                    on=["latitude", "longitude"])
    # we get some duplicates, ugly solution atm. should be solved before it happens
    hemnet_data = hemnet_data[~hemnet_data.duplicated(keep="first")]
    hemnet_data_mapped = pd.concat([hemnet_data, hemnet_data_nan_location])
    # hemnet_data.to_csv("hemnet_data/hemnet_house_data_processed.csv", index=False)
    return hemnet_data_mapped
