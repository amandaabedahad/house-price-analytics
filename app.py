import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import folium
import locale
from dash import dash_table
import geopandas as gpd
import numpy as np

# TODO: need to structure this file nicely

locale.setlocale(locale.LC_ALL, 'sv_SE.utf-8')
app = dash.Dash(__name__)

style_function = lambda x: {'fillColor': '#ffffff',
                            'color': '#000000',
                            'fillOpacity': 0.1,
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000',
                                'color': '#000000',
                                'fillOpacity': 0.50,
                                'weight': 0.1}

# Read hemnet data and geo data
data = pd.read_csv("hemnet_data/hemnet_house_data_processed.csv")
shp_file = "geospatial_data_polygons_areas/JUR_PRIMÄROMRÅDEN_XU_region.shp"

# Clean the data
geo_data_raw = gpd.read_file(shp_file)
geo_data = geo_data_raw[["PRIMÄROMRÅ", "PRIMÄRNAMN", "geometry"]]
geo_data = geo_data.rename(columns={"PRIMÄRNAMN": "region_NEW"})
data = data.dropna(subset=["latitude", "longitude", "price_sqr_m"])


# Create map to include in DASH-app
ma = folium.Map(
    location=[data["latitude"].mean(), data["longitude"].mean()],
    zoom_start=7)

# TODO: delete these hardcoded values only decides zoom on map
ma.fit_bounds([(57.652402, 11.914561), (57.777214, 12.074102)])

# Group data by region
data_grouped_by_region = data.groupby(["region_NEW"], as_index=False).mean()

# Merge the grouped data with the polygon data --> yields the average data for each polygon (region)
geo_data = geo_data.merge(data_grouped_by_region[["price_sqr_m", "region_NEW"]], on="region_NEW", how="left")
geo_data["price_sqr_m"] = geo_data["price_sqr_m"].dropna().apply(round)

myscale = np.linspace(geo_data["price_sqr_m"].min(), geo_data["price_sqr_m"].max(), num=10).tolist()

# interactive part for folium map. Defined the tip that is shown when hovering over region
interactive_regions = folium.features.GeoJson(
    geo_data,
    style_function=style_function,
    control=False,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['region_NEW', 'price_sqr_m'],
        aliases=['Region: ', 'Price per square meter in kr: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
    )
)

# Visuals regions and color depending on the value of price per square meter.
folium.Choropleth(geo_data=geo_data,
                  name="Choropleth",
                  data=geo_data,
                  columns=["region_NEW", "price_sqr_m"],
                  key_on="feature.properties.region_NEW",
                  fill_color='RdPu',
                  nan_fill_color="White",
                  nan_fill_opacity=0.3,
                  fill_opacity=0.8,
                  line_opacity=0.2,
                  smooth_factor=0,
                  legend_name='SEK',
                  bins=myscale).add_to(ma)

ma.add_child(interactive_regions)
ma.keep_in_front(interactive_regions)
folium.LayerControl().add_to(ma)
ma.save('map_city.html')

# These dataframes are used for plots in the dash app
data_grouped_by_rooms = data.groupby(["nr_rooms"]).mean()
fig_scatter = px.scatter(data_grouped_by_rooms, y="final_price", title="Average price for different number of rooms",
                         labels={"final_price": "Average sold price",
                                 "nr_rooms": "Number of rooms"})

data_grouped_by_date = data.groupby(["sold_date"]).mean()
fig_prices_over_time = px.line(data_frame=data_grouped_by_date, y="final_price", title="Average price over time",
                               labels={"final_price": "Average sold price",
                                       "sold_date": "Sold date"})

fig_price_location = None
summary_data_df = pd.DataFrame(
    {"info": {0: "Number of objects in dataset", 1: "Number of apartments", 2: "Number of houses"},
     "Value": {0: data.shape[0], 1: (data["housing_type"] == "Lägenhet").sum(),
               2: (data["housing_type"] == 'Villa').sum()}})

app.layout = html.Div(
    children=[
        html.Div(
            children=[html.H1(children="Hemnet Analytics and Insights", className="header-title"),
                      html.P(children="Some description blabla", className="header-description")],
            className="header"),
        html.Div(
            children=[html.P("Average cost per square meter in different regions"),
                      html.Iframe(id='map1', srcDoc=open('map_city.html', 'r').read(), width='100%', height='500'),
                      html.P("Data statistics"),
                      dash_table.DataTable(summary_data_df.to_dict('records'),
                                           [{"name": i, "id": i} for i in summary_data_df.columns])],
            className="card"
        ),
        html.Div(
            children=[dcc.Graph(figure=fig_scatter, className='plot'),
                      dcc.Graph(figure=fig_prices_over_time, className="plot")],
            className="parent"
        ),
        html.Div(
            children=[dcc.Dropdown(
                id="test",
                options=[{"label": object_type, "value": object_type} for object_type in data.housing_type.unique()]
            )]
        )
    ],
    className="wrapper"
)

if __name__ == "__main__":
    app.run_server(debug=True)
