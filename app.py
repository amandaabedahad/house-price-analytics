import pickle
import dash
import joblib
import torch
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import folium
import locale
import geopandas as gpd
import numpy as np
import branca.colormap as cmp
from dash import Input, Output, State
from data_process_functions import get_long_lat
from neural_net import Simple_nn


# TODO: need to structure this file nicely

def use_neural_net_model(x, path_model="nn_model.pkl"):
    net = Simple_nn()
    net.load_state_dict(torch.load(path_model))
    price_prediction = net(torch.tensor(x).float())
    return price_prediction.item()


def use_random_forest_model(x, path_model="random_forest_model.joblib"):
    model = joblib.load(path_model)
    price_prediction = model.predict(x)
    return round(price_prediction[0])


if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, 'sv_SE.utf8')
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
    data_all = pd.read_csv("hemnet_data/hemnet_house_data_processed.csv")
    shp_file = "geospatial_data_polygons_areas/JUR_PRIMÄROMRÅDEN_XU_region.shp"

    # Clean the data
    geo_data_raw = gpd.read_file(shp_file)
    geo_data = geo_data_raw[["PRIMÄROMRÅ", "PRIMÄRNAMN", "geometry"]]
    geo_data = geo_data.rename(columns={"PRIMÄRNAMN": "region"})
    data = data_all.dropna(subset=["latitude", "longitude", "price_sqr_m"])

    # Create map to include in DASH-app
    ma = folium.Map(
        location=[data["latitude"].mean(), data["longitude"].mean()],
        zoom_start=7)

    # TODO: delete these hardcoded values only decides zoom on map
    ma.fit_bounds([(57.652402, 11.914561), (57.777214, 12.074102)])

    # Group data by region
    data_grouped_by_region = data.groupby(["region"], as_index=False).mean()

    # Merge the grouped data with the polygon data --> yields the average data for each polygon (region)
    geo_data = geo_data.merge(data_grouped_by_region[["price_sqr_m", "region"]], on="region", how="left")
    geo_data["price_sqr_m"] = geo_data["price_sqr_m"].dropna().apply(round)

    myscale = np.linspace(geo_data["price_sqr_m"].min(), geo_data["price_sqr_m"].max(), num=10).tolist()

    # interactive part for folium map. Defined the tip that is shown when hovering over region
    interactive_regions = folium.features.GeoJson(
        geo_data,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['region', 'price_sqr_m'],
            aliases=['Region: ', 'Price per square meter in kr: '],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    )

    colormap = cmp.linear.YlGnBu_09.to_step(data=geo_data['price_sqr_m'].dropna(), method='quant',
                                            quantiles=[0, 0.1, 0.75, 0.9, 0.98, 1])

    style_function_2 = lambda x: {"weight": 0.5,
                                  'color': 'black',
                                  'fillColor': colormap(x['price_sqr_m']),
                                  'fillOpacity': 0.75}

    # Visuals regions and color depending on the value of price per square meter.
    folium.Choropleth(geo_data=geo_data,
                      name="Choropleth",
                      data=geo_data,
                      columns=["region", "price_sqr_m"],
                      key_on="feature.properties.region",
                      fill_color="YlGn",
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
    fig_scatter = px.scatter(data_grouped_by_rooms, y="final_price",
                             title="Average price for different number of rooms",
                             labels={"final_price": "Average sold price",
                                     "nr_rooms": "Number of rooms"})

    data_grouped_by_date = data.groupby(["sold_date"]).mean()
    fig_prices_over_time = px.line(data_frame=data_grouped_by_date, y="final_price", title="Average price over time",
                                   labels={"final_price": "Average sold price",
                                           "sold_date": "Sold date"})

    fig_price_location = None

    app.layout = html.Div(
        children=[
            html.Div(
                children=[html.H1(children="Hemnet Analytics and Insights", className="header-title"),
                          html.P(children="Some description blabla", className="header-description")],
                className="header"),
            html.Div(
                children=[
                    html.P("This dashboard aims to provide useful information and insights of the objects sold in "
                           "Gothenburg, Sweden. The data is scraped from www.hemnet.se. The"
                           "presented insights are based on the following dataset"),
                    html.Table([
                        html.Tr([html.Th("Number of objects in dataset  "),
                                 html.Th("Number of apartments   "),
                                 html.Th("Number of houses  ")]),
                        html.Tr([html.Td(data.shape[0]),
                                 html.Td((data["housing_type"] == "Lägenhet").sum()),
                                 html.Td((data["housing_type"] == 'Villa').sum())])
                    ], className="styled-table")]
            ),
            html.Div(
                children=[html.P("Average cost per square meter in different regions"),
                          html.Iframe(id='map1', srcDoc=open('map_city.html', 'r').read(), width='100%',
                                      height='500'),
                          ],
                className="card"
            ),
            html.Div(
                children=[html.H2("Predict house price"),
                          html.P("The following parameters will be used to predict the sold price for apartments, "
                                 "using ML"),
                          dcc.Input(id="square-meters", type="number", placeholder="square meters"),
                          dcc.Input(id="number-rooms", type="number", placeholder="number of rooms"),
                          dcc.Input(id="address", type="text", placeholder="address"),
                          html.Button('Predict price', id="submit-button", n_clicks=0, className="button"),
                          html.P(id="prediction-output")
                          ], className="center"
            ),
            html.Div(
                children=[html.H2("Insights from data"),
                          html.P("Select one or several listing types"),
                          dcc.Dropdown(
                              id="object-filter",
                              options=[{"label": object_type, "value": object_type} for object_type in
                                       data.housing_type.unique()],
                              value="Lägenhet",
                              multi=True
                          )]
            ),
            html.Div(
                [
                    html.Div(
                        children=[dcc.Graph(figure=fig_scatter, className='plot', id="price-per-room"),
                                  dcc.Graph(id="price-over-time", figure=fig_prices_over_time, className="plot")],
                        className="parent"
                    )]
            ),
        ],
        className="wrapper"
    )


    ### Callback for dropdown list and ML input ##############################

    @app.callback(
        Output("price-over-time", "figure"),
        [
            Input("object-filter", "value"),
        ],
    )
    def update_charts(object_type):
        if type(object_type) == str:
            object_type = [object_type]
        filtered_data = data_all[data_all["housing_type"].isin(object_type)]
        filtered_data_groupby_type = filtered_data.groupby(["sold_date"]).mean()
        price_over_time_figure = px.line(data_frame=filtered_data_groupby_type, y="final_price",
                                         title="Average price over time",
                                         labels={"final_price": "Average sold price",
                                                 "sold_date": "Sold date"})
        price_over_time_figure.update_layout(transition_duration=500)
        return price_over_time_figure


    @app.callback(
        Output("prediction-output", "children"),
        [Input("submit-button", "n_clicks"),
         State("square-meters", "value"),
         State("number-rooms", "value"),
         State("address", "value")], prevent_initial_call=True
    )
    def predict_price(n_clicks, square_meters, number_rooms, address):
        # TODO: doublecheck the order of the features. "latitude", "longitude"
        latitude, longitude, _ = get_long_lat(address)
        x = np.array([square_meters, number_rooms, latitude, longitude])
        std_scaler = pickle.load(open("standard_scaler.pkl", "rb"))
        x_scaled = std_scaler.transform(x.reshape(1, -1))

        price_prediction = use_random_forest_model(x_scaled)

        output_string = f"The predicted price is {price_prediction} kr"
        return output_string


    app.run_server(debug=True)
