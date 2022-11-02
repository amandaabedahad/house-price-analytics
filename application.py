# coding=utf-8
import copy
import dash
import joblib
# import torch
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import folium
import locale
import geopandas as gpd
import numpy as np
from dash import Input, Output, State
# from data_process_functions import get_long_lat
# from neural_net import Simple_nn
import branca.colormap as cm
from fake_useragent import UserAgent
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
from sklearn.model_selection import train_test_split
from neural_net import get_percentage_off
import dash_bootstrap_components as dbc


def select_regions_with_nr_samples(data, nr_samples_threshold):
    """
    Returns data with regions where the number of samples for each region is greater than threshold. This to get some
    statistical significance when plotting in box plot, and to clean up the plot a bit.

    :param data: pandas data frame
    :param nr_samples_threshold: threshold which decides which regions to keep
    :return: filtered data set
    """
    df_part = data[["housing_type", "price_sqr_m", "region"]]
    df_part = df_part.loc[df_part.housing_type == "Lägenhet"]
    freq_regions_data = df_part["region"].value_counts()
    freq_over_threshold = freq_regions_data.loc[freq_regions_data > nr_samples_threshold]
    new_selection = df_part.loc[df_part["region"].isin(freq_over_threshold.index)]
    return new_selection


# TODO: need to structure this file nicely
def do_geocode(address, geolocator, attempt=1, max_attempts=10):
    try:
        return geolocator.geocode(address)
    except GeocoderUnavailable:
        if attempt <= max_attempts:
            return do_geocode(address, geolocator, attempt=attempt + 1)
        print("url attempts exceeded")
        raise


def get_long_lat(sample, pbar=None,
                 city='Göteborgs kommun'):  # TODO: change this hardcoded city and find more efficient calcs
    address = sample + ', ' + city
    ua = UserAgent()
    header = {
        "User-Agent": ua.random
    }

    geolocator = Nominatim(user_agent=str(header))
    location = do_geocode(address, geolocator)
    # location = geolocator.geocode(address)
    if pbar is not None:
        pbar.update(1)
    if location is None:
        return None, None, None
    address_info = location.address.split(',')
    post_code = address_info[-2]
    return location.latitude, location.longitude, post_code


def use_neural_net_model(x, path_model="nn_model.pkl"):
    net = Simple_nn()
    net.load_state_dict(torch.load(path_model))
    price_prediction = net(torch.tensor(x).float())
    return price_prediction.item()


def use_random_forest_model(x, path_model="random_forest_model.joblib"):
    model = joblib.load(path_model)
    prediction = model.predict(x)
    predicted_price = np.round(prediction[:, 0])
    predicted_rent = np.round(prediction[:, 1])
    return predicted_price, predicted_rent


def find_similar_listings(x, postcode):
    # find listings in same region with same number of rooms and ish same square meter.

    apartment_data = copy.deepcopy(data)
    apartment_data = apartment_data.reset_index()

    y = apartment_data[["final_price", "rent_month"]]
    X = apartment_data.drop(columns=["final_price", "rent_month"])
    # important to have same random state as when trained the model
    # so that no training samples are used here for testing.
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nr_rooms = x[1]
    nr_square_m = x[0]
    nr_room_options = [nr_rooms - 1 if nr_rooms > 2 else 1, nr_rooms, nr_rooms + 1]
    similar_listings = X_test[
        (X_test["post_code"] == postcode) & X_test["nr_rooms"].isin(nr_room_options)]
    if similar_listings.empty:
        return None, None
    X_sim_listings = X.iloc[similar_listings.index]
    y_sim_listings = y.iloc[similar_listings.index]

    return X_sim_listings, y_sim_listings


locale.setlocale(locale.LC_ALL, 'sv_SE.utf8')
app = dash.Dash(__name__)
application = app.server

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
data = data[data["housing_type"] == "Lägenhet"]
# Group data by region
data_grouped_by_region = data.groupby(["region"], as_index=False).mean()

# Merge the grouped data with the polygon data --> yields the average data for each polygon (region)
geo_data_map = geo_data.merge(data_grouped_by_region[["price_sqr_m", "region"]], on="region", how="left")

geo_data_map = geo_data_map.dropna(subset=["price_sqr_m"])
geo_data_map["price_sqr_m"] = geo_data_map["price_sqr_m"].apply(round)

# interactive part for folium map. Defined the tip that is shown when hovering over region
interactive_regions = folium.features.GeoJson(
    geo_data_map,
    style_function=style_function,
    control=False,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['region', 'price_sqr_m'],
        aliases=['Region: ', 'Price per square meter in kr: '],
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
    )
)

linear_colormap = cm.linear.YlOrRd_09.to_step(data=geo_data_map["price_sqr_m"],
                                              index=[10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
                                                     100000])

linear_colormap.caption = "Price per square meter (SEK)"
folium.GeoJson(
    geo_data_map,
    style_function=lambda feature: {
        'fillColor': linear_colormap(feature["properties"]["price_sqr_m"]),
        'fillOpacity': 0.5,
        'color': 'black',
        'weight': 0,
        'dashArray': '5, 5'
    }
).add_to(ma)
ma.add_child(linear_colormap)

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

app.title = "Hemnet Analytics"

app.layout = html.Div(
    children=[
        html.Div(
            children=[html.H1(children="Hemnet Analytics and Insights", className="header-title"),
                      html.P(children="This dashboard aims to provide useful information and insights of the "
                                      "objects sold in Gothenburg, Sweden. This is an extension to information "
                                      "already presented on the Swedish house market - www.hemnet.se.",
                             className="header-description")],
            className="header"),
        html.Div(
            children=[html.H2("Overview of the data set"),
                      html.P("The insights presented are based upon data from www.hemnet.se. The "
                             "presented insights are based on the following dataset"),
                      html.P("As clearly observed in the bar plot, the majority of listings are sold apartments. "
                             "Meaningful insights can only be drawn with a data set with sufficient number of "
                             "samples, which is why the following analytics are presented for apartments only."
                             "As more data is added to the data set, other housing types will be analysed"),
                      dcc.Graph(figure=px.bar(data_frame=data_all, x="housing_type"), id="bar_plot"),
                      ]
        ),
        html.Div(
            children=[html.H2("Pricing situation in Gothenburg"),
                      html.P("The interactive map presents how the average cost per square meter varies in "
                             "different regions in Gothenburg. The cost is displayed in SEK. We can observe some "
                             "regions where the price is noticeably higher - close to the ocean and in the city"),
                      html.Iframe(id='map1', srcDoc=open('map_city.html', 'r').read(), width='100%',
                                  height='500'),
                      html.P("For a more in depth visualisation of the pricing situation, continue to the box plot, "
                             "presented at the bottom of this page. ")
                      ],
            className="card"
        ),
        html.Div(
            children=[html.H2("Predict house price"),
                      html.P("As of today, Hemnet provides house price prediction as a beta service on their"
                             " webpage. This is however limited only to apartments in Stockholm, Sweden. "
                             "Similar functionality is here presented for Gothenburg.  The following parameters "
                             "will be used to predict the sold price for apartments, "
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
        html.Div(
            children=[html.H3("In depth view of the pricing situation"),
                      html.P("The box plot"), html.P("how to read plot"),
                      dcc.Graph(figure=px.box(data_frame=select_regions_with_nr_samples(data, nr_samples_threshold=20),
                                              x="region", y="price_sqr_m"), id="box_plot")]
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
    latitude, longitude, post_code = get_long_lat(address)
    x = np.array([square_meters, number_rooms, latitude, longitude])

    # No need to scale for random forest regression
    # std_scaler = pickle.load(open("standard_scaler.pkl", "rb"))
    # x_scaled = std_scaler.transform(x.reshape(1, -1))

    price_prediction, rent_prediction = use_random_forest_model(x.reshape(1, -1))

    X_similar_listings, y_similar_listings = find_similar_listings(x, post_code)
    if X_similar_listings is None:
        output_similar_listings = "No similar listings could be found."
    else:
        x_similar_listings = X_similar_listings[["sqr_meter", "nr_rooms", "latitude", "longitude"]].values

        price_prediction_sim_listings, rent_prediction_sim_listings = use_random_forest_model(x_similar_listings)

        price_percentage_off_similar_listings = round(get_percentage_off(y_similar_listings["final_price"].values,
                                                                         price_prediction_sim_listings).mean(axis=0), 1)
        rent_percentage_off = round(get_percentage_off(y_similar_listings["rent_month"].values,
                                                       rent_prediction_sim_listings).mean(axis=0), 1)
        output_similar_listings = "The predicted price percentage off on similar listings in this area is " \
                                  f"{price_percentage_off_similar_listings} and {rent_percentage_off} for rent."

    output_string = f"The predicted price is {price_prediction[0]} kr and the predicted rent is " \
                    f"{rent_prediction[0]} kr. " + output_similar_listings
    return output_string


if __name__ == "__main__":
    application.run(debug=False, host='0.0.0.0', port=80)
