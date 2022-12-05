"""
Application run script. Hosts app locally.
"""
# -*- coding: utf-8 -*-
import copy
import locale
import joblib
import plotly.express as px
import folium
import geopandas as gpd
import numpy as np
from dash import Input, Output, State, html, dcc, Dash
import branca.colormap as cm
import dash_bootstrap_components as dbc
from dotenv import load_dotenv
from data_process_functions import get_long_lat, clean_address_sample
from sql_queries import get_pandas_from_database
from ML_models import prep_data_ml, get_percentage_off


def select_regions_with_nr_samples(data, nr_samples_threshold):
    """
    Returns a dataframe where regions with fewer samples than the threshold are filtered out.

    Parameters
    ----------
    data: pd
        Pandas frame with data
    nr_samples_threshold: int
        Threshold for how many samples in each region that we want to keep

    Returns
    -------
    new_selection: dataframe
        Dataframe where regions with fewer samples than threshold are filtered out
    """
    df_part = data[["housing_type", "price_sqr_m", "region"]]
    df_part = df_part.loc[df_part.housing_type == "Lägenhet"]
    freq_regions_data = df_part["region"].value_counts()
    freq_over_threshold = freq_regions_data.loc[freq_regions_data > nr_samples_threshold]
    new_selection = df_part.loc[df_part["region"].isin(freq_over_threshold.index)]
    return new_selection


def use_random_forest_model(x, path_model="random_forest_model.joblib"):
    model = joblib.load(path_model)
    prediction = model.predict(x)
    predicted_price = np.round(prediction[:, 0])
    predicted_rent = np.round(prediction[:, 1])
    return predicted_price, predicted_rent


def find_similar_listings(x, postcode):
    """
    Find similar listings to a listing x. Similar if same post code, similar number of rooms and
    square meter.

    Parameters
    ----------
    x: list
        Attributes of listing, long lat sqrm rooms
    postcode: str
        Post code of listing x

    Returns
    -------
    similar_listings: dataframe, None
        Pandas frame of containing the features of the similar listings.
        If None, no similar listings were found
    y_similar_listings: dataframe, None
        Target information of similar listings, that is price and monthly fee.
    """

    all_data_from_database = copy.deepcopy(data_all)
    data_used_by_ML = prep_data_ml(all_data_from_database)
    listing_information_train_or_test = get_pandas_from_database("listing_train_or_test_set")

    test_listings = data_used_by_ML.loc[
        (data_used_by_ML["listing_id"].isin(listing_information_train_or_test["listing_id"])) &
        (listing_information_train_or_test["listing_in_train_set"] == 0)]

    y_test = test_listings[["final_price", "rent_month", "listing_id"]]
    X_test = test_listings.drop(columns=["final_price", "rent_month"])

    X_test["post_code"] = all_data_from_database[
        all_data_from_database["listing_id"].isin(X_test["listing_id"])]["post_code"].str.strip()

    nr_rooms = x[1]
    nr_square_m = x[0]
    nr_room_options = [nr_rooms - 1 if nr_rooms > 2 else 1, nr_rooms, nr_rooms + 1]
    similar_listings = X_test[
        (X_test["post_code"] == postcode) & X_test["nr_rooms"].isin(nr_room_options) &
        X_test["sqr_meter"].isin(range(int(nr_square_m) - 20, int(nr_square_m) + 20))]
    if similar_listings.empty:
        return None, None
    y_similar_listings = y_test[y_test["listing_id"].isin(similar_listings["listing_id"])]
    return similar_listings, y_similar_listings


def plot_scatter_rooms_type(data_to_group_and_plot):
    grouped_data = data_to_group_and_plot.groupby(["nr_rooms"]).mean()
    # exclude non-integer rooms
    grouped_data = grouped_data.drop(grouped_data.index[grouped_data.index % 1 != 0])
    fig = px.scatter(grouped_data, y="final_price",
                     title="Average price for different number of rooms",
                     labels={"final_price": "Average sold price",
                             "nr_rooms": "Number of rooms"})
    fig.update_traces(marker={'size': 15})
    fig.update_layout(xaxis=dict(tick0=min(grouped_data.index.unique()),
                                 dtick=1))

    prettify_plots(fig)
    return fig


def plot_prices_over_time(data_to_group_and_plot):
    grouped_data = data_to_group_and_plot.groupby(["sold_date"]).mean()
    fig = px.line(data_frame=grouped_data, y="final_price",
                  title="Average price over time",
                  labels={"final_price": "Average sold price",
                          "sold_date": "Sold date"})
    prettify_plots(fig)
    return fig


def plot_box_plot(data_to_plot, threshold=0):
    plot_data = select_regions_with_nr_samples(data_to_plot, nr_samples_threshold=threshold)
    fig = px.box(data_frame=plot_data, x="price_sqr_m", y="region",
                 labels={"price_sqr_m": "Price per square meter", "region": "Region"})
    prettify_plots(fig)

    fig.update_yaxes(showgrid=False)
    return fig


def prettify_plots(fig_plot):
    fig_plot.update_traces(
        marker_color='#7fafdf')
    fig_plot.update_xaxes(gridwidth=0.5, gridcolor='Grey')
    fig_plot.update_yaxes(gridwidth=0.5, gridcolor='Grey')

    fig_plot.update_layout({
        'paper_bgcolor': "#1f2630",
        'plot_bgcolor': "#1f2630",
    })

    fig_plot["layout"]["font"]["color"] = "#7fafdf"
    fig_plot["data"][0]["marker"]["color"] = "#7fafdf"
    fig_plot["data"][0]["opacity"] = 1
    fig_plot["data"][0]["marker"]["line"]["width"] = 0


locale.setlocale(locale.LC_ALL, 'sv_SE.utf8')
app = Dash(__name__)
application = app.server

style_function = lambda x: {'fillColor': '#ffffff',
                            'color': '#000000',
                            'fillOpacity': 0.1,
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000',
                                'color': '#000000',
                                'fillOpacity': 0.50,
                                'weight': 0.1}
load_dotenv()

# Read hemnet data and geo data
data_all = get_pandas_from_database("processed_data")

SHP_FILE = "geospatial_data_polygons_areas/JUR_PRIMÄROMRÅDEN_XU_region.shp"

# Clean the data
geo_data_raw = gpd.read_file(SHP_FILE)
geo_data = geo_data_raw[["PRIMÄROMRÅ", "PRIMÄRNAMN", "geometry"]]
geo_data = geo_data.rename(columns={"PRIMÄRNAMN": "region"})
data_nan_dropped = data_all.dropna(subset=["latitude", "longitude", "price_sqr_m"])

# Create map to include in DASH-app
ma = folium.Map(
    location=[data_nan_dropped["latitude"].mean(), data_nan_dropped["longitude"].mean()],
    zoom_start=7)

# TODO: delete these hardcoded values only decides zoom on map
ma.fit_bounds([(57.652402, 11.914561), (57.777214, 12.074102)])
data_apartments = data_nan_dropped[data_nan_dropped["housing_type"] == "Lägenhet"]
# Group data by region
data_grouped_by_region = data_apartments.groupby(["region"], as_index=False).mean()

# Merge the grouped data with the polygon data --> yields the average data for each polygon (region)
geo_data_map = geo_data.merge(data_grouped_by_region[["price_sqr_m", "region"]], on="region",
                              how="left")

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
        style="background-color: white; color: #333333; font-family: arial; "
              "font-size: 12px; padding: 10px;"
    )
)

linear_colormap = cm.linear.YlOrRd_09.to_step(data=geo_data_map["price_sqr_m"],
                                              index=[10000, 20000, 30000, 40000, 50000, 60000,
                                                     70000, 80000, 90000,
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

fig_bar_plot = px.histogram(data_frame=data_all, x="housing_type",
                            labels={"housing_type": "Housing type", "count": "Number of listings"})

prettify_plots(fig_bar_plot)

loading_style = {'position': 'absolute', 'align-self': 'center'}

app.title = "Hemnet Analytics"

square_meter_slide = list(np.linspace(10, 300, 11))
nr_rooms_slide = list(np.linspace(1, 10, 10))

app.layout = html.Div(
    children=[
        html.Div([
            html.H2("HEMNET ANALYTICS AND INSIGHTS", className="app__header__title"),
            html.P([
                "This dashboard aims to provide useful information and insights of the "
                "objects sold in Gothenburg, Sweden.", html.Br(),
                " This is an extension to information "
                "already presented on the Swedish house market - www.hemnet.se."],
                className="app__header__title--grey",
            ),
        ],
            className="app__header__desc",
        ),
        html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="left-column",
                    children=[
                        html.Div(
                            children=[
                                html.H6("Pricing situation in Gothenburg",
                                        className="graph__title"),
                                html.P(
                                    "The interactive map presents how the average cost per square "
                                    "meter "
                                    "varies in "
                                    "different regions in Gothenburg. We can "
                                    "observe some "
                                    "regions where the price is noticeably higher - close to the "
                                    "ocean and "
                                    "in the city")
                            ],
                        ),
                        html.Div(
                            id="heatmap-container",
                            children=[
                                html.Iframe(id='map1', srcDoc=open('map_city.html', 'r').read(),
                                            height='80%', width='100%'),
                                html.P(
                                    "For a more in depth visualisation of the pricing situation, "
                                    "continue to the box plot, "
                                    "presented at the bottom of this page. ")
                            ]
                        )
                    ]
                ),
                html.Div(
                    id="predict_house_price_container",
                    children=[
                        html.H2("Predict house price using ML", className="graph__title"),
                        html.P(
                            "As of today, Hemnet provides house price prediction as a beta service"
                            " on their webpage. This is however limited only to apartments in "
                            "Stockholm, Sweden. Similar functionality is here presented for "
                            "Gothenburg.  The following parameters "
                            "will be used to predict the sold price for apartments, using ML"),
                        html.P(
                            className="slider-text",
                            children="Drag the slider to change number of square meters:",
                        ),
                        dcc.Slider(
                            id="square-meters",
                            min=10,
                            value=50,
                            step=1,
                            marks={
                                str(sqrm): {
                                    "label": str(sqrm),
                                    "style": {"color": "#7fafdf"},
                                }
                                for sqrm in square_meter_slide
                            },
                            tooltip={'always_visible': True, 'placement': 'bottom'}
                        ),
                        html.P(
                            className="slider-text",
                            children="Drag the slider to change number of rooms:",
                        ),
                        dcc.Slider(
                            id="number-rooms",
                            min=1,
                            max=10,
                            step=1,
                            value=2,
                            marks={
                                str(nr): {
                                    "label": str(nr),
                                    "style": {"color": "#7fafdf"},
                                }
                                for nr in nr_rooms_slide
                            },
                            tooltip={'always_visible': True, 'placement': 'bottom'}
                        ),
                        html.P(
                            className="slider-text",
                            children="Provide address of listing:",
                        ),
                        dcc.Input(id="address", type="text", placeholder="address",
                                  className="text-input"),
                        html.Button('Predict price', id="submit-button", n_clicks=0,
                                    className="button"),
                        html.Div(
                            [
                                html.P(id="prediction-output"),

                                dcc.Loading(
                                    id="loading",
                                    type="circle",
                                    parent_style=loading_style
                                ),
                            ],
                        ),
                        html.Div(
                            id="graph-container",
                            children=[
                                html.H6("Overview of the data set", className="graph__title"),
                                html.P(
                                    "Meaningful insights can only be drawn with a data set with "
                                    "sufficient number of samples. If not otherwise mentioned, "
                                    "apartment data is used to obtain the results"),
                                dcc.Graph(figure=fig_bar_plot,
                                          id="bar_plot"),
                            ]
                        ),
                    ],
                ),
            ]
        ),
        html.Div(
            children=[
                html.H2("Additional insights from data", className="graph__title"),
                html.P("Select one or several listing types"),
                dcc.Dropdown(
                    id="object-filter",
                    options=[
                        {"label": object_type, "value": object_type} for object_type in
                        data_nan_dropped.housing_type.unique()
                    ],
                    value="Lägenhet",
                    multi=True,
                    className='chart-dropdown'
                )
            ]
        ),
        html.Div(
            [
                html.Div(
                    children=[dcc.Graph(figure=plot_scatter_rooms_type(data_apartments),
                                        className='plot',
                                        id="price-per-room"),
                              dcc.Graph(id="price-over-time",
                                        figure=plot_prices_over_time(data_apartments),
                                        className="plot")],
                    className="parent"
                )
            ]
        ),
        html.Div(
            children=[html.H2("In depth view of the pricing situation", className="graph__title"),
                      html.P("The box plot visualizes the spread of a data group using statistical"
                             "metrics - more specifically the median and quartiles. For example, "
                             "someone who wants to buy an "
                             "apartment in Agnesberg can expect to pay a median of ~23k per "
                             "square meter, where most "
                             "listings lay inbetween ~20k-~26k. In addition, there exist listings "
                             "with an average of "
                             "18k per square meter, up to 35k per square meter."),
                      html.H3("In summary, this plot provides easy comparison between regions, "
                              "and a quick overview "
                              " of what prices to expect as a buyer", style={"color": "#2cfec1"}),
                      html.P("Select one or several regions for easy comparison"),
                      dcc.Dropdown(id="selected-regions",
                                   options=[
                                       {"label": object_type, "value": object_type} for object_type
                                       in data_apartments["region"].unique()
                                   ],
                                   multi=True,
                                   className="chart-dropdown"
                                   ),
                      dcc.Graph(figure=plot_box_plot(data_apartments, threshold=10), id="box_plot",
                                className="boxplot-container-all")
                      ]
        ),
    ],
    id="root",
)


### Callback for dropdown list and ML input ###
@app.callback(
    [Output("price-over-time", "figure"), Output("price-per-room", "figure")],
    [
        Input("object-filter", "value"),
    ],
)
def update_charts(object_type):
    if type(object_type) == str:
        object_type = [object_type]
    filtered_data = data_all[data_all["housing_type"].isin(object_type)]
    price_over_time_figure = plot_prices_over_time(filtered_data)
    price_type_nr_rooms = plot_scatter_rooms_type(filtered_data)
    return price_over_time_figure, price_type_nr_rooms


@app.callback(
    [Output("prediction-output", "children"),
     Output('loading', 'parent_style'),
     Output("prediction-output", "style"), ],
    [Input("submit-button", "n_clicks"),
     State("square-meters", "value"),
     State("number-rooms", "value"),
     State("address", "value")], prevent_initial_call=True
)
def predict_price(n_clicks, square_meters, number_rooms, address):
    new_loading_style = loading_style
    latitude, longitude, post_code = get_long_lat(address)

    if clean_address_sample(address) is None:
        return "The format of the address is wrong and not allowed. Try another one.", \
               new_loading_style, {"color": "red"}
    if latitude is None:
        return "The address is not found in Gothenburg municipality. Try another one", \
               new_loading_style, {"color": "red"}

    x = np.array([square_meters, number_rooms, latitude, longitude])
    price_prediction, rent_prediction = use_random_forest_model(x.reshape(1, -1))

    X_similar_listings, y_similar_listings = find_similar_listings(x, post_code)
    if X_similar_listings is None:
        output_similar_listings = None, None
    else:
        x_similar_listings = \
            X_similar_listings[["sqr_meter", "nr_rooms", "latitude", "longitude"]].values

        price_prediction_sim_listings, rent_prediction_sim_listings = \
            use_random_forest_model(x_similar_listings)

        price_percentage_off_similar_listings = \
            round(get_percentage_off(y_similar_listings["final_price"].values,
                                     price_prediction_sim_listings).mean(axis=0), 1)
        rent_percentage_off = round(get_percentage_off(y_similar_listings["rent_month"].values,
                                                       rent_prediction_sim_listings).mean(axis=0),
                                    1)
        output_similar_listings = price_percentage_off_similar_listings, rent_percentage_off

    style_output_predictions = {"font-size": "2em", "margin": "0", "margin-top": "0",
                                "font-weight": "Bold",
                                "color": "#2cfec1"}

    output_string = html.P(
        children=[
            html.P("The predicted price for the listing is:"),
            html.P([locale.currency(price_prediction[0], grouping=True), " ± ",
                    output_similar_listings[0], "%"],
                   style=style_output_predictions),
            html.P("The predicted monthly fee for the listing above is:"),
            html.P([locale.currency(rent_prediction[0], grouping=True), " ± ",
                    output_similar_listings[1], "%"],
                   style=style_output_predictions),
            dbc.Tooltip(children=[
                "Using the ML-model, the average percentage off on similar listings are "
                "calculated. Listings are considered to be similar if:",
                html.Ul([html.Li("Located in the same area (same post code)"),
                         html.Li("Similar number of rooms - equal, one more or one less"),
                         html.Li("Similar size - 20sqm more or less")]),
                "If no value is shown, no similar listings was found."],
                target="howto-tooltip", placement="bottom",
                style={"color": "white", "width": "40% ", "background-color": "#1e2633",
                       "border": "1px solid #bbb"}),

            html.Span("how are the percentages calculated?", id="howto-tooltip",
                      style={"textDecoration": "underline", "cursor": "pointer", "color": "white"})
        ]
    )

    return output_string, new_loading_style, None


@app.callback(
    [Output("box_plot", "figure"), Output("box_plot", "className")],
    Input("selected-regions", "value")
)
def update_box_plot(selected_regions):
    if selected_regions is None or len(selected_regions) == 0:
        fig = plot_box_plot(data_apartments, threshold=10)
        style = "boxplot-container-all"
    else:
        data_selected_regions = data_apartments[data_apartments["region"].isin(selected_regions)]
        fig = plot_box_plot(data_selected_regions)
        style = "boxplot-container-few"
    return fig, style


if __name__ == "__main__":
    application.run(debug=False, host='0.0.0.0', port=80)
