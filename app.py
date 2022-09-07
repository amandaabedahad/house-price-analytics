import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import folium
import locale
from dash import dash_table

locale.setlocale(locale.LC_ALL, 'sv_SE.utf-8')

data = pd.read_csv("hemnet_data/hemnet_house_data_2022-09-07.csv")

app = dash.Dash(__name__)

data = data.dropna(subset=["latitude", "longitude"])
ma = folium.Map(
    location=[data["latitude"].mean(), data["longitude"].mean()],
    zoom_start=7)

# TODO: delete these hardcoded values only decides zoom on map
ma.fit_bounds([(57.652402, 11.914561), (57.777214, 12.074102)])
geo_data = data[["latitude", "longitude"]]

# want to plot region-wise instead of every house.. group by region

data_grouped_by_region = data.groupby(["region"]).mean()

for region, row in data_grouped_by_region.iterrows():
    avg_sqr_meter_cost = round(row["price_sqr_m"])
    folium.Marker([row["latitude"], row["longitude"]],
                  popup=f"{region}, on avg {avg_sqr_meter_cost} kr/sqr meter").add_to(ma)

ma.save('map_city.html')
data_grouped_by_rooms = data.groupby(["nr_rooms"]).mean()
fig_scatter = px.scatter(data_grouped_by_rooms, y="final_price", title="Average price for different number of rooms",
                         labels={"final_price": "Average sold price",
                                 "nr_rooms": "Number of rooms"})

data_grouped_by_date = data.groupby(["sold_date"]).mean()
fig_prices_over_time = px.line(data_frame=data_grouped_by_date, y="final_price", title="Average price over time",
                               labels={"final_price": "Average sold price",
                                       "sold_date": "Sold date"})

# data_grouped_by_housing_type = data.groupby(["housing_type"]).mean()
# 'fig_housing_type_price = px.line(data_frame=data_grouped_by_housing_type, y="")

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
                      html.Iframe(id='map1', srcDoc=open('map_city.html', 'r').read(), width='100%', height='300'),
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
