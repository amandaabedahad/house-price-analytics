""" contains several different ML-models"""
import pickle

import joblib
import sklearn
from scipy.stats import zscore
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# import scikitplot.estimators as esti
import numpy as np
from neural_net import get_percentage_off


def plot_corr_heatmap(df):
    plt.clf()
    sns.heatmap(df.corr(), vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.3, annot=True, cbar_kws={"shrink": .70})


def scatter_plot(y_true, y_predicted):
    plt.scatter(range(len(y_true)), y_true)
    plt.scatter(range(len(y_true)), y_predicted)
    plt.show()


def visualize_tree(rf, X):
    tree = rf.estimators_[5]
    export_graphviz(tree, out_file='tree.dot', feature_names=X.columns, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')


def lat_long_to_polar(data):
    x = data["longitude"]
    y = data["latitude"]

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    data = data.drop(["longitude", "latitude"], axis=1)
    data["r"] = r
    data["theta"] = theta
    return data


def simple_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.33, random_state=42)
    std = StandardScaler()
    minmax = MinMaxScaler()
    X_train = minmax.fit_transform(X=X_train)
    X_test = minmax.transform(X_test)

    reg = LinearRegression().fit(X_train, y_train)

    y_predict = reg.predict(X_test)
    print(mean_squared_error(y_test, y_predict))
    print(reg.score(X_test, y_test))


def random_forest_regressor(X, y, logger=None):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

    param_grid = {'n_estimators': [300, 400, 500],
                  'max_features': [0.1, 0.3, 0.6]
                  }
    # print(sklearn.metrics.get_scorer_names())
    RandForest = RandomForestRegressor(n_jobs=-1, random_state=0, bootstrap=True, criterion="absolute_error")

    Tuned_RandForest = GridSearchCV(estimator=RandForest, param_grid=param_grid, scoring='neg_root_mean_squared_error',
                                    cv=5)

    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    Tuned_RandForest.fit(X_train, y_train)
    print(Tuned_RandForest.best_params_)
    y_predict = Tuned_RandForest.predict(X_test)
    # esti.plot_learning_curve(Tuned_RandForest, X, y)
    plt.show()
    print(np.sqrt(mean_squared_error(y_test, y_predict)))
    mean_percentage_off_test_set = get_percentage_off(y_test, y_predict).mean(axis=0)
    print(mean_percentage_off_test_set)
    if logger:
        logger.info(f"Mean percentage off on test set is: {mean_percentage_off_test_set}")
    return Tuned_RandForest, std


def update_ml_model(hemnet_house_data, logger):
    # Only look at apartments at the moment
    logger.info("updating machine learning model: random forest")
    filtered_data = hemnet_house_data[hemnet_house_data["housing_type"] == "Lägenhet"]
    data = filtered_data[["sqr_meter", "nr_rooms", "final_price", "latitude", "longitude", "rent_month"]].dropna()
    data = data.drop(filtered_data[filtered_data["rent_month"] == 0].index, axis=0)

    y = data[["final_price", "rent_month"]]
    X = data.drop(columns=["final_price", "rent_month"])
    model, std = random_forest_regressor(X, y, logger)

    logger.info("New model and standard scaler is saved")
    pickle.dump(std, open("standard_scaler.pkl", "wb"))
    joblib.dump(model, "random_forest_model.joblib")


if __name__ == "__main__":
    hemnet_house_data = pd.read_csv("hemnet_data/hemnet_house_data_processed.csv")
    hemnet_house_data.plot(kind="scatter", x="latitude", y="longitude", alpha=0.4, c="final_price",
                           cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
    # plt.show()
    filtered_data = hemnet_house_data[hemnet_house_data["housing_type"] == "Lägenhet"]
    data = filtered_data[["sqr_meter", "nr_rooms", "final_price", "latitude", "longitude", "rent_month"]].dropna()
    data = data.drop(filtered_data[filtered_data["rent_month"] == 0].index, axis=0)
    y = data[["final_price", "rent_month"]]
    X = data.drop(columns=["final_price", "rent_month"])

    forest_model, std = random_forest_regressor(X, y)

    pickle.dump(std, open("standard_scaler.pkl", "wb"))
    joblib.dump(forest_model, "random_forest_model.joblib")
