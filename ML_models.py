""" contains several different ML-models"""
import xgboost as xgb
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from neural_net import get_percentage_off
from sql_queries import reset_train_indices_in_table, get_pandas_from_database, remove_and_update_table
from dotenv import load_dotenv


def plot_corr_heatmap(df):
    plt.clf()
    sns.heatmap(df.corr(), vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.3, annot=True, cbar_kws={"shrink": .70})


def scatter_plot(y_true, y_predicted):
    plt.scatter(range(len(y_true)), y_true)
    plt.scatter(range(len(y_true)), y_predicted)
    plt.show()


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

    reg = LinearRegression().fit(X_train, y_train)

    y_predict = reg.predict(X_test)
    print(mean_squared_error(y_test, y_predict))
    print(reg.score(X_test, y_test))


def random_forest_regressor(data, logger=None):
    y = data[["final_price", "rent_month"]]
    X = data.drop(columns=["final_price", "rent_month"])

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, random_state=42)

    reset_train_indices_in_table("listing_train_or_test_set")

    data_listing_information = get_pandas_from_database("listing_train_or_test_set")
    listings_in_train_condition = data_listing_information["listing_id"].isin(X_train_df["listing_id"].values)
    data_listing_information["listing_in_train_set"][listings_in_train_condition] = 1
    remove_and_update_table(data_listing_information, "listing_train_or_test_set")

    X_train = X_train_df.drop(columns=["listing_id"]).values
    X_test = X_test_df.drop(columns=["listing_id"]).values
    y_train = y_train_df.values
    y_test = y_test_df.values

    param_grid = {'n_estimators': [300, 400, 500],
                  'max_features': [0.1, 0.3, 0.6]
                  }
    # print(sklearn.metrics.get_scorer_names())
    RandForest = RandomForestRegressor(n_jobs=-1, random_state=0, bootstrap=True, criterion="absolute_error")

    Tuned_RandForest = GridSearchCV(estimator=RandForest, param_grid=param_grid, scoring='neg_root_mean_squared_error',
                                    cv=5)
    Tuned_RandForest.fit(X_train, y_train)
    best_estimator = Tuned_RandForest.best_estimator_
    print(Tuned_RandForest.best_params_)
    y_predict = best_estimator.predict(X_test)
    plt.show()
    print(np.sqrt(mean_squared_error(y_test, y_predict)))
    mean_percentage_off_test_set = get_percentage_off(y_test, y_predict).mean(axis=0)
    print(mean_percentage_off_test_set)
    if logger:
        logger.info(f"Mean percentage off on test set is: {mean_percentage_off_test_set}")
        logger.info(f"The parameters of the grid search were {Tuned_RandForest.best_params_}")
    return best_estimator


def xgboost_model(data, logger=None):
    y = data[["final_price", "rent_month"]]
    X = data.drop(columns=["final_price", "rent_month"])

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, random_state=42)

    # reset_train_indices_in_table("listing_train_or_test_set")

    # data_listing_information = get_pandas_from_database("listing_train_or_test_set")
    # listings_in_train_condition = data_listing_information["listing_id"].isin(X_train_df["listing_id"].values)
    # data_listing_information["listing_in_train_set"][listings_in_train_condition] = 1
    # remove_and_update_table(data_listing_information, "listing_train_or_test_set")

    X_train = X_train_df.drop(columns=["listing_id"]).values
    X_test = X_test_df.drop(columns=["listing_id"]).values
    y_train = y_train_df.values
    y_test = y_test_df.values

    params = {'max_depth': [3, 6, 8],
              #'learning_rate': [0.01, 0.1, 0.2, 0.3],
              #'subsample': np.arange(0.5, 1.0, 0.1),
              #'colsample_bytree': np.arange(0.4, 1.0, 0.1),
              #'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
              'n_estimators': [200, 400]}
    xgbr = xgb.XGBRegressor(seed=20)
    clf = GridSearchCV(estimator=xgbr,
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       verbose=1)

    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    print(clf.best_params_)
    print(np.sqrt(mean_squared_error(y_test, y_predict)))
    mean_percentage_off_test_set = get_percentage_off(y_test, y_predict).mean(axis=0)
    print(mean_percentage_off_test_set)

    plt.clf()
    sns.histplot([y_test[:, 0], y_predict[:, 0]], multiple="dodge")
    plt.legend(labels=["Ground truth", "Predicted value"])
    plt.show()


def prep_data_ml(data_all):
    apartment_data = data_all[data_all["housing_type"] == "Lägenhet"]
    data_filtered = apartment_data[
        ["sqr_meter", "nr_rooms", "final_price", "latitude", "longitude", "rent_month", "listing_id"]].dropna()
    data_final = data_filtered.drop(data_filtered[data_filtered["rent_month"] == 0].index, axis=0)
    return data_final


def update_ml_model(hemnet_house_data, logger):
    # Only look at apartments at the moment
    logger.info("updating machine learning model: random forest")

    data = prep_data_ml(hemnet_house_data)

    model = random_forest_regressor(data, logger)

    logger.info("New model and standard scaler is saved")
    joblib.dump(model, "random_forest_model.joblib")


if __name__ == "__main__":
    load_dotenv('.env')

    # Read hemnet data and geo data
    data_all = get_pandas_from_database("processed_data")

    data = prep_data_ml(data_all)

    forest_model = random_forest_regressor(data)
    # xgboost_model(data)
    joblib.dump(forest_model, "random_forest_model.joblib")
