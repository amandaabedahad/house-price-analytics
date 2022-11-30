import os
import pymysql.cursors
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv('.env')


def create_server_connection(host_name, user_name, password, db):
    connection = pymysql.connect(host=host_name,
                                 user=user_name,
                                 password=password,
                                 db=db,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor,
                                 ssl={"fake_flag_to_enable_tls": True})

    return connection


def get_pandas_from_database(table):
    connection = create_server_connection(os.environ.get('DATABASE_HOST_NAME'),
                                          os.environ.get('DATABASE_USERNAME'),
                                          os.environ.get('DATABASE_PASSWORD'),
                                          os.environ.get('DATABASE_NAME'))
    sql_query = f"SELECT * from {table} ORDER BY listing_id DESC"
    df = pd.read_sql(sql_query, connection)
    connection.close()
    return df


def insert_to_database(new_data, table):
    connection = create_server_connection(os.environ.get('DATABASE_HOST_NAME'),
                                          os.environ.get('DATABASE_USERNAME'),
                                          os.environ.get('DATABASE_PASSWORD'),
                                          os.environ.get('DATABASE_NAME'))
    values_raw = (new_data.shape[1] - 1) * "%s," + "%s"

    if table == 'listing_information' or 'listing_train_or_test_set':
        new_data_sorted = new_data.sort_values(by="listing_id", ascending=False)
    else:
        new_data_sorted = new_data.sort_values(["sold_date", "final_price", "address"], ascending=[True, True, True])
    if table == "raw_data":
        sql_insert_raw_data = f"INSERT INTO {table} " \
                              f"(address, housing_type, region, city, sqr_meter, nr_rooms, " \
                              f"rent_month, final_price, sold_date, price_increase, price_sqr_m, land_area, other_srq, " \
                              f"listing_id) " \
                              f"VALUES ({values_raw})"
    elif table == "processed_data":
        sql_insert_raw_data = f"INSERT INTO {table} " \
                              f"(address, housing_type, city, sqr_meter, nr_rooms, " \
                              f"rent_month, final_price, sold_date, price_increase, price_sqr_m, land_area, other_srq," \
                              f"latitude, longitude, post_code, region, listing_id) " \
                              f"VALUES ({values_raw})"
    elif table == "listing_information":
        sql_insert_raw_data = f"INSERT INTO {table} (listing_id) VALUES ({values_raw})"
    elif table == "listing_train_or_test_set":
        sql_insert_raw_data = f"INSERT INTO {table} VALUES ({values_raw})"

    new_data_removing_nan = new_data_sorted.replace({np.nan: None})

    cursor = connection.cursor()
    cursor.executemany(sql_insert_raw_data, new_data_removing_nan.values.tolist())
    connection.commit()
    connection.close()


def reset_train_indices_in_table(table):
    connection = create_server_connection(os.environ.get('DATABASE_HOST_NAME'),
                                          os.environ.get('DATABASE_USERNAME'),
                                          os.environ.get('DATABASE_PASSWORD'),
                                          os.environ.get('DATABASE_NAME'))
    sql_reset_rows = f"UPDATE {table} SET listing_in_train_set = 0"
    cursor = connection.cursor()
    cursor.execute(sql_reset_rows)
    connection.commit()
    connection.close()


# This function works, but takes way too long time to execute. Takes time to find the right index where to update,
# even when using primary keys.
def update_rows_in_table(new_data, table):
    connection = create_server_connection(os.environ.get('DATABASE_HOST_NAME'),
                                          os.environ.get('DATABASE_USERNAME'),
                                          os.environ.get('DATABASE_PASSWORD'),
                                          os.environ.get('DATABASE_NAME'))
    sql_update_rows = f"UPDATE {table} SET listing_in_train_set = %s WHERE listing_id = %s ORDER BY listing_id asc"
    reordered_column_data = new_data[["listing_in_train_set", "listing_id"]].values.tolist()
    cursor = connection.cursor()
    cursor.executemany(sql_update_rows, reordered_column_data)
    connection.commit()
    connection.close()


# Instead of update, remove content of whole table and insert again
def remove_and_update_table(new_data, table):
    connection = create_server_connection(os.environ.get('DATABASE_HOST_NAME'),
                                          os.environ.get('DATABASE_USERNAME'),
                                          os.environ.get('DATABASE_PASSWORD'),
                                          os.environ.get('DATABASE_NAME'))
    sql_query_drop_column_listing_in_train = f"DELETE FROM {table}"
    cursor = connection.cursor()
    cursor.execute(sql_query_drop_column_listing_in_train)

    sql_insert_query = f"INSERT INTO {table} VALUES (%s, %s)"

    cursor.executemany(sql_insert_query, new_data.values.tolist())
    connection.commit()
    connection.close()
