import pymysql.cursors
import pandas as pd
import numpy as np


def create_server_connection(host_name, user_name, password, db):
    connection = pymysql.connect(host=host_name,
                                 user=user_name,
                                 password=password,
                                 db=db,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor,
                                 ssl={"fake_flag_to_enable_tls": True})

    return connection


def get_pandas_from_database(connection, table):
    sql_query = f"SELECT * from {table} ORDER BY sold_date DESC"
    df = pd.read_sql(sql_query, connection)
    return df


def insert_to_database(connection, new_data, table):
    values_raw = (new_data.shape[1] - 1) * "%s," + "%s"
    sql_insert_raw_data = f"INSERT INTO {table} VALUES ({values_raw})"

    new_data_removing_nan = new_data.replace({np.nan: None})
    cursor = connection.cursor()
    cursor.executemany(sql_insert_raw_data, new_data_removing_nan.values.tolist())
    connection.commit()

