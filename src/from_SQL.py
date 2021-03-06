# This script defines functions to retrieve data from SQL database at will.

import pandas as pd
import numpy as np
import datetime as dt
import psycopg2 as pg2



def from_SQL():
    """
    Function connects to PostgreSQL server and retrieves monthly and annual data

    Returns
    -------
    monthly_df: pandas DataFrame
        DataFrame containing monthly electricity generated by state and energy source
    annual_df: pandas DataFrame
        DataFrame containing annual electricity generated by state and energy source
    """
    # Connecting to PostgreSQL server to retrieve data
    conn = pg2.connect(dbname='postgres', 
                        user='postgres', 
                        host='localhost', 
                        port='5432', 
                        password='password')
    cur = conn.cursor()

    # Query monthly data from SQL database
    query = """
    SELECT * FROM monthly;
    """
    cur.execute(query)
    conn.commit()
    record = cur.fetchall()

    # Building monthly data dataframe
    monthly_df = pd.DataFrame(data=record, columns=['STATE', 'SOURCE', 'MONTH', 'DATA'])
    monthly_df['STATE + SOURCE'] = monthly_df.STATE + ' ' + monthly_df.SOURCE
    monthly_df = monthly_df.pivot(index='STATE + SOURCE', columns='MONTH', values='DATA')
    monthly_df['STATE'] = monthly_df.index.to_series().apply(lambda x: x[:2])
    monthly_df['SOURCE'] = monthly_df.index.to_series().apply(lambda x: x[3:])
    monthly_df.reset_index(drop=True, inplace=True)
    new_order = ['STATE', 'SOURCE']
    for col in monthly_df.columns[:-2]:
        new_order.append(col)
    monthly_df = monthly_df[new_order]

    # Query annual data from SQL database
    query = """
    SELECT * FROM annual;
    """
    cur.execute(query)
    conn.commit()
    record = cur.fetchall()

    # Building annual data dataframe
    annual_df = pd.DataFrame(data=record, columns=['STATE', 'SOURCE', 'YEAR', 'DATA'])
    annual_df['STATE + SOURCE'] = annual_df.STATE + ' ' + annual_df.SOURCE
    annual_df = annual_df.pivot(index='STATE + SOURCE', columns='YEAR', values='DATA')
    annual_df['STATE'] = annual_df.index.to_series().apply(lambda x: x[:2])
    annual_df['SOURCE'] = annual_df.index.to_series().apply(lambda x: x[3:])
    annual_df.reset_index(drop=True, inplace=True)
    new_order = ['STATE', 'SOURCE']
    for col in annual_df.columns[:-2]:
        new_order.append(col)
    annual_df = annual_df[new_order]

    # Closing connection to PostgreSQL server
    cur.close()
    conn.close()

    return monthly_df, annual_df


def from_SQL_predict():
    """
    Function connects to PostgreSQL server and retrieves 2020 monthly predictions

    Returns
    -------
    predictions_df: pandas DataFrame
        DataFrame containing monthly predictions for 2020 electricity generated by 
        state and energy source
    """
    # Connecting to PostgreSQL server to retrieve data
    conn = pg2.connect(dbname='postgres', 
                        user='postgres', 
                        host='localhost', 
                        port='5432', 
                        password='password')
    cur = conn.cursor()

    # Query predictions data from SQL database
    query = """
    SELECT * FROM predictions;
    """
    cur.execute(query)
    conn.commit()
    record = cur.fetchall()

    # Building predictions data dataframe
    predictions_df = pd.DataFrame(data=record, columns=['STATE', 'SOURCE', 'MONTH', 'DATA'])
    predictions_df['STATE + SOURCE'] = predictions_df.STATE + ' ' + predictions_df.SOURCE
    predictions_df = predictions_df.pivot(index='STATE + SOURCE', columns='MONTH', values='DATA')
    predictions_df['STATE'] = predictions_df.index.to_series().apply(lambda x: x[:2])
    predictions_df['SOURCE'] = predictions_df.index.to_series().apply(lambda x: x[3:])
    predictions_df.reset_index(drop=True, inplace=True)
    new_order = ['STATE', 'SOURCE']
    for col in predictions_df.columns[:-2]:
        new_order.append(col)
    predictions_df = predictions_df[new_order]

    # Closing connection to PostgreSQL server
    cur.close()
    conn.close()

    return predictions_df
