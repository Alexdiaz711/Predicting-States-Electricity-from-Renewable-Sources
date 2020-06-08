# This script uses triple exponential smoothing to predict electricity generation
# from renewable resources for all 50 states

from from_SQL import from_SQL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
import datetime as dt
import psycopg2 as pg2
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Importing data from PostgreSQL Database
    monthly, annual = from_SQL()

    # Creating dataframe to make predicitons for renewable energy sources
    print('Gathering Data...')
    predictions = {}
    renewables = ['solar', 'wind', 'hydro', 'bio', 'geo', 'renewables']
    renewables_df = monthly[monthly.SOURCE.isin(renewables)]
    renewables_df.reset_index(inplace=True, drop=True)

    # Holt-Winters Triple Exponential Smoothing
    print('Making predictions...')
    zeros_keys = []
    for i in range(renewables_df.shape[0]):
        if renewables_df.iloc[i].STATE != 'US':
            key = renewables_df.iloc[i].STATE + '_' + renewables_df.iloc[i].SOURCE
            train = renewables_df.iloc[i, 2:].astype(float)
            if train[-1] == 0:
                zeros_keys.append(key)
            fit = ExponentialSmoothing(train, seasonal_periods=12, damped=True,  
                            trend='add', seasonal='add').fit()
            predictions[key] = fit.forecast(12)

    # setting zero predicitons to actual zero 
    dates = predictions['AK_bio'].index
    for ind in zeros_keys:
        predictions[ind] = pd.Series(data = np.full((12,), 0.0), index=dates)

    # Setting predictions of all renewables to be sum of predictions 
    # from the energy sources
    states = list(renewables_df.STATE.unique())
    states.remove('US')
    for state in states:
        predictions[state + '_renewables'] = (predictions[state + '_solar'] +
                                            predictions[state + '_wind'] +
                                            predictions[state + '_hydro'] +
                                            predictions[state + '_bio'] +
                                            predictions[state + '_geo'])

    # Connecting to PostgreSQL database to store predictions
    print('Storing predictions in SQL database...')
    conn = pg2.connect(dbname='postgres', user='postgres', host='localhost', port='5432', password='password')
    cur = conn.cursor()

    # Create Monthly predictions table
    query = """
    CREATE TABLE predictions (
        state CHAR(2)
    ,   source VARCHAR(20)
    ,   month DATE
    ,   data DEC(7,6)
        );
    """
    cur.execute(query)
    conn.commit()

    # Populate monthly table
    for k, v in predictions.items(): 
        state = k[:2]
        source = k[3:]
        for i, d in enumerate(v):
            month = v.index[i].strftime("%Y-%m-%d")
            data = d
            query = """
            INSERT INTO predictions (state, source, month, data) 
            VALUES ('{}', '{}', '{}', {});
            """.format(state, source, month, data)
            cur.execute(query)
            conn.commit()

    # Closing PostgreSQL connection
    print('Finished!')
    cur.close()
    conn.close()
