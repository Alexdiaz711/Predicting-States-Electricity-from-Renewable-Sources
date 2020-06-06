from src.from_SQL import from_SQL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import warnings
warnings.filterwarnings("ignore")



def RMSE(y_true, y_hat):
    '''
    Function to compute Root Mean Squared Error

    Inputs
    ------
    y_true : list or np.array of true values
    y_hat : list or np.array of predicted values

    Outputs
    ------
    float: Root Mean Squared Error
    '''
    return np.sqrt(np.mean((y_true - y_hat)**2))

def create_source_df(source):
    '''
    Function to used in create_source_dict function. Creates individual
    dataframe for an energy source's data.

    Inputs
    ------
    source : string with name of energy source.

    Outputs
    ------
    pandas DataFrame: Dataframe containing data for an individual energy source.
    '''
    sources[source] = current[current['SOURCE']==source].iloc[:,2:].T
    num = sources[source].columns[0]
    sources[source] = sources[source].rename({num: 'Data'}, axis=1)
    sources[source]['Data'] = sources[source]['Data'].astype(float)
    return sources[source]

def create_sources_dict():
    '''
    Function to create dictionary of dataframes, each containing data for an
    individual energy source

    Outputs
    ------
    dictionary: Dictionary of DataFrames, each containing data for an individual energy source
    '''
    sources = {}
    sources['solar'] = create_source_df('solar')
    sources['wind'] = create_source_df('wind')
    sources['hydro'] = create_source_df('hydro')
    sources['bio'] = create_source_df('bio')
    sources['geo'] = create_source_df('geo')
    sources['coal'] = create_source_df('coal')
    sources['nat_gas'] = create_source_df('nat_gas')
    sources['nuclear'] = create_source_df('nuclear')
    sources['other'] = create_source_df('other')
    sources['other_gas'] = create_source_df('other_gas')
    sources['petro'] = create_source_df('petro')
    for source in sources.keys():
        year = []
        for ind in sources[source].index:
            year.append(ind.year)
        sources[source]['Year'] = year
    return sources

def build_model_uni():
    '''
    Function to build Univariate LSTM Nueral Network ready to be trained

    Outputs
    ------
    model: LSTM Nueral Network (untrained)
    '''
    model = Sequential()
    model.add(LSTM(32, input_shape=(24, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(12, activation='linear'))
    model.compile(optimizer='rmsprop',
                  loss='mse')
    return model

def build_model_multi():
    '''
    Function to build Multivariate LSTM Nueral Network ready to be trained

    Outputs
    ------
    model: LSTM Nueral Network (untrained)
    '''
    model = Sequential()
    model.add(LSTM(32, input_shape=(24, 11), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(12, activation='linear'))
    model.compile(optimizer='rmsprop',
                  loss='mse')
    return model

def windowize_data(data, n_prev, n_future):
    '''
    Function to windowize time-series data to be used in supervised machine learning
    problem using LSTM Nueral Network

    Outputs
    ------
    x: np.array of training features
    y: np.array of training target variables
    '''    
    n_predictions = len(data) - n_prev - n_future + 1
    x = np.full((n_predictions, n_prev, 1), None)
    y = np.full((n_predictions, n_future), None)
    for i, d in enumerate(data):
        if i >= n_prev:
            if i <= (len(data) - n_future):
                y[i - n_prev] = np.array(data[i:i+n_future])
                x[i - n_prev] = np.array(data[i-n_prev:i]).reshape(n_prev,1)
    y = y.reshape((len(data) - n_prev - n_future + 1), n_future, 1)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return x, y



# Importing the data from the PostgreSQL database and storing in a pandas DF.
monthly, annual = from_SQL()
states = {}

# creating dictionary of dataframes, one for each state
for state in monthly.STATE.unique():
    if state != 'US':
        states[state] = monthly[monthly.STATE == state]


# Using California's data to evaluate model performace
state = 'CA'
current = states[state]
CA_predicts = {}
RMSE_dict = {}
RMSE_dict[state] = pd.DataFrame(columns=['solar', 'wind', 'hydro', 'bio', 'geo'],
                       index=['Baseline', 'Holt-Winters', 'ARIMA', 'LSTM Univariate', 'LSTM Multivariate'],
                       data=None,
                      )
renewables = ['solar', 'wind', 'hydro', 'bio', 'geo']

# Baseline: Linear Regression
CA_predicts['Baseline'] = {}
sources = create_sources_dict()
for source in renewables:
    sources[source]['Time'] = np.arange(228)
    sources[source]['Constant'] = 1
    sources[source] = sources[source][['Constant', 'Time', 'Data', 'Year']]
    train = sources[source].iloc[:-12, :]
    X_train = train[['Constant', 'Time']].to_numpy()
    y_train = train['Data'].to_numpy()
    test = sources[source].iloc[-12:, :]
    X_test = test[['Constant', 'Time']].to_numpy()
    y_hat = sm.OLS(y_train, X_train).fit().predict(X_test)
    CA_predicts['Baseline'][source] = test
    CA_predicts['Baseline'][source]['Predict'] = y_hat
    CA_predicts['Baseline'][source] = CA_predicts['Baseline'][source][['Data', 'Predict']]
    RMSE_dict[state].loc['Baseline', source] = round(RMSE(CA_predicts['Baseline'][source].Data,
                                                          CA_predicts['Baseline'][source].Predict), 6)

# Holt-Winters Triple Exponential Smoothing
CA_predicts['Holt-Winters'] = {}
sources = create_sources_dict()
for source in renewables:
    train = sources[source].iloc[:-12, :].Data
    fit = ExponentialSmoothing(train, seasonal_periods=12, damped=True,  
                        trend='add', seasonal='add').fit()
    y_hat = fit.forecast(12)
    CA_predicts['Holt-Winters'][source] = sources[source].iloc[-12:, :]
    CA_predicts['Holt-Winters'][source]['Predict'] = y_hat
    CA_predicts['Holt-Winters'][source] = CA_predicts['Holt-Winters'][source][['Data', 'Predict']]
    RMSE_dict[state].loc['Holt-Winters', source] = round(RMSE(CA_predicts['Holt-Winters'][source].Data,
                                                              CA_predicts['Holt-Winters'][source].Predict), 6)

# Arima
CA_predicts['ARIMA'] = {}
sources = create_sources_dict()
for source in renewables:
    train = sources[source].iloc[:-12, :].Data
    model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), 
                    enforce_stationarity=False,enforce_invertibility=False).fit()
    results = model.get_prediction('2019-01-01', '2019-12-01', 
               dynamic=True)
    y_hat = results.predicted_mean
    CA_predicts['ARIMA'][source] = sources[source].iloc[-12:, :]
    CA_predicts['ARIMA'][source]['Predict'] = y_hat
    CA_predicts['ARIMA'][source] = CA_predicts['ARIMA'][source][['Data', 'Predict']]
    RMSE_dict[state].loc['ARIMA', source] = round(RMSE(CA_predicts['ARIMA'][source].Data,
                                                       CA_predicts['ARIMA'][source].Predict), 6)

# LSTM Univariate
CA_predicts['LSTM-Uni'] = {}
sources = create_sources_dict()
n_previous = 24
n_future = 12
n_test = n_previous + n_future
for source in renewables:
    tf.keras.backend.clear_session()
    train = sources[source].iloc[:-n_test, :].Data
    test = sources[source].iloc[-n_test:, :].Data
    x_test = test[:-n_future].to_numpy().reshape(1, n_previous, 1)
    model = build_model_uni()
    x_train, y_train = windowize_data(train, n_previous, n_future)
    model.fit(x_train, y_train, batch_size=32, epochs=100)
    y_hat = model.predict(x_test)
    CA_predicts['LSTM-Uni'][source] = sources[source].iloc[-n_future:, :]
    CA_predicts['LSTM-Uni'][source]['Predict'] = y_hat[0]
    CA_predicts['LSTM-Uni'][source] = CA_predicts['LSTM-Uni'][source][['Data', 'Predict']]
    RMSE_dict[state].loc['LSTM Univariate', source] = round(RMSE(CA_predicts['LSTM-Uni'][source].Data,
                                                                 CA_predicts['LSTM-Uni'][source].Predict), 6)

# LSTM Multivariate
CA_predicts['LSTM-Multi'] = {}
sources = create_sources_dict()
n_previous = 24
n_future = 12
n_test = n_previous + n_future
x_train = {}
y_train = {}
x_test = {}
all_sources = [s for s in sources.keys()]
source_ind = {x: i for i, x in enumerate(all_sources)}
for source in all_sources:
    train = sources[source].iloc[:-n_test, :].Data
    test = sources[source].iloc[-n_test:, :].Data
    x_train[source], y_train[source] = windowize_data(train, n_previous, n_future)
    x_test[source] = test[:-n_future].to_numpy().reshape(1, n_previous, 1)
x_train_combined = np.full((157, 24, 11), 1.0)
x_test_combined = np.full((1, 24, 11), 1.0)
for i in range(157):
    for j in range(24):
        temp = []
        for source in all_sources:
            temp.append(x_train[source][i][j][0])
        x_train_combined[i][j] = np.array(temp)
for k in range(24):
    temp2 = []
    for source in all_sources:
        temp2.append(x_test[source][0][k][0])
    x_test_combined[0][k] = np.array(temp2)
for source in renewables:
    not_target = list(sources.keys())
    not_target.remove(source)
    tf.keras.backend.clear_session()
    model = build_model_multi()
    model.fit(x_train_combined, y_train[source], batch_size=32, epochs=100)
    y_hat = model.predict(x_test_combined)
    CA_predicts['LSTM-Multi'][source] = sources[source].iloc[-n_future:, :]
    CA_predicts['LSTM-Multi'][source]['Predict'] = y_hat[0]
    CA_predicts['LSTM-Multi'][source] = CA_predicts['LSTM-Multi'][source][['Data', 'Predict']]
    RMSE_dict[state].loc['LSTM Multivariate', source] = round(RMSE(CA_predicts['LSTM-Multi'][source].Data,
                                                                   CA_predicts['LSTM-Multi'][source].Predict), 6)


# Creating Plots of Holt-Winters predictions compared to the baseline
# for illustrative purposes
title_dict = {'solar': 'CA Electricity Generated from Solar Energy',
              'wind': 'CA Electricity Generated from Wind Energy',
              'hydro': 'CA Electricity Generated from Hydroelectric Energy',
              'bio': 'CA Electricity Generated from Biomass Energy',
              'geo': 'CA Electricity Generated from Geothermal Energy'}
for source in renewables:
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(sources[source].iloc[-60:,:].Data, label='Actual data')
    for model in ['Baseline', 'Holt-Winters']:
        ax.plot(CA_predicts[model][source].Predict, label=model, ls='--')
    ax.grid(alpha=0.5)
    ax.legend()
    ax.set_title(title_dict[source], fontsize=14)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('% of Total Electricity', fontsize=14)
    plt.savefig('images/CA_' + source + '.png')


# Printing Table of RMSE for each model's predicitons of California's Electricity
# Generated from each energy source in 2020.
CA_RMSE = RMSE_dict['CA'].rename({'solar': 'Solar',
                        'wind': 'Wind',
                        'hydro': 'Hydroelectric',
                        'bio': 'Biomass',
                        'geo': 'Geothermal'}, axis=1)
print(CA_RMSE)