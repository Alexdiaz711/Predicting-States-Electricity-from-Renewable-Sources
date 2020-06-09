# This script performs analysis on the states' predicitons

from from_SQL import from_SQL, from_SQL_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import datetime as dt
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Importing the data from the PostgreSQL database and storing in a pandas DF.
    print('Importing data...')
    monthly_df, annual_df = from_SQL()
    predict_df = from_SQL_predict()
    renewables = ['solar', 'wind', 'hydro', 'bio', 'geo', 'renewables']
    monthly_renew = monthly_df[monthly_df.SOURCE.isin(renewables)]
    monthly_renew = monthly_renew[monthly_renew.STATE != 'US']
    monthly_renew.reset_index(drop=True, inplace=True)
    joined = monthly_renew.merge(predict_df)
    results_df = joined[['STATE', 'SOURCE']]


    # Fitting linear trends to each prediciton
    print('Fitting linear trends to predicitons...')
    fit_slopes = []
    fit_intercepts = []
    for i in range(joined.shape[0]):
        temp_df = pd.DataFrame(joined.iloc[i, -13:].astype(float)).rename({i: 'Predict'}, axis=1)
        temp_df['Constant'] = 1
        temp_df['Time'] = range(13)
        y = temp_df.pop('Predict').astype(float).to_numpy()
        X = temp_df.astype(float).to_numpy()
        model = sm.OLS(y, X).fit()
        fit_intercepts.append(model.params[0])
        fit_slopes.append(model.params[1])
    results_df['SLOPE'] = fit_slopes
    results_df['INTERCEPT'] = fit_intercepts


    # Printing top 3 states in terms of highest linear fit slope for each renewable resource
    print('Making tables and plots...')
    results_df['Change for 2020'] = results_df.SLOPE * 12
    for source in renewables:
        temp = results_df[results_df.SOURCE == source]
        temp2 = temp[['STATE', 'Change for 2020']].sort_values('Change for 2020', ascending=False)
        states = temp2.STATE
        changes = temp2['Change for 2020']
        print('Largest slope for {} prediction linear fit:'.format(source))
        for i in range(temp2.shape[0]):
            print('State: {}, {:.2} %-points/year'.format(states.iloc[i], changes.iloc[i]*100))
        print()


    # Plotting prediction for solar in CA as an example of analysis
    fig, ax = plt.subplots(figsize=(12,7)) 
    mask1 = joined.STATE == 'CA'
    mask2 = joined.SOURCE == 'solar'
    mask3 = results_df.STATE == 'CA'
    mask4 = results_df.SOURCE == 'solar'
    temp1 = joined[mask1][mask2].iloc[0, -60:-12]
    temp2 = joined[mask1][mask2].iloc[0, -13:]
    temp3 = results_df[mask3][mask4]
    m = temp3.SLOPE.iloc[0]
    b = temp3.INTERCEPT.iloc[0]
    dates = joined.columns[-13:]
    ax.plot(temp1*100, label='Historical Data', alpha=0.65)
    ax.plot(temp2*100, label='Prediction', alpha=0.65, color='g')
    x = np.arange(0,13)
    y = m*x + b
    ax.plot(dates, y*100, label='Prediction Linear Fit', color='r')
    fig.text(.65, .2, s='fit slope = +{:.2f}%/year'
            .format(temp3['Change for 2020'].iloc[0]*100), color='r', fontsize=14)
    ax.grid(alpha=0.35)
    ax.legend(fontsize=12)
    ax.set_title('Solar in California', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('% of State Total Electricity Generated', fontsize=14)
    plt.savefig('images/CA_solar_predict.png')


    # Plotting leading state for each renewable resource
    leaders_dict = {'solar': 'MA',
                    'wind': 'OR',
                    'hydro': 'SD',
                    'bio': 'ME',
                    'geo': 'ID',
                    'renewables': 'SD'}
    titles = ['Solar in Massachusetts', 
            'Wind in Oregon', 
            'Hydroelectric in South Dakota',
            'Biomass in Maine',
            'Geothermal in Idaho',
            'All Renewables in South Dakota']
    text_y = [0.75, 0.61, 0.9, 0.1, 0.35, 0.2]
    text_x = [.13, .52, .85, .2, .52, .85]
    fig, axs = plt.subplots(2,3, figsize=(15,8))
    axs_f = axs.flatten() 
    for i, (k, v) in enumerate(leaders_dict.items()):
        mask1 = joined.STATE == v
        mask2 = joined.SOURCE == k
        mask3 = results_df.STATE == v
        mask4 = results_df.SOURCE == k
        temp1 = joined[mask1][mask2].iloc[0, 2:-12]
        temp2 = joined[mask1][mask2].iloc[0, -13:]
        temp3 = results_df[mask3][mask4]
        m = temp3.SLOPE.iloc[0]
        b = temp3.INTERCEPT.iloc[0]
        dates = joined.columns[-13:]
        axs_f[i].plot(temp1*100, label='Historical Data', alpha=0.65)
        axs_f[i].plot(temp2*100, label='Prediction', alpha=0.65, color='g')
        x = np.arange(0,13)
        y = m*x + b
        axs_f[i].plot(dates, y*100, label='Prediction Linear Fit', color='r')
        fig.text(text_x[i], text_y[i], s='fit slope = +{:.2f}%/year'
                .format(temp3['Change for 2020'].iloc[0]*100), color='r', fontsize=12)
        axs_f[i].grid(alpha=0.35)
        if i == 0:
            axs_f[i].legend(fontsize=12)
        axs_f[i].set_title(titles[i], fontsize=12)
        if i in [3, 4, 5]:
            axs_f[i].set_xlabel('Time', fontsize=12)
        if i == 0 or i ==3:
            axs_f[i].set_ylabel('% of State Total Electricity Generated', fontsize=12)
    fig.tight_layout()
    plt.savefig('images/Leaders.png')