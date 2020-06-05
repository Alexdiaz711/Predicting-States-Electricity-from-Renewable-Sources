# <div align="center">Predicting States Electricity Generated from Renewables</div>
#### <div align="center">by Alex Diaz-Clark</div>
A project that includes training and evaluating various models used to predict the percentage of electricity generated from renewable resources for each state. 

## Background

Most states claim to be making efforts to shift their resources used to produce electricity away from fossil fuels and other non-renewable resources. This project aims to identify the states which have had success in their efforts, and forcast the percentage of electricity generated from renewable resources for each state for the year 2020. 

The goals of this project are as follows:
* Develop a time-series forcasting model to be used for predicting the percentage of electricity generated from renewable resources for each state based on monthly data from the years 2001-2019.   
* Use the model to predict the monthly percentage of electricity generated from renewable resources for each state for the year 2020.
* Analyze the 2020 forecasts to identify which states are poised to make the most or least progress towards independence from non-renewable resources for electricity generation.
* Create an interactive data dashboard where users can explore the data including the 2020 forecasts.
* Create a webpage to host the interactive data dashboard so users from anywhere can access it.

## The Data

The dataset used for this project was downloaded from the Energy Information Administration (EIA), a part of the US Department of Energy. The dataet can be downloaded by clicking [here](https://www.eia.gov/electricity/data/state/generation_monthly.xlsx "Download dataset from EIA"), or navigating to the [EIA's Detailed State Data page for electricity](https://www.eia.gov/electricity/data/state/ "EIA webpage") (just incase the download link gets changed).

The dataset contains data for every state's electricity generation aggregated by state, month, energy source, and type of producer. Contained is the amount of electricity generated in Megawatt-hours for every month for the years between 2001 and 2019 (inclusive of both). The total amount of electricy generated (all types of producers) was aggregated by state, month, and energy source after it was downloaded. The electricity generation profile changing in time for a single state is shown below. 





