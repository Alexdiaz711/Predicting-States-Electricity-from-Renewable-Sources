# <div align="center">Predicting States' Renewable Energy Production</div>
#### <div align="center">by Alex Diaz-Clark</div>
A project that includes training and evaluating various models used to predict the percentage of electricity generated from renewable resources for each state. 

## Background

Most states claim to be making efforts to shift their resources used to produce electricity away from fossil fuels and other non-renewable resources. This project aims to identify the states which have had success in their efforts, and forcast the percentage of electricity generated from renewable resources for each state for the year 2020. 

The goals of this project are as follows:
* Develop a time-series forcasting model to be used for predicting the percentage of electricity generated from renewable resources for each state based on monthly data from the years 2001-2019.   
* Use the model to predict the monthly percentage of electricity generated from renewable resources for each state for the year 2020.
* Analyze the 2020 forecasts to identify which states are poised to increase it's percentage of electricity generated from each renewable resource in 2020.

## The Data

The dataset used for this project was downloaded from the Energy Information Administration (EIA), a part of the US Department of Energy. The dataet can be downloaded by clicking [here](https://www.eia.gov/electricity/data/state/generation_monthly.xlsx "Download dataset from EIA"), or navigating to the [EIA's Detailed State Data page for electricity](https://www.eia.gov/electricity/data/state/ "EIA webpage") (just incase the download link gets changed).

The dataset contains data for every state's electricity generation aggregated by state, month, energy source, and type of producer. Contained is the amount of electricity generated in Megawatt-hours for every month for the years between 2001 and 2019 (inclusive of both). The total amount of electricy generated (all types of producers) was aggregated by state, month, and energy source after it was downloaded. Next, the data from each energy source, for each state was tranformed to be a percentage of that states total electricity generation. As an example, the time series for the percentage of total electricity generation that is from renewable resources for California is shown below. 

<p align="center">
<img src="images/CA_energy_profile.png">
</p>

After tranforming the data to a percentage of the state's total electricity generated, an SQL database was created in a PostgreSQL server running in a Docker container. The SQL database was populated with the data awaiting future retrieval.

The script which executes all of the data processing, creates the SQL databse, and populates the SQL databse can be found at '/src/Jdatabase.py' in this repository.

## 







