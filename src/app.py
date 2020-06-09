import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from from_SQL import from_SQL, from_SQL_predict
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.tools import mpl_to_plotly
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

def cloropleth(energy_source):
    fig = go.Figure()
    for date in annual[energy_source].columns:
        fig.add_trace(
            go.Choropleth(
                locations=annual[energy_source][annual[energy_source].index != 'US'].index,
                z=round(annual[energy_source][annual[energy_source].index != 'US'][date]*100, 2),
                locationmode='USA-states',
                colorscale='Greens',
                autocolorscale=False,
                marker_line_color='grey', # line markers between states
                colorbar_title="%",
                colorbar_x=0,
                colorbar_thickness=15,
                zmin=0,
                zmax=annual[energy_source].max().max() * 100,
                name=date)
        )   
    steps = []
    for i in range(len(annual[energy_source].columns)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(annual[energy_source].columns)}],
            label='{}'.format(i + 2001)
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)  
    sliders = [dict(
        active=18,
        currentvalue={"prefix": "Year Selected: "},
        pad={"t": 9},
        steps=steps)
                ]
    fig.update_layout(
        sliders=sliders,
        title_text='Percent of Annual Electricity Generated, {} Energy'
                    .format(energy_names[energy_source]),
        geo = dict(
            scope='usa',
            projection=go.layout.geo.Projection(type = 'albers usa'),
            showlakes=False,
            lakecolor='rgb(255, 255, 255)')
        )
    return fig

def plots(source, state_list):
    fig, ax = plt.subplots()
    for i, state in enumerate(state_list):
        mask1 = joined.STATE == state
        mask2 = joined.SOURCE == source
        mask3 = results_df.STATE == state
        mask4 = results_df.SOURCE == source
        temp1 = joined[mask1][mask2].iloc[0, 2:-12]
        temp2 = joined[mask1][mask2].iloc[0, -13:]
        temp3 = results_df[mask3][mask4]
        m = temp3.SLOPE.iloc[0]
        b = temp3.INTERCEPT.iloc[0]
        dates = joined.columns[-13:]
        ax.plot(temp1*100, label=states[state], alpha=0.65)
        ax.plot(temp2*100, alpha=0.65, color='g')
        x = np.arange(0,13)
        y = m*x + b
        ax.plot(dates, y*100, color='r')
        ax.grid(alpha=0.35)
        ax.legend(fontsize=12)
        ax.set_title('State Comparison', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('% of State Total Electricity Generated', fontsize=12)
    return fig

monthly_df, annual_df = from_SQL()
predict_df = from_SQL_predict()
renewables = ['solar', 'wind', 'hydro', 'bio', 'geo', 'renewables']
monthly_renew = monthly_df[monthly_df.SOURCE.isin(renewables)]
monthly_renew = monthly_renew[monthly_renew.STATE != 'US']
monthly_renew.reset_index(drop=True, inplace=True)
joined = monthly_renew.merge(predict_df)
results_df = joined[['STATE', 'SOURCE']]

monthly = {}
for source in renewables:
    monthly[source] = (joined[joined.SOURCE==source].set_index('STATE')
                                                    .drop('SOURCE', axis=1)
                                                    .astype(float)
                        )
                        
annual = {}
for source in renewables:
    annual[source] = (annual_df[annual_df.SOURCE==source].set_index('STATE')
                                                            .drop('SOURCE', axis=1)
                                                            .astype(float)
                        )

states = {'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 
        'GA': 'Georgia', 'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 
        'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 
        'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 
        'MN': 'Minnesota', 'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana', 
        'NC': 'North Carolina', 'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 
        'NJ': 'New Jersey', 'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 
        'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennesee', 'TX': 'Texas', 
        'UT': 'Utah', 'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin',
        'WV': 'West Virginia', 'WY': 'Wyoming', 'AZ': 'Arizona'
        }

energy_names = {'solar': 'Solar',
                'wind': 'Wind',
                'hydro': 'Hydroelectric',
                'bio': 'Biomass',
                'geo': 'Geothermal',
                'renewables': 'All Renewable'}
energy_abr = {v: k for k, v in energy_names.items()}

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

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("States Renewable Energy Data Dashboard",
            style={
                'textAlign': 'center'
            })
    ]),

    html.Div([
        html.Div([
            html.Label('Select Energy Source:'),
            dcc.Dropdown(
                        id='renewable_source',
                        options=[{'label': v, 'value': k} for k, v in energy_names.items()],
                        value='renewables',
                        clearable=False)
                ], 
                    style={'width': '40%', 'display': 'inline-block'}
        ),
        html.Div([
            html.Label('Select States to Compare:',),    
            dcc.Dropdown(
                        id='states',
                        options=[{'label': v, 'value': k} for k, v in states.items()],
                        value='CA',
                        clearable=True,
                        multi=True)
                ],
                    style={'width': '40%', 'float': 'right', 'display': 'inline-block'}
        )
    ]),

    html.Div([
        dcc.Graph(
            id='cloropleth',
            figure=cloropleth('renewables'),
            style={'width': '50%', 'display': 'inline-block'}
        ),
        dcc.Graph(
            id='time_series',
            figure=mpl_to_plotly(plots('renewables', ['CA'])),
            style={'width': '50%', 'float': 'right', 'display': 'inline-block'}
        )
    ])
])

@app.callback(Output('cloropleth', 'figure'),
              [Input('renewable_source', 'value')])
def update_cloropleth(value):
    return cloropleth(value)

@app.callback(Output('time_series', 'figure'),
              [Input('renewable_source', 'value'),
               Input('states', 'value')])
def update_time_series(value1, value2):
    return mpl_to_plotly(plots(value1, value2))

if __name__ == "__main__":
    app.run_server(host='127.0.0.1', debug=True, dev_tools_hot_reload=False)