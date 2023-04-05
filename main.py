import streamlit as st

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from shapely.geometry import Point
import matplotlib.pyplot as plt

st.write("# Wildfire Data Visualization")
df = pd.read_csv('data_fires.csv')


# frequency of fire over time
df2 = df.groupby(['FIRE_YEAR', 'STATE']).agg('count').reset_index()
st.write("### Number of Fires over Time")
state = st.selectbox('State', df2['STATE'].unique())
df2 = df2.pivot(index='FIRE_YEAR', columns='STATE', values='OBJECTID')
fig, ax = plt.subplots()

# year = st.select_slider('Year', df2['FIRE_YEAR'].unique())
ax.plot(df2[state].index, df2[state].values)
ax.set_xticks(df2[state].index)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylim(0,max(df2[state].values)+100)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Fires')
ax.set_title('Number of Fires in ' + state + ' over time')
st.pyplot(fig)

# ARIMA model   

def train_arima_model(series):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
#     print(model_fit.summary())
    predictions = model_fit.forecast(10)
    # fig = plt.figure(figsize=(18,8))
    plt.xlabel('Year')
    plt.ylabel('Number of Fires')
    plt.title('Number of Fires in ' + state + ' over time')
    plt.ylim(0,max(series)+100)
    plt.plot(list(range(1992, 2016)), series.values, label='history', color='blue')
    plt.plot(list(range(2015, 2026)), [series[2015]] + predictions.tolist(), label='predictions', color='red')
    plt.xticks(list(range(1992, 2026)), rotation=90)
    plt.legend()
    return fig

st.write("### ARIMA model prediction of Number of Fires over Time")
# state = st.selectbox('State', df2['STATE'].unique())

# fig2, ax2 = plt.subplots()
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# usa_map = gpd.read_file("USA_States/USA_States.shp")
# geometry = [Point(xy) for xy in zip(df2['LONGITUDE'], df2['LATITUDE'])]
# gdf = gpd.GeoDataFrame(df2, geometry=geometry) 
# gdf.plot(ax=usa_map.plot(figsize=(40, 24)), marker='o', color='red', markersize=15)

st.pyplot(train_arima_model(df2[state]))