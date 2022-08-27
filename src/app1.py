import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px 
import streamlit as st
import numpy as np 
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import optuna
import statsmodels.api as sm
import datetime
import plotly.offline as py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import plot_yearly
from prophet.plot import add_changepoints_to_plot
from utils import Helpers



stocks = pd.read_csv('src\sp500_stocks.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
broad = stocks.query("Symbol == 'AVGO'")
mono = stocks.query("Symbol == 'MPWR'")
inc = stocks.query("Symbol == 'LRCX'")
broad['Close'].plot(label = "Broadcom Inc.")
mono['Close'].plot(label = 'Monolithic Power Systems')

inc['Close'].plot(label = 'Inc. MPWR Lam Research Corporation LRCX')

st.title('Stock Prices Semiconduction')
st.subheader('Stocks')
plt.legend()
plt.show()

@st.cache
def load_data(): 
    stocks = pd.read_csv('src\sp500_stocks.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
    #EDA Stocks
    stocks.isna().sum()
    stocks[stocks["Adj Close"].isna()]
    stocks[stocks["Adj Close"].isna()].isna().sum()
    stocks = stocks.dropna()
    stocks[stocks['Volume']==0]
    stocks = stocks[stocks['Volume']>0]
    stocks_recet= stocks.reset_index()
    sp=stocks_recet[stocks_recet['Symbol'].isin(("AVGO","MPWR","LRCX"))]
    sp=sp[['Date','Open']]
    sp=sp.groupby('Date').sum('Open')
    sp_recet= sp.reset_index()
    sp = sp_recet.rename(columns={'Date': 'ds','Open': 'y'})
    sp_sample = sp[(sp.ds.dt.year>=2016) & (sp.ds.dt.year<2021)].reset_index(drop=True).sort_values("ds")
    return sp_sample

df_ch = load_data()


model1 = Prophet(interval_width=0.95)
model1.add_country_holidays(country_name='US')
model1.fit(df_ch)
future = model1.make_future_dataframe(periods=30, freq="B")
forecast = model1.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model1.plot(forecast)
final_model2 = Prophet(interval_width=0.95, weekly_seasonality=False, changepoint_prior_scale=0.9)
final_model2.add_seasonality(name='yearly', period=365, fourier_order=8)
final_model2.add_country_holidays(country_name='US')
forecast = final_model2.fit(df_ch).predict(future)
fig = final_model2.plot_components(forecast)
plt.show()


# Heroku uses the last version of python, but it conflicts with 
# some dependencies. Low your version by adding a runtime.txt file
# https://stackoverflow.com/questions/71712258/