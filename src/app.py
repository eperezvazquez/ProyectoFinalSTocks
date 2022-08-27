#Aplicamos los imports primero
# linear algebra
import numpy as np 

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# stocks related missing info
import yfinance as yf

# ignoring the warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#ranking the stocks
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import optuna

# Evaluar si las elimino
import statsmodels.api as sm

#Timer series
import datetime

#!pip install fbprophet --quiet
import plotly.offline as py

#Guardar modelo
import pickle

#Aplicamos los from luego de los imports

# Evaluar si las elimino
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#ranking the stocks
from plotly.subplots import make_subplots

#Prophet Model Stuff
#!pip install fbprophet --quiet

from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import plot_yearly
from prophet.plot import add_changepoints_to_plot

#Aplicando Utils
from utils import Helpers


SP500_Comp = pd.read_csv('src\sp500_companies.csv',parse_dates=[0], infer_datetime_format=True,index_col=0)
price = pd.read_csv('src\sp500_index.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
stocks = pd.read_csv('src\sp500_stocks.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

#EDA 500 companies
SP500_Comp[SP500_Comp['State'].isnull()].head(3)
SP500_Comp = SP500_Comp.drop(['State'], axis=1)
SP500_Comp.head(3)
SP500_Comp[SP500_Comp['Revenuegrowth'].isnull()]
helpers  = Helpers()
helpers.replace_null(SP500_Comp,'CVS', 'Revenuegrowth', 'revenueGrowth')
SP500_Comp[SP500_Comp['Symbol'] == 'CVS']
SP500_Comp[SP500_Comp['Fulltimeemployees'].isnull()]
SP500_Comp.loc[SP500_Comp['Fulltimeemployees'].isnull(), 'Fulltimeemployees'] = SP500_Comp['Fulltimeemployees'].mode()[0]
SP500_Comp[SP500_Comp['Fulltimeemployees'].isnull()]
SP500_Comp[SP500_Comp['Ebitda'].isnull()].head(3)
missing_EBITDA = SP500_Comp[SP500_Comp['Ebitda'].isnull()]
count_EBITDA = missing_EBITDA.groupby(['Sector', 'Industry'])['Industry'].count()
count_EBITDA
Financial_companies = SP500_Comp[SP500_Comp['Sector']=='Financial Services']
count_fc = Financial_companies.groupby(['Sector', 'Industry'])['Industry'].count()
count_fc
for col in SP500_Comp.columns:
    b = SP500_Comp[col].unique()
    if len(b)<20:
        print(f'{col} has {len(b)} unique values -->> {b}', end = '\n\n')
SP500_Comp.isna().sum()
SP500_Comp = SP500_Comp.fillna(0)
df_indice_recet= SP500_Comp.reset_index()
helpers  = Helpers()
df_nuevo =helpers.remover_outliers('Currentprice', df_indice_recet,umbral = 1.5)
df_max=df_indice_recet[df_indice_recet['Currentprice']>412]
df_max[df_max['Sector']=='Technology'].sort_values(by=['Sector','Currentprice'],ascending=False)
#EDA INDEX
price.isna().sum()
#EDA Stocks
stocks.isna().sum()
stocks[stocks["Adj Close"].isna()]
stocks[stocks["Adj Close"].isna()].isna().sum()
stocks = stocks.dropna()
stocks[stocks['Volume']==0]
stocks = stocks[stocks['Volume']>0]
sns.set_style('darkgrid')
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (20, 9)
plt.rcParams['figure.facecolor'] = '#00000000'
plt.rcParams['lines.linewidth'] = 2
# plt.style.use('fivethirtyeight')
broad = stocks.query("Symbol == 'AVGO'")
mono = stocks.query("Symbol == 'MPWR'")
inc = stocks.query("Symbol == 'LRCX'")
broad['Close'].plot(label = "Broadcom Inc.")
mono['Close'].plot(label = 'Monolithic Power Systems')

inc['Close'].plot(label = 'Inc. MPWR Lam Research Corporation LRCX')

plt.title('Stock Prices Semiconduction')
plt.legend()
plt.show()


fig = go.Figure(data=[go.Candlestick(x=inc.index,
                open=inc['Open'],
                high=inc['High'],
                low=inc['Low'],
                close=inc['Close'])])
fig.update_layout(
    title='Inc. MPWR Lam Research Corporation LRCX Stock Chart',
    yaxis_title='Inc Stock',
)

broad['Volume'].plot(label = "Broadcom Inc.", figsize = (18, 9))
mono['Volume'].plot(label = 'Monolithic Power Systems Inc. MPWR ')
inc['Volume'].plot(label = 'Lam Research Corporation LRCX')

plt.title('Volume of stock traded')
plt.legend()

broad.iloc[broad['Volume'].argmax()]
broad.iloc[255:265].Open.plot(figsize = (15, 7))
broad['TotalTraded'] = broad['Open'] * broad['Volume']
mono['TotalTraded']= mono['Open'] * mono['Volume']
inc['TotalTraded']= inc['Open'] * inc['Volume']
broad['TotalTraded'].plot(label = "Broadcom Inc.", figsize = (18, 9))
mono['TotalTraded'].plot(label = 'Monolithic Power Systems Inc. MPWR')
inc['TotalTraded'].plot(label = 'Lam Research Corporation LRCX')

plt.title('Total Traded Amount')
plt.legend()
broad.iloc[broad['TotalTraded'].argmax()]
broad['returns'] = (broad['Close']/broad['Close'].shift(1))-1
mono['returns'] = (mono['Close']/mono['Close'].shift(1))-1
inc['returns'] = (inc['Close']/inc['Close'].shift(1))-1
plt.hist(broad['returns'], alpha=0.5, label='Broadcom Stocks Returns', bins=50, color='b')
plt.hist(mono['returns'], alpha=0.5, label='Monolithic Stocks Returns', bins=50, color='r')
plt.hist(inc['returns'], alpha=0.5, label='Lam Research Stocks Returns', bins=50, color='g')
plt.legend()
plt.rcParams['figure.figsize'] = (12, 9)

gnt_returns = pd.concat([broad['returns'], mono['returns'], inc['returns']], axis=1)
gnt_returns.columns = ['Broadcom Inc', 'Monolithic Power', 'Lam Research']

gnt_returns.rolling(window=30).std().plot(figsize=(20, 10), title="30 Day Rolling Standard Deviation");

#APPLICATION MODEL
price_recet= price.reset_index()
stocks_recet= stocks.reset_index()
price_recet['Date'] = pd.to_datetime(price_recet['Date'])
stocks_recet['Date'] = pd.to_datetime(stocks_recet['Date'])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    plt.figure(figsize=(16,5))
    plt.title("S&P 500 Index Value --- {} to {}".format(price_recet['Date'].min().date(), price_recet['Date'].max().date()))
    plt.plot(price_recet['Date'], price_recet['S&P500'])
    plt.locator_params(axis="x", nbins=15)
  
sp = price_recet.rename(columns={'Date': 'ds','S&P500': 'y'})
sp_sample = sp[(sp.ds.dt.year>=2016) & (sp.ds.dt.year<2021)].reset_index(drop=True).sort_values("ds")

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    plt.figure(figsize=(16,5))
    plt.title("S&P 500 Index Value --- {} to {}".format(sp_sample['ds'].min().date(), sp_sample['ds'].max().date()))
    plt.plot(sp_sample['ds'].dt.to_pydatetime(), sp_sample['y'])
   

sp = price_recet.rename(columns={'Date': 'ds','S&P500': 'y'})
sp_sample = sp[(sp.ds.dt.year>=2016) & (sp.ds.dt.year<2021)].reset_index(drop=True).sort_values("ds")

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    plt.figure(figsize=(16,5))
    plt.title("S&P 500 Index Value --- {} to {}".format(sp_sample['ds'].min().date(), sp_sample['ds'].max().date()))
    plt.plot(sp_sample['ds'].dt.to_pydatetime(), sp_sample['y'])

model1 = Prophet(interval_width=0.95)
model1.add_country_holidays(country_name='US')
model1.fit(sp_sample)

future = model1.make_future_dataframe(periods=30, freq="B")
forecast = model1.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

final_model = Prophet(interval_width=0.95, weekly_seasonality=False, changepoint_prior_scale=0.9)
final_model.add_seasonality(name='yearly', period=365, fourier_order=8)
final_model.add_country_holidays(country_name='US')
forecast = final_model.fit(sp_sample).predict(future)
fig = final_model.plot(forecast)

broad = stocks_recet.query("Symbol == 'AVGO'")
mono = stocks_recet.query("Symbol == 'MPWR'")
inc = stocks_recet.query("Symbol == 'LRCX'")
sp=stocks_recet[stocks_recet['Symbol'].isin(("AVGO","MPWR","LRCX"))]
print (sp['Symbol'].value_counts())
sp=sp[['Date','Open']]
sp=sp.groupby('Date').sum('Open')
sp_recet= sp.reset_index()
model1 = Prophet(interval_width=0.95)
model1.add_country_holidays(country_name='US')
model1.fit(sp_sample)
future = model1.make_future_dataframe(periods=30, freq="B")
forecast = model1.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model1.plot(forecast)
final_model2 = Prophet(interval_width=0.95, weekly_seasonality=False, changepoint_prior_scale=0.9)
final_model2.add_seasonality(name='yearly', period=365, fourier_order=8)
final_model2.add_country_holidays(country_name='US')
forecast = final_model2.fit(sp_sample).predict(future)
fig = final_model2.plot(forecast)

# Save the model
filename ='models\TimeSeries.sav'
pickle.dump(final_model2, open(filename, 'wb'))

