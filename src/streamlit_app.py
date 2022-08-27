import altair as alt
import pandas as pd
import streamlit as st
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
#from vega_datasets import data

st.set_page_config(
    page_title="Time series annotations", page_icon="⬇", layout="centered"
)

@st.experimental_memo
def get_data():
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
    source = sp[(sp.ds.dt.year>=2016) & (sp.ds.dt.year<2021)].reset_index(drop=True).sort_values("ds")
    return source

data = get_data()


@st.experimental_memo(ttl=60 * 60 * 24)
def get_chart(data):
    hover = alt.selection_single(
        fields=["Date"],
        nearest=True,
        on="mouseover",
        empty="none",
    
    )

    lines = (
        alt.Chart(data, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="Date",
            y="Open",
            color="symbol",
            # strokeDash="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y="price",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Date", title="Date"),
                alt.Tooltip("Open", title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


st.title("⬇ Time series annotations")

st.write("Give more context to your time series using annotations!")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Choose a ticker (⬇💬👇ℹ️ ...)", value="⬇")
with col2:
    ticker_dx = st.slider(
        "Horizontal offset", min_value=-30, max_value=30, step=1, value=0
    )
with col3:
    ticker_dy = st.slider(
        "Vertical offset", min_value=-30, max_value=30, step=1, value=-10
    )

# Original time series chart. Omitted `get_chart` for clarity
source = get_data()
chart = get_chart(source)



# Create a chart with annotations


model1 = Prophet(interval_width=0.95)
model1.add_country_holidays(country_name='US')
model1.fit(data)
future = model1.make_future_dataframe(periods=30, freq="B")
forecast = model1.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model1.plot(forecast)
final_model2 = Prophet(interval_width=0.95, weekly_seasonality=False, changepoint_prior_scale=0.9)
final_model2.add_seasonality(name='yearly', period=365, fourier_order=8)
final_model2.add_country_holidays(country_name='US')
forecast = final_model2.fit(data).predict(future)
fig = final_model2.plot_components(forecast)
plt.show()


# Display both charts together
st.altair_chart((chart).interactive(), use_container_width=True)

st.write("## Code")

st.write(
    "See more in our public [GitHub repository](https://github.com/streamlit/example-app-time-series-annotation)"
)

st.code(
    f"""
import altair as alt
import pandas as pd
import streamlit as st
from vega_datasets import data
@st.experimental_memo
def get_data():
    source = data.stocks()
    source = source[source.date.gt("2004-01-01")]
    return source
source = get_data()
# Original time series chart. Omitted `get_chart` for clarity
chart = get_chart(source)
# Input annotations
ANNOTATIONS = [
    ("Mar 01, 2008", "Pretty good day for GOOG"),
    ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
    ("Nov 01, 2008", "Market starts again thanks to..."),
    ("Dec 01, 2009", "Small crash for GOOG after..."),
]
# Create a chart with annotations
annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
annotations_df.date = pd.to_datetime(annotations_df.date)
annotations_df["y"] = 0
annotation_layer = (
    alt.Chart(annotations_df)
    .mark_text(size=15, text="{ticker}", dx={ticker_dx}, dy={ticker_dy}, align="center")
    .encode(
        x="date:T",
        y=alt.Y("y:Q"),
        tooltip=["event"],
    )
    .interactive()
)
# Display both charts together
st.altair_chart((chart + annotation_layer).interactive(), use_container_width=True)
""",
    "python",
)



