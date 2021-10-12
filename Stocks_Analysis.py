#Stocks Analysis

# import all the packege that we need 
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet as pt 
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

#Select a start date and end date 
start = "2017-01-01"
end = date.today().strftime('%Y-%m-%d')
#Setting a title for app
st.title("Stocks Prediction Application")


#setting the stocks that you interested
#OR we can let users to input a correct stock
stocks = ("AAPL","GOOG","MSFT", "GME", "SENS", "AMC")
user_input = st.text_input("Enter Your Stock: ")
stocks = stocks + (user_input,)
#selecting box for select the stock you interested
selected_stock = st.selectbox("Select the stock you want to predict: ", stocks)

#Years for prediction(1 to 4 years)
n_years = st.slider("Years of Preditions: ", 1, 4)
period = n_years*365

#load data method
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace = True)
    return data

#Loading data using the method
load_data_status = st.text("Loading data...")
data = load_data(selected_stock)
load_data_status.text("Loading data...Done")

#Showing the raw data
st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'],y = data['Open'], name = 'Stock_open_price'))
    fig.add_trace(go.Scatter(x = data['Date'],y = data['Close'], name = 'Stock_Close_price'))
    fig.layout.update(title_text = 'Time Series Data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
    
plot_raw_data();

def plot_kline(data):
    fig = go.Figure(go.Candlestick(x = data['Date'], open = data['Open'], high = data['High'],
    low = data['Low'], close = data['Close'], name = 'Stock_Kline'))
    
    fig.layout.update(title_text = 'Time Series Data Candlestick', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
plot_kline(data);
#Forecasting using facebook prophet


df_train_close = data[['Date','Close']]
df_train_close = df_train_close.rename(columns = {"Date": "ds", "Close": "y"})

df_train_open = data[['Date','Open']]
df_train_open = df_train_open.rename(columns = {"Date": "ds", "Open": "y"})

#Creating a fbprophet model for closing
m = pt()
m.fit(df_train_close)

future = m.make_future_dataframe(periods = period)
forecast_data = m.predict(future)

#showing the data
st.subheader('Predicted Data Close Price')
st.write(forecast_data)

#Creating a fbprophet model for opening
m1 = pt()
m1.fit(df_train_open)
future_open = m.make_future_dataframe(periods = period)
forecast_data_open = m1.predict(future_open)

#showing the data
st.subheader('Predicted Data Open Price')
st.write(forecast_data_open)

#making cande stick plots for forecasting data
fig = go.Figure(go.Candlestick(x = forecast_data['ds'], open = forecast_data_open['trend'], high = forecast_data_open['yhat_upper'],
low = forecast_data_open['yhat_lower'], close = forecast_data['trend'], name = 'Stock_Prediction_Kline'))

fig.layout.update(title_text = 'Time Series Data Candlestick with Prediction', xaxis_rangeslider_visible = True)
st.plotly_chart(fig)



fig1 = plot_plotly(m,forecast_data)
st.plotly_chart(fig1)
st.write('forecast component')
fig2 = m.plot_components(forecast_data)
st.write(fig2)




