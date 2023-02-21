import streamlit as st
import datetime

import yfinance as yf
import pandas_datareader as web
import pandas as pd

from plotly import graph_objs as go

from fbprophet import Prophet
from fbprophet.plot import plot_plotly

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#---defining the app
# st.set_page_config(layout="wide")
st.title('Stock analysing app')

#---side bar set-up
st.sidebar.header('Choose the company for analysis')

#company choose
stocks = ('GOOG', 'AAPL', 'MSFT', 'TSLA')
selected_stock = st.sidebar.selectbox('Select company', stocks)

#date period choose
start_date = st.sidebar.date_input("Start date", datetime.date(2012, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2019, 12, 17))

#period of prediction choose
n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365


#Forecast method select
method = st.sidebar.radio('Forecast method', 
                          ('Additive regression model', 
                           'Neural network'))


#getting the data
# @st.cache
def load_data(ticker):
    df = web.DataReader(ticker, 
                        data_source='yahoo', 
                        start = start_date.strftime("%Y-%m-%d"), 
                        end = end_date.strftime("%Y-%m-%d"))
    df.reset_index(inplace=True)
    return df

#table
data = load_data(selected_stock)
st.subheader('Raw data')
st.write(data.tail())


#Figuring the  ema
df = data[['Close']]
df.reset_index(level=0, inplace=True)
df.columns=['ds','y']

exp1 = df.y.ewm(span=20, adjust=False).mean()
exp2 = df.y.ewm(span=50, adjust=False).mean()


#draw a plot
fig = go.Figure()
#fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
fig.add_trace(go.Scatter(x=data['Date'], y=exp1, name='EMA 20 Day'))
fig.add_trace(go.Scatter(x=data['Date'], y=exp2, name='EMA 50 Day'))
fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Predict forecast with Prophet.
st.subheader('Forecast data')
#if method == 'Neural network':
if method == 'Additive regression model':
    with st.spinner('Please wait...'):
        #data opt
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        
        #make forecast
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        
        

        # Show and plot forecast
        st.write(forecast.tail())

        #Show the type of the forecast

        # st.write(type(forecast))
        
        # Show forecast graph
        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)


        #calc the error
        # sd = datetime.date(2019,12,18)
        # fd = datetime.date(2020,12,18)
        # fc = web.DataReader(selected_stock, data_source='yahoo', start = sd.strftime("%Y-%m-%d"), end = fd.strftime("%Y-%m-%d"))
        
#Predic forecast with LSTM
elif method == 'Neural network':
#if method == 'Additive regression model':
    with st.spinner('Please waitl...'):
        #filter data
        datas = data.filter(['Close'])
        dataset = datas.values
        training_data_len = len(dataset)
        
        
        #scale data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        
        #create training data set
        train_data = scaled_data[0:training_data_len, :]
        x_train = []
        y_train = []
        
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i,0])
            
        x_train, y_train = np.array(x_train),np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
        
        #build model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences= False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        #compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')


        #Train the model
        model.fit(x_train,y_train, batch_size=1, epochs=1)
        
            
        new_df = data.filter(['Close'])
        last_60_days = new_df[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        st.metric(label="Forecast for the next day", value=pred_price)
        #print(pred_price)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
         
        
        
        
        
