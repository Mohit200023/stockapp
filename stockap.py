#import library
import streamlit as st
from  fbprophet import Prophet
import yfinance as yf 
from datetime import date 
from  fbprophet.plot import plot_plotly
from plotly import graph_objs as go

#date from which we want to collect data

START = '2014-01-01'
TODAY = date.today().strftime('%Y-%m-%d')


st.sidebar.title("Stock Information")
tickersymbols= st.sidebar.selectbox("Ticker",["AAPL","GOOG","MSFT","GME","SBIN.NS","TTM","TCS.NS"])
tickerdata= yf.Ticker(tickersymbols)


string_logo ='<img src =%s>' % tickerdata.info['logo_url']
st.sidebar.markdown(string_logo, unsafe_allow_html=True)


string_name = tickerdata.info['longName']
st.sidebar.header('**%s**'% string_name)


string_summary = tickerdata.info['longBusinessSummary']
st.sidebar.info(string_summary)

st.title("Stock Prediction App")

#selectbox using ticker
stocks=("AAPL","GOOG","MSFT","GME","SBIN.NS","TTM","TCS.NS")
selected_stcoks= st.selectbox("Select dataset for prediction",stocks)


#selction bar of year for forcasting data 
n_years = st.slider('Year of Prediction:',1 , 4)
period = n_years*365

#data caputering 
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data 

#styling     
data_load_state = st.text("Loading data......")
data= load_data(selected_stcoks)
data_load_state.text ("Loading data......done!")

#data represatation 
st.subheader('Raw Data')
st.write(data.tail())


#fuction for graph
def plot_raw_data():
    fig= go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y= data['Open'],name = 'stock_open '))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],name = 'stock_close '))
    fig.layout.update(title_text ='Time Series Data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
plot_raw_data()


#ML or forcasting
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={'Date':"ds",'Close':"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#graph representation  of forecast data
st.subheader("Forecast data")
st.write(forecast.tail())

st.subheader('Forecast data')
Fig1= plot_plotly(m,forecast)
st.plotly_chart(Fig1)