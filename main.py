import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

#RFR : risk free rate
# V - volatility
#N - number of sims
#T time in years until expiration

def montecarlosim(StockPrice, RFR, V, T, step, N, StrikePrice):
    dt = T/ step
    Z = np.random.standard_normal((N, step)) #create a matrix of size (N, step) filled with numbers drawn from a standard normal distribution
    W = np.cumsum(np.sqrt(dt)*Z, axis = 1)

    timestep = np.linspace(0,T,step+1) # Create N+1 points between 0 and 1 which will correspond to the number of values computed per MC sim
    S = np.zeros((N,step+1)) # sets up a matric of size N, step+1 filled with 0s
    S[:,0] = StockPrice 

    for i in range(1,step+1):
        S[:,i] = S[:,i-1] * np.exp((RFR - 0.5 * V ** 2) * dt + V * (W[:,i-1]-W[:,i-2] if i>1 else 0)) # Brownian Motion Formula

    endprices = S[:, -1]

    call_payout = np.maximum(endprices-StrikePrice, 0)
    put_payout = np.maximum(StrikePrice - endprices, 0)

    discountedcallpayout = np.exp(-RFR*T) * call_payout
    discountedputpayout = np.exp(-RFR*T)* put_payout

    callprice = np.mean(discountedcallpayout)
    putprice= np.mean(discountedputpayout)

    return(timestep, S.T, callprice, putprice)
    
st.title('Interactive Monte Carlo stock price simulator')

st.sidebar.header('Contract Input Parameters')
currentstockprice = st.sidebar.number_input('Current Stock Price', value = 100)
Riskfreerate = st.sidebar.number_input('Risk Free Rate', value = 0.05)
Volatility = st.sidebar.number_input('Volatility (0-1)', value= 0.2)
timetomaturity = st.sidebar.number_input('Time to Maturity (Years)', value = 1)
NoSims = st.sidebar.number_input('Number of simulations', value = 1000)
StrikePrice = st.sidebar.number_input('Strike Price of Option',value = 100)

timestep, S_T, callprice, putprice = montecarlosim(currentstockprice,Riskfreerate,Volatility, timetomaturity, 252, NoSims, StrikePrice)


plt.figure(figsize=(10,6))
plt.plot(timestep, S_T, linewidth = 0.5)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title("Monte Carlo simulation of stock prices")
plt.grid(True)

st.pyplot(plt.gcf())

column1, column2 = st.columns(2)


with column1:
    st.subheader('Monte Carlo Simulation')
    st.write(f'**Call Price**: {callprice:.2f}')
    st.write(f'**Put Price**: {putprice:.2f}')

def blachscholes(Stockprice, Strikeprice, RiskFreeRate, timetomaturity, Volatility):
    d1 = (np.log(Stockprice/Strikeprice)+ (RiskFreeRate+0.5 * Volatility**2)* timetomaturity)/ (Volatility*np.sqrt(timetomaturity))
    d2 = d1- Volatility * np.sqrt(timetomaturity)

    callprice = Stockprice * norm.cdf(d1)  - Strikeprice * np.exp(-RiskFreeRate*timetomaturity) * norm.cdf(d2)
    putprice = Strikeprice * np.exp(-Riskfreerate*timetomaturity) * norm.cdf(-d2) - Stockprice * norm.cdf(-d1)

    return(callprice, putprice)

callprice, putprice = blachscholes(currentstockprice,StrikePrice, Riskfreerate, timetomaturity, Volatility)

with column2:
    st.subheader("Black Scholes Valuation")
    st.write(f'**Call Price**: {callprice:.2f}')
    st.write(f'**Put Price** : {putprice:.2f}')










