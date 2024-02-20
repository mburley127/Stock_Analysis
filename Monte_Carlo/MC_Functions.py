### Library Import Initialization
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


### Function to Import Stock Tickers
def import_stock_data(tickers, start_date):
    data = pd.DataFrame()
    if len([tickers]) == 1:
        data[tickers] = yf.download(tickers, start_date)['Adj Close']
        data = pd.DataFrame(data)
    else:
        for t in tickers:
            data[t] = yf.download(tickers, start_date)['Adj Close']
    return data


### Function to compute Daily Log Returns
def log_returns(data):
    return (np.log(1+data.pct_change()))


### Function to Compute Simple Returns
def simple_returns(data):
    return ((data/data.shift(1))-1)


### Function to Calculate Drift 
def drift_calc(data, return_type = 'log'):
    if return_type == 'log':
        ret = log_returns(data)
    elif return_type == 'simple':
        ret = simple_returns(data)
    
    # Drift calculation - drift represents the expected or average return of the asset over time
    u = ret.mean()
    var = ret.var()
    drift = u - (0.5*var)

    try:
        return drift.values
    except:
        return drift
    

### Function to Compute Daily Returns
def daily_returns(data, days, iterations, return_type = 'log'):
    drift = drift_calc(data, return_type)
    if return_type == 'log':
        try:
            stdev = log_returns(data).std().values
        except:
            stdev = log_returns(data).std()
    elif return_type == 'simple':
        try:
            stdev = simple_returns(data).std().values
        except:
            stdev = simple_returns(data).std()    
    # This distribution is called cauchy distribution
    # 1. np.random.rand(days, iterations): Generates an array of random numbers drawn from a uniform distribution between 0 and 1
    # 2. norm.ppf(): applies the inverse CDF of the standard normal dist to transform the random numbers to follow a normal distribution
    # 3. stdev * norm.ppf(np.random.rand(days, iterations)): Scales the randomly generated normal-dist values by the standard deviation of the asset's log returns
    Z = norm.ppf(np.random.rand(days, iterations))
    daily_rets = np.exp(drift + stdev * Z) # GBM equation
    return daily_rets


### Function to compute and combine market returns to stock data
def market_data_combination(data, mark_ticker = "^GSPC", start='2010-1-1'):
    market_data = import_stock_data(mark_ticker, start)
    # Computes the logarithmic returns of the market data and remove any NaN values
    market_rets = log_returns(market_data).dropna()
    # Calulate the annualized return of the market index. by computing the mean return of the market data (presumably on a daily basis), 
    # multiplies it by 252 (the number of trading days in a year), exponentiates the result (to convert it back to a percentage), and 
    # subtracts 1 to get the percentage return
    ann_return = np.exp(market_rets.mean()*252).values - 1
    # Merge the existing stock data and the market data
    data = data.merge(market_data, left_index=True, right_index=True)
    return data, ann_return


### Compute CAPM and Sharpe Ratio (Allows us to compare each stock to the market)
def beta_sharpe(data, mark_ticker = '^GSPC', start='2010-1-1', riskfree = 0.025):
    """
    Input: 
    1. data: dataframe of stock price data
    2. mark_ticker: ticker of the market data you want to compute CAPM metrics with (default is ^GSPC)
    3. start: data from which to download data (default Jan 1st 2010)
    4. riskfree: the assumed risk free yield (US 10 Year Bond is assumed: 2.5%)
    
    Output:
    1. Dataframe with CAPM metrics computed against specified market procy
    """
    ## Beta
    # Beta = Cov(R_i,R_m) / Var(R_m)
    # cov(R_i,R_m) = covariance between stock returns R_i and mkt returns R_m
    # var(R_m) = variance of mkt returns R_m
    dd, mark_ret = market_data_combination(data, mark_ticker, start)
    # Compute log return covariance
    log_ret = log_returns(dd)
    cov = log_ret.cov()*252 # Annualize for 252 trading days
    cov = pd.DataFrame(cov.iloc[:-1,-1])
    # Compute log return covariance
    mrk_var = log_ret.iloc[:,-1].var()*252
    beta = cov / mrk_var
    # Compute std dev of returns for each asset, annualizes it for 252 trading days), and stores result in a DataFrame 
    stdev_ret = pd.DataFrame(((log_ret.std()*252**0.5)[:-1]), columns = ['STD'])
    # Merge beta values with the std dev of returns
    beta = beta.merge(stdev_ret, left_index = True, right_index = True)
    
    ## CAPM
    # R_p = R_f + Beta_i(R_m - R_f)
    # R_p = Expected Return of the portfolio
    # R_f = Risk free interest rate (2.5%)
    for i, row in beta.iterrows():
        beta.at[i,'CAPM'] = riskfree + (row[mark_ticker] * (mark_ret - riskfree)) # row[mark_ticker] * (mark_ret - riskfree) calculates the risk premium of the asset

    ## Sharpe Ratio
    # Sharpe Ratio = (R_p - R_f) / sigma_p
    # sigma_p = Std dev of expected portfolio return
    for i, row in beta.iterrows():
        beta.at[i,'Sharpe'] = ((row['CAPM']-riskfree)/(row['STD']))
    beta.rename(columns={'^GSPC':"Beta"}, inplace=True)
    
    return beta


### Function to calculated the probability of a stock being higher than a chosen value
def probs_find(predicted, higherthan, on = 'value'):
    """
    This function calculated the probability of a stock being above a certain threshhold, which can be defined as a value (final stock price) or return rate (percentage change)
    Input: 
    1. predicted: dataframe with all the predicted prices (days and simulations)
    2. higherthan: specified threshhold to which compute the probability (ex. 0 on return will compute the probability of at least breakeven)
    3. on: 'return' or 'value', the return of the stock or the final value of stock for every simulation over the time specified
    4. ticker: specific ticker to compute probability for
    """
    if on == 'return':
        predicted0 = predicted.iloc[0,0]
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [(i*100) / predicted0 for i in predList if (( i - predicted0)*100) / predicted0 >= higherthan]
        less = [(i*100) / predicted0 for i in predList if ((i - predicted0)*100) / predicted0 < higherthan]
    elif on == 'value':
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [i for i in predList if i >= higherthan]
        less = [i for i in predList if i < higherthan]
    else:
        print("'on' must be either value or return")
    return (len(over) / (len(over) + len(less)))


### Function to multiply the daily return with the price of the stock of the previous day
def simulate_mc(data, days, iterations, return_type='log', plot=True):
    # Generate daily returns
    returns = daily_returns(data, days, iterations, return_type)
    # Create empty matrix the same size as returns
    price_list = np.zeros_like(returns)
    # Put the last actual price in the first row of matrix
    price_list[0] = data.iloc[-1]
    # Calculate current day asset price by multiplying the previous day (price_list[t-1]) by the return for the current day (returns[t])
    for t in range(1,days):
        price_list[t] = price_list[t-1]*returns[t]
    
    # Plot Option
    if plot == True:
        x = pd.DataFrame(price_list).iloc[-1]
        fig, ax = plt.subplots(1,2, figsize=(14,4))
        sns.distplot(x, ax = ax[0])
        sns.distplot(x, hist_kws = {'cumulative': True}, kde_kws = {'cumulative': True}, ax = ax[1])
        plt.xlabel("Stock Price")
        plt.show()
    
    # Print desired asset info
    try:
        [print(nam) for nam in data.columns]
    except:
        print(data.name)
    
    # Print Number of Days Simulated
    print(f"Days: {days - 1}")
    # Print the expected value of the asset's price by computing the mean of the last row of price_list and rounds it to two decimal places
    print(f"Expected Value: ${round(pd.DataFrame(price_list).iloc[-1].mean(), 2)}")
    # Computes the difference between the average price at the end of the simulation and the initial price, divide it by the average price 
    # at the end, and expresses the result as a percentage rounded to two decimal places
    print(f"Return: {round(100*(pd.DataFrame(price_list).iloc[-1].mean() - price_list[0,1]) / pd.DataFrame(price_list).iloc[-1].mean(), 2)}%")
    # Return the probabilty the assets return is >= 0
    print(f"Probability of Breakeven: {probs_find(pd.DataFrame(price_list), 0, on = 'return')}")
   
    return pd.DataFrame(price_list)

### Function to Loop through the stock list and show output
def monte_carlo(tickers, days_forecast, iterations, start_date = '2010-1-1', return_type = 'log', plotten=False):
    # Import Data
    data = import_stock_data(tickers, start_date)
    # Compute Beta
    inform = beta_sharpe(data, mark_ticker='^GSPC', start = start_date, riskfree = 0.025)
    simulatedDF = []

    for t in range(len(tickers)):
        # Assign the price_list dataframe to y
        y = simulate_mc(data.iloc[:,t], (days_forecast+1), iterations, return_type)

        if plotten == True:
            forplot = y.iloc[:,0:10]
            forplot.plot(figsize=(15,4))
        # Prints the computed beta, Sharpe ratio, and CAPM return for each stock
        print(f"Beta: {round(inform.iloc[t, inform.columns.get_loc('Beta')], 2)}")
        print(f"Sharpe: {round(inform.iloc[t, inform.columns.get_loc('Sharpe')], 2)}") 
        print(f"CAPM Return: {round(100*inform.iloc[t,inform.columns.get_loc('CAPM')], 2)}%")
        y['ticker'] = tickers[t]
        cols = y.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        y = y[cols]
        simulatedDF.append(y)
    
    simulatedDF = pd.concat(simulatedDF)
    
    return simulatedDF

