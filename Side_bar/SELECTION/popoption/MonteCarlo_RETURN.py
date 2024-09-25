import numpy as np
import pandas as pd
from numba import njit
import math
import time

# Assumptions:
# The stock price volatility is equal to the implied volatility and remains constant.
# Geometric Brownian Motion is used to model the stock price.
# Risk-free interest rates remain constant.
# The Black-Scholes Model is used to price options contracts.
# Dividend yield is not considered.
# Commissions are not considered.
# Assignment risks are not considered.
# Earnings date and stock splits are not considered.


def monteCarlo_exp_return(underlying, rate, sigma, days_to_expiration, closing_days_array, trials, initial_credit,
                   min_profit, strikes, bsm_func, yahoo_stock, instr_type):

    log_returns = np.log(1 + yahoo_stock['Close'].pct_change())
    # Define the variables
    u = log_returns.mean()
    var = log_returns.var()

    # Calculate drift and standard deviation
    drift = u - (0.5 * var)

    # Compute the logarithmic returns using the Closing price
    log_returns = np.log(yahoo_stock['Close'] / yahoo_stock['Close'].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    volatility = log_returns.rolling(window=252).std() * np.sqrt(252)
    volatility = volatility[-1]

    dt = 1 / 365  # 365 calendar days in a year

    length = len(closing_days_array)
    max_closing_days = max(closing_days_array)

    sigma = sigma / 100
    rate = rate / 100

    indices = [0] * length

    profit_last_day_short = []

    stock_price_list = []
    r_list = []
    trial_list = []

    start = time.time()
    # for c in range(trials):
    #
    #     epsilon_cum = 0
    #     t_cum = 0
    #
    #     for i in range(length):
    #         indices[i] = 0

        # +1 added to account for first day. sim_prices[0,...] = underlying price.
        # for r in range(max_closing_days + 1):
        #
        #     # Brownian Motion
        #     W = (dt ** (1 / 2)) * epsilon_cum
        #
        #     # Geometric Brownian Motion
        #     # signal = (rate - 0.5 * (sigma ** 2)) * t_cum
        #     signal = drift - 0.5 * (volatility ** 2) * t_cum
        #     # noise = sigma * W
        #     noise = volatility * W
        #     y = noise + signal
        #     stock_price = underlying * np.exp(y)  # Stock price on current day
        #     epsilon = np.random.randn()
        #     epsilon_cum += epsilon
        #     t_cum += dt
        #
        #     # Prevents crashes
        #     if stock_price <= 0:
        #         stock_price = 0.001
        #
        #     stock_price_list.append(stock_price)
        #     r_list.append(r)
        #     trial_list.append(c)

            # debit = bsm_func(stock_price, strikes, rate, dt * (days_to_expiration - r), sigma, instr_type)
            #
            # profit = initial_credit - debit  # Profit if we were to close on current day
            #
            # if r == max_closing_days - 1:
            #     # print('days_to_expiration_short', days_to_expiration_short)
            #     # print('r', r)
            #     profit_last_day_short.append(profit)



    stock_std = yahoo_stock['Close'][-245:].std()
    stock_price_list = np.arange(underlying-stock_std, underlying+stock_std, stock_std/100)
    print('stock_std', stock_std)
    print('stock_price_list', stock_price_list)


    bsm_df = pd.DataFrame()
    bsm_df['stock_price'] = stock_price_list
    # bsm_df['trial'] = trial_list
    bsm_df['strike'] = strikes[0]
    # bsm_df['r'] = r_list
    bsm_df['dte'] = dt * days_to_expiration  #(max_closing_days/2)  dt * 1
    bsm_df['sigma'] = sigma
    bsm_df['instr_type'] = instr_type
    bsm_df['rate'] = rate

    print('bsm_df')
    print(bsm_df)

    # bsm_df = bsm_df[bsm_df['r'] == max_closing_days-1]
    # print(bsm_df)

    end = time.time() - start
    print('return_stock_loop:', end)

    start = time.time()
    debit = bsm_func(bsm_df)
    profit = initial_credit - debit

    bsm_df['profit'] = profit

    expected_profit = bsm_df['profit'].mean()
    end = time.time() - start

    print('return_black76_loop:', end)

    return expected_profit
