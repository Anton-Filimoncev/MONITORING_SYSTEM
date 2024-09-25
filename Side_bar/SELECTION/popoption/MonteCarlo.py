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

def monteCarlo(underlying, rate, sigma, days_to_expiration, closing_days_array, trials, initial_credit,
                   min_profit, strikes, bsm_func, yahoo_stock, instr_type):
    profit_list = []
    price_list = []
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

    counter1 = [0] * length
    dtc = [0] * length
    dtc_history = np.zeros((length, trials))

    indices = [0] * length

    stock_price_list = []
    r_list = []
    trial_list = []

    start = time.time()

    for c in range(trials):

        epsilon_cum = 0
        t_cum = 0

        for i in range(length):
            indices[i] = 0

        # +1 added to account for first day. sim_prices[0,...] = underlying price.
        for r in range(max_closing_days + 1):

            # Brownian Motion
            W = (dt ** (1 / 2)) * epsilon_cum

            # Geometric Brownian Motion
            # signal = (rate - 0.5 * (sigma ** 2)) * t_cum
            signal = drift - 0.5 * (sigma ** 2) * t_cum
            # noise = sigma * W
            noise = volatility * W
            y = noise + signal
            stock_price = underlying * np.exp(y)  # Stock price on current day
            # print(stock_price)
            epsilon = np.random.randn()
            epsilon_cum += epsilon
            t_cum += dt

            # Prevents crashes
            if stock_price <= 0:
                stock_price = 0.001

            stock_price_list.append(stock_price)
            r_list.append(r)
            trial_list.append(c)

            sum = 0

            for i in range(length):
                if indices[i] == 1:  # Checks if combo has been evaluated
                    sum += 1
                    continue
                # else:
                #     if min_profit[i] <= profit:  # If target profit hit, combo has been evaluated
                #         counter1[i] += 1
                #         dtc[i] += r
                #         dtc_history[i, c] = r
                #
                #         indices[i] = 1
                #         sum += 1
                #     elif r >= closing_days_array[i]:  # If closing days passed, combo has been evaluated
                #         indices[i] = 1
                #         sum += 1

            if sum == length:  # If all combos evaluated, break and start new trial
                break

    bsm_df = pd.DataFrame()
    bsm_df['stock_price'] = stock_price_list
    bsm_df['trial'] = trial_list
    bsm_df['strike'] = strikes[0]
    bsm_df['r'] = r_list
    bsm_df['dte'] = dt * (days_to_expiration - bsm_df['r'])
    bsm_df['sigma'] = sigma
    bsm_df['instr_type'] = instr_type
    bsm_df['rate'] = rate

    end = time.time() - start
    print('stock_loop:', end)

    start = time.time()
    debit = bsm_func(bsm_df)
    end = time.time() - start
    print('black76_loop:', end)

    profit = initial_credit - debit  # Profit if we were to close on current day

    bsm_df['profit'] = profit

    counter1 = [0]


    for trialus in bsm_df['trial'].unique():
        one_trial_df = bsm_df[bsm_df['trial'] == trialus]
        if len(one_trial_df[one_trial_df['profit'] >= min_profit[0]]) > 0:
            counter1[0] += 1



    pop_counter1 = [c / trials * 100 for c in counter1]
    pop_counter1 = [round(x, 2) for x in pop_counter1]


    # Taken from Eq. 2.20 from Monte Carlo theory, methods and examples, by Art B. Owen
    pop_counter1_err = [2.58 * (x * (100 - x) / trials) ** (1 / 2) for x in pop_counter1]
    pop_counter1_err = [round(x, 2) for x in pop_counter1_err]

    # avg_dtc = []  # Average days to close
    # avg_dtc_error = []
    #
    # # Taken from Eq. 2.34 from Monte Carlo theory, methods and examples, by Art B. Owen, 2013
    # for index in range(length):
    #     if counter1[index] > 0:
    #         avg_dtc.append(dtc[index] / counter1[index])
    #
    #         n_a = counter1[index]
    #         mu_hat_a = dtc[index] / n_a
    #         summation = 0
    #
    #         for value in dtc_history[index, :]:
    #             if value == 0:  # if 0 then it means that min_profit wasn't hit
    #                 continue
    #
    #             summation = summation + (value - mu_hat_a) ** 2
    #
    #         s_a_squared = (1 / n_a) * summation  # changed from n_a - 1 for simplicity
    #         std_dev = ((n_a - 1) * s_a_squared) ** (1 / 2) / n_a
    #         avg_dtc_error.append(2.58 * std_dev)
    #
    #     else:
    #         avg_dtc.append(0)
    #         avg_dtc_error.append(0)
    #
    # avg_dtc = [round(x, 2) for x in avg_dtc]
    #
    # avg_dtc_error = [round(x, 2) for x in avg_dtc_error]

    cvar = bsm_df.sort_values('profit')[:int(len(bsm_df)*0.05)]['profit'].mean()
    return pop_counter1, pop_counter1_err, cvar
