import numpy as np
import pandas as pd
from numba import jit
import math

# Assumptions:
# The stock price volatility is equal to the implied volatility and remains constant.
# Geometric Brownian Motion is used to model the stock price.
# Risk-free interest rates remain constant.
# The Black-Scholes Model is used to price options contracts.
# Dividend yield is not considered.
# Commissions are not considered.
# Assignment risks are not considered.
# Earnings date and stock splits are not considered.


def monteCarlo_return(underlying, rate, sigma_short, sigma_long, days_to_expiration_short, days_to_expiration_long, closing_days_array, trials, initial_credit,
                   min_profit, strikes, bsm_func, yahoo_stock):

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

    sigma_short, sigma_long = sigma_short / 100, sigma_long / 100
    rate = rate / 100

    counter1 = [0] * length
    dtc = [0] * length
    dtc_history = np.zeros((length, trials))

    indices = [0] * length

    profit_last_day_short = []

    P_short_cals_list = []
    P_long_cals_list = []
    stock_price_list = []
    pop_list = []

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
            signal = drift - 0.5 * (volatility ** 2) * t_cum
            # noise = sigma * W

            noise = volatility * W
            y = noise + signal
            stock_price = underlying * np.exp(y)  # Stock price on current day

            epsilon = np.random.randn()
            epsilon_cum += epsilon
            t_cum += dt

            # Prevents crashes
            if stock_price <= 0:
                stock_price = 0.001

            time_fraction_short = dt * (days_to_expiration_short - r)
            time_fraction_long = dt * (days_to_expiration_long - days_to_expiration_short)

            # time_fraction_short = dt * (days_to_expiration_short)
            # time_fraction_long = dt * (days_to_expiration_long)


            debit, P_short_cals, P_long_cals = bsm_func(stock_price, strikes, rate, time_fraction_short, time_fraction_long, sigma_short, sigma_long)

            profit = debit + initial_credit  # Profit if we were to close on current day

            if r == max_closing_days-1:
                # print('days_to_expiration_short', days_to_expiration_short)
                # print('r', r)
                profit_last_day_short.append(profit)
                P_short_cals_list.append(P_short_cals)
                P_long_cals_list.append(P_long_cals)
                stock_price_list.append(stock_price)

                sum = 0

                if min_profit[i] <= profit:  # If target profit hit, combo has been evaluated
                    counter1[i] += 1
                    dtc[i] += r
                    dtc_history[i, c] = r

                    indices[i] = 1
                    sum += 1
                elif r >= closing_days_array[i]:  # If closing days passed, combo has been evaluated
                    indices[i] = 1
                    sum += 1

                if sum == length:  # If all combos evaluated, break and start new trial
                    break
    #
    pop_counter1 = [c / trials * 100 for c in counter1]
    pop_counter1 = [round(x, 2) for x in pop_counter1]

    probabilities = 1 / trials


    main_df = pd.DataFrame({
        'Profit': profit_last_day_short,
        'Proba': probabilities,
        'stock_price': stock_price_list,
    })

    main_df = main_df.sort_values('Profit').reset_index(drop=True)
    main_df['Correct_proba'] = np.nan

    for num, main_df_row in main_df.iterrows():
        current_profit = main_df_row['Profit']
        local_df = main_df[main_df['Profit'] >= current_profit]
        # local_df = local_df[local_df['Profit'] <= min_profit[0]]
        current_proba = local_df['Proba'].sum()
        main_df['Correct_proba'].iloc[num] = current_proba

    print('main_df')
    print(main_df)
    print(main_df[main_df['Profit'] >= min_profit[0]])

    print('profit_last_day_short', profit_last_day_short)
    print('probabilities', (main_df[main_df['Profit'] >= min_profit[0]] * 1/trials).sum())
    print('pop_counter1', pop_counter1[0])
    print('aaaaaa', pop_counter1[0]/100 * np.sum(profit_last_day_short))
    print(sigma_short)
    print(rate)
    print(strikes)
    # print(profit_last_day_short)

    # expected_profit = (main_df[main_df['Profit'] >= min_profit[0]] * 1/trials)['Profit'].sum()
    # print('expected_profit', expected_profit)

    expected_profit = (main_df['Correct_proba'] * main_df['Profit']).mean()

    return expected_profit
