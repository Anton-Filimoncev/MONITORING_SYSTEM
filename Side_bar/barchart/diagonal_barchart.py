import pandas as pd
import numpy as np
import scipy.stats as stats
import time
import datetime
import yfinance as yf
import math
import requests
import credential
import mibian
from ..popoption.ShortPut import shortPut
from scipy.stats import norm
from ..popoption.poptions import *
from ..popoption.ShortCall import shortCall
from ..popoption.ShortStrangle import shortStrangle
from ..popoption.PutCalendar import putCalendar


def get_black_scholes_greeks(type_option, stock_price, strike, risk_rate, vol_opt, days_to_exp):
    if type_option == 'p':
        c = mibian.BS([stock_price, strike, risk_rate, days_to_exp], volatility=vol_opt * 100)
        return c.putDelta, c.gamma, c.putTheta, c.vega

    if type_option == 'c':
        c = mibian.BS([stock_price, strike, risk_rate, days_to_exp], volatility=vol_opt * 100)
        return c.callDelta, c.gamma, c.callTheta, c.vega


def get_tick_from_csv_name(csv_position_df):
    ticker_name = csv_position_df.split("\\")[-1].split("_")[0]
    return ticker_name


def _gbs(option_type, fs, x, t, r, b, v):
    # -----------
    # Create preliminary calculations
    t__sqrt = math.sqrt(t)
    d1 = (math.log(fs / x) + (b + (v ** 2) / 2) * t) / (v * t__sqrt)
    d2 = d1 - v * t__sqrt

    print('AAAAAAAAAAoption_type', option_type)

    if option_type == "c":
        # it's a call
        value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
        delta = math.exp((b - r) * t) * norm.cdf(d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) - (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(d1) - r * x * math.exp(-r * t) * norm.cdf(d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = x * t * math.exp(-r * t) * norm.cdf(d2)
    else:
        # it's a put
        value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))
        delta = -math.exp((b - r) * t) * norm.cdf(-d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = (-(fs * v * math.exp((b - r) * t) * norm.cdf(d1)) / (2 * t__sqrt) + (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(-d1) + r * x * math.exp(-r * t) * norm.cdf(-d2))

        # theta =(-fs * v * np.exp(-r * t) * norm.pdf(d1) / (2 * np.sqrt(t))
        #      + r * x * np.exp(-r * t) * norm.cdf(-d2))/100

        vega = (math.exp((b - r) * t) * fs * t__sqrt * norm.cdf(d1))
        rho = -x * t * math.exp(-r * t) * norm.cdf(-d2)

    return value, delta, gamma, theta, vega, rho


def strike_price(strike):
    if strike < 1:
        strike = str(strike).replace('.', '')
        out_strike = '00000' + strike + '00'
    elif strike < 10:
        strike = str(strike).replace('.', '')
        out_strike = '0000' + strike + '00'
    elif strike < 100:
        strike = str(strike).replace('.', '')
        out_strike = '000' + strike + '00'
    elif strike < 1000:
        strike = str(strike).replace('.', '')
        out_strike = '00' + strike + '00'
    elif strike < 10000:
        strike = str(strike).replace('.', '')
        out_strike = '0' + strike + '00'
    return out_strike


def get_contract_calendar(new_position_df, ticker):
    str_exp_date_long = new_position_df['Exp_date_long'].values[0].strftime('%Y-%m-%d').split('-')
    contract_price = strike_price(new_position_df['Strike_long'].values[0])
    contract_long = ticker + str_exp_date_long[0][-2:] + str_exp_date_long[1] + str_exp_date_long[
        2] + 'P' + contract_price

    str_exp_date_short = new_position_df['Exp_date_short'].values[0].strftime('%Y-%m-%d').split('-')
    contract_price = strike_price(new_position_df['Strike_short'].values[0])
    contract_short = ticker + str_exp_date_short[0][-2:] + str_exp_date_short[1] + str_exp_date_short[
        2] + 'P' + contract_price

    return contract_long, contract_short


def get_contract(new_position_df, ticker):
    str_exp_date = new_position_df['Exp_date'].values[0].strftime('%Y-%m-%d').split('-')
    contract_price = strike_price(new_position_df['Strike'].values[0])
    contract = ticker + str_exp_date[0][-2:] + str_exp_date[1] + str_exp_date[2] + 'P' + contract_price
    return contract


def calculate_option_price(type_option, stock_price, strike, risk_rate, vol_opt, days_to_exp):
    if type_option == 'P':
        c = mibian.BS([stock_price, strike, risk_rate, days_to_exp], volatility=vol_opt * 100)
        return c.putPrice

    if type_option == 'C':
        c = mibian.BS([stock_price, strike, risk_rate, days_to_exp], volatility=vol_opt * 100)
        return c.callPrice


def get_proba_50_calendar(current_price, yahoo_data, put_long_strike, put_long_price, put_short_strike, put_short_price,
                          sigma_short, sigma_long, days_to_expiration_short, days_to_expiration_long, risk_rate):
    closing_days_array = [days_to_expiration_short]
    percentage_array = [30]
    trials = 3000

    proba_50 = putCalendar(current_price, sigma_short, sigma_long, risk_rate, trials, days_to_expiration_short,
                           days_to_expiration_long, closing_days_array, percentage_array, put_long_strike,
                           put_long_price, put_short_strike, put_short_price, yahoo_data)

    return proba_50


def get_proba_50_put(current_price, yahoo_data, short_strike, short_price, sigma, days_to_expiration, risk_rate):
    closing_days_array = [days_to_expiration]
    percentage_array = [50]
    trials = 3000

    proba_50 = shortPut(
        current_price,
        sigma,
        risk_rate,
        trials,
        days_to_expiration,
        closing_days_array,
        percentage_array,
        short_strike,
        short_price,
        yahoo_data,
    )

    return proba_50


def get_abs_return(price_array, type_option, days_to_exp, days_to_exp_short, history_vol, current_price, strike, prime,
                   vol_opt):
    put_price_list = []
    call_price_list = []
    proba_list = []
    price_gen_list = []

    for stock_price_num in range(len(price_array)):
        try:
            P_below = stats.norm.cdf(
                (np.log(price_array[stock_price_num] / current_price) / (
                        history_vol * math.sqrt(days_to_exp_short / 365))))
            P_current = stats.norm.cdf(
                (np.log(price_array[stock_price_num + 1] / current_price) / (
                        history_vol * math.sqrt(days_to_exp_short / 365))))
            proba_list.append(P_current - P_below)
            if type_option == 'Short':
                c = mibian.BS([price_array[stock_price_num + 1], strike, 4, 1], volatility=vol_opt * 100)
            if type_option == 'Long':
                c = mibian.BS([price_array[stock_price_num + 1], strike, 4, days_to_exp], volatility=vol_opt * 100)

            put_price_list.append(c.putPrice)
            call_price_list.append(c.callPrice)
            price_gen_list.append(price_array[stock_price_num + 1])
        except:
            pass

    put_df = pd.DataFrame({
        'gen_price': price_gen_list,
        'put_price': put_price_list,
        'call_price': call_price_list,
        'proba': proba_list,
    })

    put_df['return'] = (put_df['put_price'] - prime)

    if type_option == 'Short':
        return ((prime - put_df['put_price']) * put_df['proba']).sum()

    if type_option == 'Long':
        return ((put_df['put_price'] - prime) * put_df['proba']).sum()


def expected_return_calc(vol_put_short, vol_put_long, current_price, history_vol, days_to_exp_short, days_to_exp_long,
                         strike_put_long, strike_put_short, prime_put_long, prime_put_short):
    # print('expected_return CALCULATION ...')

    price_array = np.arange(current_price - current_price / 2, current_price + current_price, 0.2)

    short_finish = get_abs_return(price_array, 'Short', days_to_exp_short, days_to_exp_short, history_vol,
                                  current_price, strike_put_short,
                                  prime_put_short,
                                  vol_put_short)

    long_finish = get_abs_return(price_array, 'Long', days_to_exp_long, days_to_exp_short, history_vol, current_price,
                                 strike_put_long,
                                 prime_put_long,
                                 vol_put_long)

    expected_return = (short_finish + long_finish) * 100

    return expected_return


def solo_position_calc_new(row, yahoo_stock, sigma, days_to_expiration, days_to_expiration_min, closing_days_array,
                               underlying, strike, prime):
    trials = 3_000
    pos_type = row['side']
    short_long = row['count']
    print('pos_type', pos_type)
    rate = row['rate']
    # closing_days_array = row['closing_days_array']
    percentage_array = row['percentage_array']

    if '-' in row['symbol'] or '=' in row['symbol']:
        instr_type = 'FUT'

    print('underlying', underlying)
    print('sigma', sigma)
    print('rate', rate)
    print('trials', trials)
    print('days_to_expiration', days_to_expiration)
    print('days_to_expiration_min', days_to_expiration_min)
    print('closing_days_array', closing_days_array)
    print('percentage_array', percentage_array)

    print('strike', strike)
    print('prime', prime)
    print('instr_type', instr_type)

    if pos_type == 'Put':
        if short_long > 0:
            response = longPut(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                               [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
        else:
            response = shortPut(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
    if pos_type == 'Call':
        if short_long > 0:
            response = longCall(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
        else:
            response = shortCall(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                 [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)

    if pos_type == 'Stock':
        if short_long > 0:
            response = longStock(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                 [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
        else:
            response = shortStock(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                  [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)

    return response


def emulate_position(input_new_df, path, path_bento, risk_rate):
    new_position_df = pd.DataFrame()
    pos_type = input_new_df['position_type'].values[0]
    ticker = input_new_df['symbol'].values[0]
    ticker_b = input_new_df['symbol_bento'].values[0]
    print('/', ticker, '/')
    print(ticker)
    print(input_new_df['symbol'].values)
    yahoo_price_df = yf.download(ticker)
    print(yahoo_price_df)
    start_price = yahoo_price_df['Close'].iloc[-1]
    KEY = credential.password['marketdata_KEY']
    log_returns = np.log(yahoo_price_df["Close"] / yahoo_price_df["Close"].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    hv = log_returns.rolling(window=200).std() * np.sqrt(365)
    hv = hv.iloc[-1]
    current_price = yahoo_price_df['Close'].iloc[-1]

    if (pos_type == 'F. Covered' or pos_type == 'F. Strangle' or pos_type == 'F. Diagonal' or pos_type == 'F. Ratio 112'
            or pos_type == 'F. Put/Call'):
        min_dte = input_new_df['days_to_exp'].min()

        response_df = pd.DataFrame()

        for num, row in input_new_df.iterrows():

            response = pd.DataFrame(
                solo_position_calc_new(row, yahoo_price_df, row['iv'],
                                       row['days_to_exp'], min_dte, min_dte,
                                       row['underlying'], row['strike'],
                                       row['prime']))
            response[['cvar', 'exp_return']] = (
                        response[['cvar', 'exp_return']] * np.abs(row['count']) * row['multiplier'])

            # ------  GREEKS  ------

            local_side = 'p'
            if row['side'] == 'CALL':
                local_side = 'c'

            if row['side'] == 'STOCK':
                value, delta, gamma, theta, vega, rho = np.nan, 1, np.nan, np.nan, np.nan, np.nan
                delta_bs, gamma_bs, theta_bs, vega_bs = np.nan, np.nan, np.nan, np.nan

            else:

                value, delta, gamma, theta, vega, rho = _gbs(local_side, row['underlying'], row['strike'],
                                                             row['days_to_exp'] / 365, 0.04, 0.04, row['iv'] / 100)

                delta_bs, gamma_bs, theta_bs, vega_bs = get_black_scholes_greeks(local_side, row['underlying'],
                                                                                 row['strike']
                                                                                 , risk_rate, row['iv'] / 100,
                                                                                 row['days_to_exp'])

            response['delta'] = (delta * row['count'])
            response['gamma'] = (gamma * row['count'])
            response['theta'] = (theta * row['count'])
            response['vega'] = (vega * row['count'])
            response['rho'] = (rho * row['count'])

            response['theta_bs'] = (theta_bs * row['count'])
            response['vega_bs'] = (vega_bs * row['count'])

            response[['theta_bs', 'vega_bs']] = response[['theta_bs', 'vega_bs']] * row['multiplier']

            response_df = pd.concat([response_df, response])

        print('response_df')
        print(response_df)
        response_df = response_df.mean()
        response_df = response_df.round(4)

        input_new_df['pop'] = response_df['pop']
        input_new_df['current_expected_return'] = response_df['exp_return']
        input_new_df['cvar'] = response_df['cvar']

        print('input_new_df')
        print(input_new_df[['pop', 'current_expected_return', 'cvar']])

        return input_new_df[['pop', 'current_expected_return', 'cvar']].drop_duplicates()



def calc_position_open_update(input_new_df, yahoo_price_df, risk_rate):
    open_cost = 0
    input_new_df['days_remaining'] = (input_new_df['exp_date'].iloc[0] - datetime.datetime.now().date()).days

    min_dte = input_new_df['days_to_exp'].min()

    response_df = pd.DataFrame()

    for num, row in input_new_df.iterrows():
        open_cost += -row['prime'] * row['count']

        response = pd.DataFrame(
            solo_position_calc_new(row, yahoo_price_df, row['iv'],
                                   row['days_to_exp'], min_dte, min_dte,
                                   row['underlying'], row['strike'],
                                   row['prime']))
        response[['cvar', 'exp_return']] = (response[['cvar', 'exp_return']] * np.abs(row['count']) * row['multiplier'])

        # ------  GREEKS  ------

        local_side = 'p'
        if row['side'] == 'CALL':
            local_side = 'c'

        if row['side'] == 'STOCK':
            value, delta, gamma, theta, vega, rho = np.nan, 1, np.nan, np.nan, np.nan, np.nan
            delta_bs, gamma_bs, theta_bs, vega_bs = np.nan, np.nan, np.nan, np.nan

        else:

            value, delta, gamma, theta, vega, rho = _gbs(local_side, row['underlying'], row['strike'],
                                                         row['days_to_exp'] / 365, 0.04, 0.04, row['iv'] / 100)

            delta_bs, gamma_bs, theta_bs, vega_bs = get_black_scholes_greeks(local_side, row['underlying'],
                                                                             row['strike']
                                                                             , risk_rate, row['iv'] / 100,
                                                                             row['days_to_exp'])

        response['delta'] = (delta * row['count'])
        response['gamma'] = (gamma * row['count'])
        response['theta'] = (theta * row['count'])
        response['vega'] = (vega * row['count'])
        response['rho'] = (rho * row['count'])

        response['theta_bs'] = (theta_bs * row['count'])
        response['vega_bs'] = (vega_bs * row['count'])

        response[['theta_bs', 'vega_bs']] = response[['theta_bs', 'vega_bs']] * row['multiplier']

        response_df = pd.concat([response_df, response])

    print('response_df')
    print(response_df)
    response_df = response_df.mean()
    response_df = response_df.round(4)

    print('response_df mean')
    print(response_df)

    input_new_df['pop'] = response_df['pop']
    input_new_df['current_expected_return'] = response_df['exp_return']
    input_new_df['cvar'] = response_df['cvar']

    input_new_df['delta'] = response_df['delta']
    input_new_df['gamma'] = response_df['gamma']
    input_new_df['theta'] = response_df['theta']
    input_new_df['vega'] = response_df['vega']
    input_new_df['rho'] = response_df['rho']
    input_new_df['theta_bs'] = response_df['theta_bs']
    input_new_df['vega_bs'] = response_df['vega_bs']

    print('input_new_df')
    print(input_new_df)
    print('open_cost')
    print(open_cost)

    input_new_df['open_cost'] = open_cost * input_new_df['multiplier'].iloc[0]
    input_new_df['max_profit'] = input_new_df['open_cost'].abs()

    input_new_df['MP/DTE'] = input_new_df['max_profit'] / input_new_df['days_to_exp']
    input_new_df['profit_target'] = input_new_df['max_profit'] * 0.5 - (input_new_df['commission'].iloc[0])

    input_new_df['days_remaining'] = (input_new_df['exp_date'].iloc[0] - datetime.datetime.now().date()).days
    input_new_df['days_elapsed_TDE'] = (input_new_df['exp_date'].iloc[0] - input_new_df['start_date'].iloc[0]).days - input_new_df['days_remaining']
    input_new_df['%_days_elapsed'] = input_new_df['days_elapsed_TDE'] / input_new_df['days_to_exp']

    input_new_df['prime_now'] = open_cost * input_new_df['multiplier'].iloc[0]
    input_new_df['cost_to_close_market_cost'] = input_new_df['open_cost'] - input_new_df['commission']

    input_new_df['current_PL'] = input_new_df['cost_to_close_market_cost'].iloc[0] - input_new_df['open_cost'].iloc[0]
    input_new_df['current_ROI'] = input_new_df['current_PL'] / input_new_df['margin'].iloc[0]

    input_new_df['PL_TDE'] = input_new_df['current_PL'] / input_new_df['days_elapsed_TDE']

    input_new_df[['%_days_elapsed', 'current_ROI']] = input_new_df[['%_days_elapsed', 'current_ROI']] * 100

    return input_new_df


def create_new_postion(input_new_df, path, path_bento, risk_rate):
    pos_type = input_new_df['position_type'].values[0]
    ticker = input_new_df['symbol'].values[0]
    ticker_b = input_new_df['symbol_bento'].values[0]
    yahoo_price_df = yf.download(ticker)
    start_price = yahoo_price_df['Close'].iloc[-1]
    KEY = credential.password['marketdata_KEY']
    log_returns = np.log(yahoo_price_df["Close"] / yahoo_price_df["Close"].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    hv = log_returns.rolling(window=200).std() * np.sqrt(365)
    hv = hv.iloc[-1]
    current_price = yahoo_price_df['Close'].iloc[-1]

    if pos_type == 'F. Ratio 112':
        input_new_df = calc_position_open_update(input_new_df, yahoo_price_df, risk_rate)

        input_new_df[['%_days_elapsed', 'current_ROI']] = input_new_df[['%_days_elapsed', 'current_ROI']] * 100

        input_new_df = input_new_df.round(4)

        input_new_df.to_csv(f"{path}{ticker}_{input_new_df['start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                            index=False)

    if pos_type == 'F. Put/Call':
        input_new_df['hv_200'] = hv
        input_new_df = calc_position_open_update(input_new_df, yahoo_price_df, risk_rate)

        input_new_df[['%_days_elapsed', 'current_ROI']] = input_new_df[['%_days_elapsed', 'current_ROI']] * 100

        input_new_df['current_RR_ratio'] = (input_new_df['max_profit'] - input_new_df['current_PL']) / (
                input_new_df['cvar'] + input_new_df['current_PL'])
        input_new_df = input_new_df.round(4)

        input_new_df.to_csv(f"{path}{ticker}_{input_new_df['start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                            index=False)
    if pos_type == 'F. Covered':

        input_new_df['hv_200'] = hv
        input_new_df = calc_position_open_update(input_new_df, yahoo_price_df, risk_rate)

        input_new_df[['%_days_elapsed', 'current_ROI']] = input_new_df[['%_days_elapsed', 'current_ROI']] * 100

        input_new_df['current_RR_ratio'] = (input_new_df['max_profit'] - input_new_df['current_PL']) / (
                input_new_df['cvar'] + input_new_df['current_PL'])

        input_new_df['leverage'] = (input_new_df['delta'] * input_new_df['underlying'].iloc[0]) / input_new_df[
            'margin']

        input_new_df = input_new_df.round(4)

        input_new_df.to_csv(f"{path}{ticker}_{input_new_df['start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                            index=False)

    if pos_type == 'F. Diagonal':

        input_new_df['hv_200'] = hv
        input_new_df = calc_position_open_update(input_new_df, yahoo_price_df, risk_rate)


        input_new_df[['%_days_elapsed', 'current_ROI']] = input_new_df[['%_days_elapsed', 'current_ROI']] * 100

        input_new_df = input_new_df.round(4)

        input_new_df.to_csv(f"{path}{ticker}_{input_new_df['start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                            index=False)

    if pos_type == 'F. Call':
        new_position_df['Symbol'] = [ticker]
        new_position_df['Symbol Bento'] = [ticker_b]
        new_position_df['Start_date'] = input_new_df['Start_date_o_p']
        new_position_df['Exp_date'] = input_new_df['Exp_date_o_p']
        new_position_df['Strike'] = input_new_df['Strike_o_p']
        new_position_df['Number_pos'] = input_new_df['Number_pos_o_p'] * -1
        new_position_df['Start_prime'] = input_new_df['Prime_o_p']
        new_position_df['Multiplicator'] = input_new_df['Multiplicator_o_p']
        # new_position_df['Contract'] = get_contract(new_position_df, ticker)
        new_position_df['Start_price'] = input_new_df['Start_price']
        new_position_df['Dividend'] = input_new_df['Dividend']
        new_position_df['Commission'] = input_new_df['Commission_o_p']
        new_position_df['Margin_start'] = input_new_df['Margin_o_p']
        new_position_df['DTE'] = (new_position_df['Exp_date'] - new_position_df['Start_date']).values[0].days
        new_position_df['Open_cost'] = (new_position_df['Number_pos'].iloc[0] * new_position_df['Start_prime'].values[
            0]) * new_position_df['Multiplicator'].values[0]
        new_position_df['Start_delta'] = input_new_df['Delta_o_p']
        new_position_df['HV_200'] = hv
        sigma_df = get_databento.get_bento_data(ticker_b, current_price, new_position_df['DTE'].values[0],
                                                input_new_df['Strike_o_p'].iloc[0], 'C', path_bento)

        ticker, ticker_b, nearest_dte, side, path_bento
        current_IV = sigma_df['sigma'].iloc[0] * 100

        new_position_df['Price_2s_down'] = start_price - (
                2 * start_price * hv * (math.sqrt(new_position_df['DTE'].iloc[0] / 365)))
        new_position_df['Max_profit'] = new_position_df['Open_cost'].abs()
        # max_risk = calculate_option_price('P', new_position_df['Price_2s_down'].values[0], new_position_df['Strike'].values[0], risk_rate, current_IV, new_position_df['DTE'].values[0])
        new_position_df['Max_Risk'] = np.max([new_position_df['Strike'].iloc[0] * 0.1, (
                new_position_df['Price_2s_down'].iloc[0] - (
                new_position_df['Price_2s_down'].iloc[0] - new_position_df['Strike'].iloc[0])) * 0.2]) * \
                                      new_position_df['Open_cost'].abs() * 100
        new_position_df['BE_higher'] = new_position_df['Strike'] + new_position_df['Start_prime']
        new_position_df['RR_RATIO'] = new_position_df['Max_profit'] / new_position_df['Max_Risk']
        new_position_df['MP/DTE'] = new_position_df['Max_profit'] / new_position_df['DTE']
        new_position_df['Profit_Target'] = new_position_df['Max_profit'] * 0.5 - (new_position_df['Commission'])

        response = shortCall(
            start_price, current_IV, risk_rate, 2000, new_position_df['DTE'].values[0],
            [new_position_df['DTE'].values[0]],
            [50], new_position_df['Strike'].values[0], new_position_df['Start_prime'].values[0], yahoo_price_df,
            'FUT')

        print('response', response)
        response['pop'] = float(response['pop'][0]) / new_position_df['Multiplicator'].values[0]
        response['cvar'] = float(response['cvar']) * new_position_df['Multiplicator'].values[0]
        response['exp_return'] = float(response['exp_return']) * new_position_df['Multiplicator'].values[0]

        new_position_df['Expected_Return'] = response['exp_return']
        new_position_df['DTE_Target'] = int(response['avg_dtc'][0])
        new_position_df['POP_Monte_start_50'] = response['pop']
        new_position_df['Plan_ROC '] = new_position_df['Profit_Target'] / new_position_df['Margin_start']
        new_position_df['ROC_DAY_target'] = new_position_df['Plan_ROC '] / new_position_df['DTE_Target']
        #
        current_prime = sigma_df['midprice']
        print('current_prime', current_prime)
        current_delta = sigma_df['delta'].iloc[0]
        current_vega = sigma_df['vega'].iloc[0]
        current_theta = sigma_df['theta'].iloc[0]
        current_gamma = sigma_df['gamma'].iloc[0]

        new_position_df['Delta'] = current_delta
        new_position_df['Vega'] = current_vega
        new_position_df['Theta'] = current_theta
        new_position_df['Gamma'] = current_gamma

        print('pop_50 ', response['pop'], 'expected_profit ', response['exp_return'], 'cvar ', response['cvar'], )

        new_position_df['DAYS_remaining'] = (new_position_df['Exp_date'].iloc[0] - datetime.datetime.now().date()).days
        new_position_df['DAYS_elapsed_TDE'] = new_position_df['DTE'] - new_position_df['DAYS_remaining']
        new_position_df['%_days_elapsed'] = new_position_df['DAYS_elapsed_TDE'] / new_position_df['DTE']

        # response = shortCall(
        #     current_price, current_IV, risk_rate, 2000, new_position_df['DAYS_remaining'].iloc[0],
        #     [new_position_df['DAYS_remaining'].iloc[0]],
        #     [50], new_position_df['Strike'].values[0], new_position_df['Start_prime'].values[0], yahoo_price_df
        # )
        #
        new_position_df['POP_50'] = response['pop'] * 100
        new_position_df['Current_expected_return'] = response['exp_return']
        new_position_df['CVAR'] = response['cvar']

        # response['pop'] = float(response['pop'][0]) / new_position_df['Multiplicator'].values[0]
        # response['cvar'] = float(response['cvar']) * new_position_df['Multiplicator'].values[0]
        # response['exp_return'] = float(response['exp_return']) * new_position_df['Multiplicator'].values[0]

        new_position_df['Prime_now'] = current_prime
        new_position_df['Cost_to_close_Market_cost'] = ((new_position_df['Number_pos'] * current_prime) * \
                                                        new_position_df['Multiplicator'].values[0]) - new_position_df[
                                                           'Commission']
        new_position_df['Current_Margin'] = new_position_df['Margin_start'].values[0]
        new_position_df['Current_PL'] = new_position_df['Cost_to_close_Market_cost'] - new_position_df['Open_cost']
        new_position_df['Current_ROI'] = new_position_df['Current_PL'] / new_position_df['Current_Margin']
        new_position_df['Current_RR_ratio'] = (new_position_df['Max_profit'] - new_position_df['Current_PL']) / (
                new_position_df['CVAR'] + new_position_df['Current_PL'])
        new_position_df['PL_TDE'] = new_position_df['Current_PL'] / new_position_df['DAYS_elapsed_TDE']
        new_position_df['Leverage'] = (current_delta * 100 * current_price) / new_position_df['Current_Margin']
        new_position_df['Margin_2S_Down'] = np.max(
            [0.1 * new_position_df['Strike'], new_position_df['Price_2s_down'] * 0.2 - (
                    new_position_df['Price_2s_down'] - new_position_df['Strike'])]) * new_position_df[
                                                'Number_pos'].abs() * 100

        P_below = stats.norm.cdf(
            (np.log(new_position_df['BE_higher'].iloc[0] / current_price) / (
                    new_position_df['HV_200'].iloc[0] * math.sqrt(new_position_df['DAYS_remaining'].iloc[0] / 365))))

        new_position_df['Current_POP_lognormal'] = P_below

        new_position_df['MC_2S_Down'] = calculate_option_price('P', new_position_df['Price_2s_down'].iloc[0],
                                                               new_position_df['Strike'].iloc[0], risk_rate,
                                                               new_position_df['HV_200'].iloc[0],
                                                               new_position_df['DAYS_remaining']) * new_position_df[
                                            'Number_pos'].abs() * 100

        new_position_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio', ]] = \
            new_position_df[[
                '%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100
        postion_df = new_position_df.round(4)

        new_position_df.to_csv(f"{path}{ticker}_{new_position_df['Start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                               index=False)

    if pos_type == 'F. Strangle':
        input_new_df['hv_200'] = hv
        input_new_df = calc_position_open_update(input_new_df, yahoo_price_df, risk_rate)

        input_new_df['leverage'] = (input_new_df['delta'] * input_new_df['underlying'].iloc[0]) / input_new_df[
            'margin']

        input_new_df = input_new_df.round(4)

        input_new_df['current_RR_ratio'] = (input_new_df['max_profit'] - input_new_df['current_PL']) / (
                input_new_df['cvar'] + input_new_df['current_PL'])

        input_new_df['be_lower'] = input_new_df[input_new_df['side'] == 'PUT']['strike'] - (input_new_df['open_cost']/input_new_df['multiplier'].iloc[0])

        P_below = stats.norm.cdf(
            (np.log(input_new_df['be_lower'].iloc[0] / current_price) / (
                    input_new_df['hv_200'].iloc[0] * math.sqrt(input_new_df['days_remaining'].iloc[0] / 365))))

        input_new_df['current_pop_lognormal'] = 1 - P_below

        input_new_df[['%_days_elapsed', 'current_pop_lognormal', 'current_ROI', 'Current_RR_ratio', ]] = \
            input_new_df[[
                '%_days_elapsed', 'current_pop_lognormal', 'current_ROI', 'current_RR_ratio']] * 100

        input_new_df.to_csv(f"{path}{ticker}_{input_new_df['start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                            index=False)

    if pos_type == 'Put Sell':
        new_position_df['Symbol'] = [ticker]
        new_position_df['Start_date'] = input_new_df['Start_date_o_p']
        new_position_df['Exp_date'] = input_new_df['Exp_date_o_p']
        new_position_df['Strike'] = input_new_df['Strike_o_p']
        new_position_df['Number_pos'] = input_new_df['Number_pos_o_p'] * -1
        new_position_df['Start_prime'] = input_new_df['Prime_o_p']
        new_position_df['Contract'] = get_contract(new_position_df, ticker)
        new_position_df['Start_price'] = input_new_df['Start_price']
        new_position_df['Dividend'] = input_new_df['Dividend']
        new_position_df['Commission'] = input_new_df['Commission_o_p']
        new_position_df['Margin_start'] = input_new_df['Margin_o_p']
        new_position_df['DTE'] = (new_position_df['Exp_date'] - new_position_df['Start_date']).values[0].days
        new_position_df['Open_cost'] = (new_position_df['Number_pos'].iloc[0] * new_position_df['Start_prime']) * 100
        new_position_df['Start_delta'] = input_new_df['Delta_o_p']
        new_position_df['HV_200'] = hv
        new_position_df['Price_2s_down'] = start_price - (
                2 * start_price * hv * (math.sqrt(new_position_df['DTE'].iloc[0] / 365)))
        new_position_df['Max_profit'] = new_position_df['Open_cost'].abs()
        new_position_df['Max_Risk'] = np.max([new_position_df['Strike'].iloc[0] * 0.1, (
                new_position_df['Price_2s_down'].iloc[0] - (
                new_position_df['Price_2s_down'].iloc[0] - new_position_df['Strike'].iloc[0])) * 0.2]) * \
                                      new_position_df['Open_cost'].abs() * 100
        new_position_df['BE_lower'] = new_position_df['Strike'] - new_position_df['Start_prime']
        new_position_df['RR_RATIO'] = new_position_df['Max_profit'] / new_position_df['Max_Risk']
        new_position_df['MP/DTE'] = new_position_df['Max_profit'] / new_position_df['DTE']
        new_position_df['Profit_Target'] = new_position_df['Max_profit'] * 0.5 - (new_position_df['Commission'])

        quotes_df = market_data(new_position_df['Contract'].iloc[0], KEY)

        pop_50, expected_profit, avg_dtc = get_proba_50_put(start_price, yahoo_price_df,
                                                            new_position_df['Strike'].iloc[0],
                                                            new_position_df['Start_prime'].iloc[0],
                                                            quotes_df['iv'].iloc[0],
                                                            new_position_df['DTE'].iloc[0], risk_rate)

        new_position_df['Expected_Return'] = expected_profit
        new_position_df['DTE_Target'] = int(avg_dtc[0])
        new_position_df['POP_Monte_start_50'] = pop_50
        new_position_df['Plan_ROC '] = new_position_df['Profit_Target'] / new_position_df['Margin_start']
        new_position_df['ROC_DAY_target'] = new_position_df['Plan_ROC '] / new_position_df['DTE_Target']
        #
        new_position_df.to_csv(f"{path}{ticker}_{new_position_df['Start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                               index=False)

    if pos_type == 'ITM Calendar':
        new_position_df['Symbol'] = [ticker]
        new_position_df['Start_date'] = input_new_df['Start_date_o_p']
        new_position_df['Exp_date_long'] = input_new_df['Exp_date_long_o_p']
        new_position_df['Exp_date_short'] = input_new_df['Exp_date_short_o_p']
        new_position_df['Strike_long'] = input_new_df['Strike_long_o_p']
        new_position_df['Strike_short'] = input_new_df['Strike_short_o_p']
        new_position_df['Number_pos'] = input_new_df['Number_pos_o_p']
        new_position_df['Start_prime_long'] = input_new_df['Prime_long_o_p']
        new_position_df['Start_prime_short'] = input_new_df['Prime_short_o_p']
        new_position_df['Start_prime'] = input_new_df['Prime_o_p']
        new_position_df['Contract_long'], new_position_df['Contract_short'] = get_contract_calendar(new_position_df,
                                                                                                    ticker)
        new_position_df['Start_price'] = input_new_df['Start_price']
        new_position_df['Dividend'] = input_new_df['Dividend']
        new_position_df['Commission'] = input_new_df['Commission_o_p']
        new_position_df['Margin_start'] = input_new_df['Margin_o_p']
        new_position_df['DTE'] = (new_position_df['Exp_date_short'] - new_position_df['Start_date']).values[0].days
        new_position_df['Open_cost'] = (new_position_df['Number_pos'].iloc[0] * new_position_df['Start_prime']) * 100
        new_position_df['Start_delta'] = input_new_df['Delta_o_p']
        new_position_df['HV_200'] = hv
        new_position_df['Price_2s_down'] = start_price - (
                2 * start_price * hv * (math.sqrt(new_position_df['DTE'].iloc[0] / 365)))
        new_position_df['Max_profit'] = new_position_df['Open_cost'].abs()
        new_position_df['Max_Risk'] = input_new_df['Margin_o_p']
        new_position_df['BE_lower'] = new_position_df['Start_prime']
        new_position_df['RR_RATIO'] = new_position_df['Max_profit'] / new_position_df['Max_Risk']
        new_position_df['MP/DTE'] = new_position_df['Max_profit'] / new_position_df['DTE']
        new_position_df['Profit_Target'] = new_position_df['Max_profit'] * 0.5 - (new_position_df['Commission'])
        print(new_position_df['Contract_long'].iloc[0])
        print(new_position_df['Contract_short'].iloc[0])
        quotes_df_long = market_data(new_position_df['Contract_long'].iloc[0], KEY)
        quotes_df_short = market_data(new_position_df['Contract_short'].iloc[0], KEY)

        # pop_50, expected_profit, avg_dtc = get_proba_50_put(start_price, yahoo_price_df,
        #                                                     new_position_df['Strike'].iloc[0],
        #                                                     new_position_df['Start_prime'].iloc[0],
        #                                                     quotes_df['iv'].iloc[0],
        #                                                     new_position_df['DTE'].iloc[0], risk_rate)
        #
        # new_position_df['Expected_Return'] = expected_profit
        # new_position_df['DTE_Target'] = int(avg_dtc[0])
        # new_position_df['POP_Monte_start_50'] = pop_50
        # new_position_df['Plan_ROC '] = new_position_df['Profit_Target'] / new_position_df['Margin_start']
        # new_position_df['ROC_DAY_target'] = new_position_df['Plan_ROC '] / new_position_df['DTE_Target']
        #
        new_position_df.to_csv(f"{path}{ticker}_{new_position_df['Start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                               index=False)

    return


def market_data(contract, KEY):
    url = f"https://api.marketdata.app/v1/options/quotes/{contract}/?&token={KEY}"
    response_exp = requests.request("GET", url, timeout=30)
    chains_df = pd.DataFrame(response_exp.json())
    return chains_df




def update_postion_cover(csv_position_df, pos_type, risk_rate, path_bento, input_update_df):
    postion_df = pd.read_csv(csv_position_df)
    # postion_df = pd.concat([postion_df, input_update_df], axis=1)
    postion_df = pd.concat([input_update_df, postion_df], axis=1)
    postion_df = postion_df.loc[:, ~postion_df.columns.duplicated()]
    ticker = postion_df['symbol'].iloc[0]
    ticker_b = postion_df['symbol_bento'].iloc[0]
    print('tickerrrrrr', ticker)
    yahoo_price_df = yf.download(ticker)

    log_returns = np.log(yahoo_price_df["Close"] / yahoo_price_df["Close"].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    hv = log_returns.rolling(window=200).std() * np.sqrt(365)
    hv = hv.iloc[-1]

    postion_df['cur_date'] = datetime.datetime.now().date()
    postion_df[['exp_date', 'cur_date', 'start_date']] = postion_df[['exp_date', 'cur_date', 'start_date']].apply(pd.to_datetime)
    postion_df['days_to_exp'] = (postion_df['exp_date'] - postion_df['cur_date']).dt.days
    print('88888')
    postion_df['days_remaining'] = (postion_df['exp_date'].iloc[0] - datetime.datetime.now()).days
    postion_df['days_elapsed_TDE'] = (postion_df['exp_date'].iloc[0] - postion_df['start_date'].iloc[0]).days - postion_df['days_remaining']
    postion_df['%_days_elapsed'] = postion_df['days_elapsed_TDE'] / postion_df['days_to_exp']

    response_df = pd.DataFrame()
    min_dte = postion_df['days_to_exp'].min()

    current_prime = 0

    for num, row in postion_df.iterrows():
        current_prime += row['prime_current'] * row['count']
        response = pd.DataFrame(
            solo_position_calc_new(row, yahoo_price_df, row['iv_current'],
                                   row['days_to_exp'], min_dte, min_dte,
                                   row['underlying_current'], row['strike'],
                                   row['prime_current']))
        response[['cvar', 'exp_return']] = (response[['cvar', 'exp_return']] * np.abs(row['count']) * row['multiplier'])

        # ------  GREEKS  ------

        local_side = 'p'
        if row['side'] == 'CALL':
            local_side = 'c'

        if row['side'] == 'STOCK':
            value, delta, gamma, theta, vega, rho = np.nan, 1, np.nan, np.nan, np.nan, np.nan
            delta_bs, gamma_bs, theta_bs, vega_bs = np.nan, np.nan, np.nan, np.nan

        else:

            value, delta, gamma, theta, vega, rho = _gbs(local_side, row['underlying_current'], row['strike'],
                                                         row['days_to_exp'] / 365, 0.04, 0.04, row['iv_current'] / 100)

            delta_bs, gamma_bs, theta_bs, vega_bs = get_black_scholes_greeks(local_side, row['underlying_current'],
                                                                             row['strike']
                                                                             , risk_rate, row['iv_current'] / 100,
                                                                             row['days_to_exp'])

        response['delta'] = (delta * row['count'])
        response['gamma'] = (gamma * row['count'])
        response['theta'] = (theta * row['count'])
        response['vega'] = (vega * row['count'])
        response['rho'] = (rho * row['count'])

        response['theta_bs'] = (theta_bs * row['count'])
        response['vega_bs'] = (vega_bs * row['count'])

        response[['theta_bs', 'vega_bs']] = response[['theta_bs', 'vega_bs']] * row['multiplier']

        response_df = pd.concat([response_df, response])

    response_df = response_df.mean()
    response_df = response_df.round(4)

    postion_df['pop'] = response_df['pop']
    postion_df['current_expected_return'] = response_df['exp_return']
    postion_df['cvar'] = response_df['cvar']

    postion_df['delta'] = response_df['delta']
    postion_df['gamma'] = response_df['gamma']
    postion_df['theta'] = response_df['theta']
    postion_df['vega'] = response_df['vega']
    postion_df['rho'] = response_df['rho']
    postion_df['theta_bs'] = response_df['theta_bs']
    postion_df['vega_bs'] = response_df['vega_bs']

    print('postion_df')
    print(postion_df)

    postion_df['cost_to_close_market_cost'] = (current_prime * postion_df['multiplier'].iloc[0]) - postion_df['commission']
    postion_df['current_PL'] = postion_df['cost_to_close_market_cost'] + postion_df['open_cost']
    postion_df['current_ROI'] = postion_df['current_PL'] / postion_df['margin']
    postion_df['current_RR_ratio'] = (postion_df['max_profit'] - postion_df['current_PL']) / (
            postion_df['cvar'] + postion_df['current_PL'])
    postion_df['PL_TDE'] = postion_df['current_PL'] / postion_df['days_elapsed_TDE']
    postion_df['leverage'] = (response_df['delta'] * postion_df['underlying_current'].iloc[0]) / postion_df['margin']
    # postion_df['Margin_2S_Down'] = np.max([0.1 * postion_df['Strike'], postion_df['Price_2s_down'] * 0.2 - (postion_df['Price_2s_down'] - postion_df['Strike'])]) * postion_df['Number_pos'].abs() * 100

    postion_df[['%_days_elapsed', 'current_ROI', 'current_RR_ratio', ]] = postion_df[['%_days_elapsed', 'current_ROI', 'current_RR_ratio']] * 100
    postion_df = postion_df.round(4)

    postion_df.to_csv(csv_position_df, index=False)
    print('postion_df DONE!!!!')
    #
    current_position = postion_df[['days_remaining', 'days_elapsed_TDE', '%_days_elapsed',
                                   'cost_to_close_market_cost', 'margin', 'current_PL',
                                   'current_expected_return',
                                   'pop', 'current_ROI', 'current_RR_ratio', 'PL_TDE', 'leverage',
                                   'max_profit']].T

    current_position['Values'] = current_position[0]
    current_position['Name'] = current_position.index.values.tolist()
    current_position = current_position[['Name', 'Values']]
    current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
    current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

    wight_df = pd.concat([current_position1, current_position2], axis=1, )

    wight_df.columns = ['N1', 'V1', 'N2', 'V2']
    pl, marg = postion_df['current_PL'].iloc[0], postion_df['margin'].iloc[0]

    return wight_df, pl, marg


def update_postion(csv_position_df, pos_type, risk_rate, path_bento):
    print('update_postion')
    postion_df = pd.read_csv(csv_position_df)
    print('csv_position_df', csv_position_df)
    print('postion_df1')
    print(postion_df)
    ticker = postion_df['Symbol'].iloc[0]
    ticker_b = postion_df['Symbol Bento'].iloc[0]
    print('tickerrrrrr', ticker)
    yahoo_price_df = yf.download(ticker)
    print('yahoo_price_df')
    print(yahoo_price_df)
    print(postion_df['Exp_date'])
    log_returns = np.log(yahoo_price_df["Close"] / yahoo_price_df["Close"].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    hv = log_returns.rolling(window=200).std() * np.sqrt(365)
    hv = hv.iloc[-1]
    current_price = yahoo_price_df['Close'].iloc[-1]
    if pos_type == 'F. Put':
        # df_greek, current_IV = strategist_greeks.greeks_start(ticker, postion_df['Exp_date'].values[0])
        sigma_df = get_databento.get_bento_data(ticker_b, current_price, postion_df['DTE'].values[0],
                                                postion_df['Strike'].iloc[0], 'P', path_bento)
        current_IV = sigma_df['sigma'].iloc[0] * 100

        postion_df['DAYS_remaining'] = (datetime.datetime.strptime(postion_df['Exp_date'].iloc[0],
                                                                   '%Y-%m-%d') - datetime.datetime.now()).days
        postion_df['DAYS_elapsed_TDE'] = postion_df['DTE'] - postion_df['DAYS_remaining']
        postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE']
        # print(df_greek['STRIKE'])
        print(postion_df['Strike'].iloc[0])
        current_prime = sigma_df['midprice']
        print('current_prime', current_prime)
        current_delta = sigma_df['delta'].iloc[0]
        current_vega = sigma_df['vega'].iloc[0]
        current_theta = sigma_df['theta'].iloc[0]
        current_gamma = sigma_df['gamma'].iloc[0]

        postion_df['Delta'] = current_delta * postion_df['Number_pos'].values[0]
        postion_df['Vega'] = current_vega * postion_df['Number_pos'].values[0]
        postion_df['Theta'] = current_theta * postion_df['Number_pos'].values[0]
        postion_df['Gamma'] = current_gamma * postion_df['Number_pos'].values[0]

        print('shortPutF')
        print('current_price', current_price)
        print('current_IV', current_IV)
        print('risk_rate', risk_rate)
        print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])
        print('DTE', postion_df['DTE'].values[0])
        print('Strike', postion_df['Strike'].values[0])
        print('Start_prime', postion_df['Start_prime'].values[0])

        response = shortPut(
            current_price, current_IV, risk_rate, 2000, postion_df['DAYS_remaining'].iloc[0],
            [postion_df['DAYS_remaining'].iloc[0]],
            [50], postion_df['Strike'].values[0], postion_df['Start_prime'].values[0], yahoo_price_df,
            'FUT')

        print('response', response)
        response['pop'] = float(response['pop'][0])
        response['cvar'] = float(response['cvar']) * postion_df['Multiplicator'].values[0]
        response['exp_return'] = float(response['exp_return']) * postion_df['Multiplicator'].values[0]

        print('pop_50 ', response['pop'], 'expected_profit ', response['exp_return'], 'cvar ', response['cvar'], )
        postion_df['POP_50'] = response['pop']
        postion_df['Current_expected_return'] = response['exp_return']
        postion_df['CVAR'] = response['cvar']

        postion_df['Prime_now'] = current_prime
        postion_df['Cost_to_close_Market_cost'] = (postion_df['Number_pos'] * current_prime) * \
                                                  postion_df['Multiplicator'].values[0]
        postion_df['Current_Margin'] = postion_df['Margin_start'].values[0]
        postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
        postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
        postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (
                postion_df['CVAR'] + postion_df['Current_PL'])
        postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
        postion_df['Leverage'] = (current_delta * 100 * current_price) / postion_df['Current_Margin']
        postion_df['Margin_2S_Down'] = np.max([0.1 * postion_df['Strike'], postion_df['Price_2s_down'] * 0.2 - (
                postion_df['Price_2s_down'] - postion_df['Strike'])]) * postion_df['Number_pos'].abs() * 100

        print('shortPut')
        print('current_price', current_price)
        print('current_IV', current_IV)
        print('risk_rate', risk_rate)
        print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])
        print('DTE', postion_df['DTE'].values[0])
        print('Strike', postion_df['Strike'].values[0])
        print('Start_prime', postion_df['Start_prime'].values[0])

        P_below = stats.norm.cdf(
            (np.log(postion_df['BE_lower'].iloc[0] / current_price) / (
                    hv * math.sqrt(postion_df['DAYS_remaining'].iloc[0] / 365))))

        postion_df['Current_POP_lognormal'] = 1 - P_below

        postion_df['MC_2S_Down'] = calculate_option_price('P', postion_df['Price_2s_down'].iloc[0],
                                                          postion_df['Strike'].iloc[0], risk_rate,
                                                          postion_df['HV_200'].iloc[0],
                                                          postion_df['DAYS_remaining']) * postion_df[
                                       'Number_pos'].abs() * 100

        postion_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio', ]] = postion_df[[
            '%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100
        postion_df = postion_df.round(2)

        postion_df.to_csv(csv_position_df, index=False)
        print('postion_df DONE!!!!')

        current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                       'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                       'Current_expected_return',
                                       'Current_POP_lognormal', 'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE',
                                       'Leverage',
                                       'MC_2S_Down', 'Margin_2S_Down', 'Max_profit']].T

        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )

        wight_df.columns = ['N1', 'V1', 'N2', 'V2']
        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0], \
            postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, pl, marg, pop_log

    if pos_type == 'F. Call':
        sigma_df = get_databento.get_bento_data(ticker_b, current_price, postion_df['DTE'].values[0],
                                                postion_df['Strike'].iloc[0], 'C', path_bento)
        current_IV = sigma_df['sigma'].iloc[0] * 100
        postion_df['DAYS_remaining'] = (datetime.datetime.strptime(postion_df['Exp_date'].iloc[0],
                                                                   '%Y-%m-%d') - datetime.datetime.now()).days
        postion_df['DAYS_elapsed_TDE'] = postion_df['DTE'] - postion_df['DAYS_remaining']
        postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE']

        current_prime = sigma_df['midprice']
        print('current_prime', current_prime)
        current_delta = sigma_df['delta'].iloc[0]
        current_vega = sigma_df['vega'].iloc[0]
        current_theta = sigma_df['theta'].iloc[0]
        current_gamma = sigma_df['gamma'].iloc[0]

        postion_df['Delta'] = current_delta * postion_df['Number_pos'].values[0]
        postion_df['Vega'] = current_vega * postion_df['Number_pos'].values[0]
        postion_df['Theta'] = current_theta * postion_df['Number_pos'].values[0]
        postion_df['Gamma'] = current_gamma * postion_df['Number_pos'].values[0]

        print('shortCallF')
        print('current_price', current_price)
        print('current_IV', current_IV)
        print('risk_rate', risk_rate)
        print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])
        print('DTE', postion_df['DTE'].values[0])
        print('Strike', postion_df['Strike'].values[0])
        print('Start_prime', postion_df['Start_prime'].values[0])

        response = shortCall(
            current_price, current_IV, risk_rate, 2000, postion_df['DAYS_remaining'].iloc[0],
            [postion_df['DAYS_remaining'].iloc[0]],
            [50], postion_df['Strike'].values[0], postion_df['Start_prime'].values[0], yahoo_price_df,
            'FUT')

        response['pop'] = float(response['pop'][0])
        response['cvar'] = float(response['cvar']) * postion_df['Multiplicator'].values[0]
        response['exp_return'] = float(response['exp_return']) * postion_df['Multiplicator'].values[0]

        print('pop_50 ', response['pop'], 'expected_profit ', response['exp_return'], 'cvar ', response['cvar'], )
        postion_df['POP_50'] = response['pop']
        postion_df['Current_expected_return'] = response['exp_return']
        postion_df['CVAR'] = response['cvar']

        postion_df['Prime_now'] = current_prime
        postion_df['Cost_to_close_Market_cost'] = (postion_df['Number_pos'] * current_prime) * \
                                                  postion_df['Multiplicator'].values[0]
        postion_df['Current_Margin'] = postion_df['Margin_start'].values[0]
        postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
        postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
        postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (
                postion_df['CVAR'] + postion_df['Current_PL'])
        postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
        postion_df['Leverage'] = (current_delta * 100 * current_price) / postion_df['Current_Margin']
        postion_df['Margin_2S_Down'] = np.max([0.1 * postion_df['Strike'], postion_df['Price_2s_down'] * 0.2 - (
                postion_df['Price_2s_down'] - postion_df['Strike'])]) * postion_df['Number_pos'].abs() * 100

        print(f'shortCall_{ticker}')
        print('current_price', current_price)
        print('current_IV', current_IV)
        print('risk_rate', risk_rate)
        print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])
        print('DTE', postion_df['DTE'].values[0])
        print('Strike', postion_df['Strike'].values[0])
        print('Start_prime', postion_df['Start_prime'].values[0])

        P_below = stats.norm.cdf(
            (np.log(postion_df['BE_higher'].iloc[0] / current_price) / (
                    hv * math.sqrt(postion_df['DAYS_remaining'].iloc[0] / 365))))

        postion_df['Current_POP_lognormal'] = P_below

        postion_df['MC_2S_Down'] = calculate_option_price('P', postion_df['Price_2s_down'].iloc[0],
                                                          postion_df['Strike'].iloc[0], risk_rate,
                                                          postion_df['HV_200'].iloc[0],
                                                          postion_df['DAYS_remaining']) * postion_df[
                                       'Number_pos'].abs() * 100

        postion_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio', ]] = postion_df[[
            '%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100
        postion_df = postion_df.round(2)

        postion_df.to_csv(csv_position_df, index=False)

        current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                       'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                       'Current_expected_return',
                                       'Current_POP_lognormal', 'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE',
                                       'Leverage',
                                       'MC_2S_Down', 'Margin_2S_Down', 'Max_profit']].T

        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )

        wight_df.columns = ['N1', 'V1', 'N2', 'V2']
        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0], \
            postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, pl, marg, pop_log

    if pos_type == 'F. Strangle':
        df_greek, current_IV = strategist_greeks.greeks_start(ticker, postion_df['Exp_date'].values[0])
        postion_df['DAYS_remaining'] = (datetime.datetime.strptime(postion_df['Exp_date'].iloc[0],
                                                                   '%Y-%m-%d') - datetime.datetime.now()).days
        postion_df['DAYS_elapsed_TDE'] = postion_df['DTE'] - postion_df['DAYS_remaining']
        postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE']
        print(df_greek['STRIKE'])

        current_prime_put = \
            df_greek[df_greek['STRIKE'] == postion_df['Strike_Put'].iloc[0]]['PUT'].reset_index(drop=True).iloc[
                0]
        current_prime_call = \
            df_greek[df_greek['STRIKE'] == postion_df['Strike_Call'].iloc[0]]['CALL'].reset_index(drop=True).iloc[
                0]
        current_prime = current_prime_put + current_prime_call

        print('current_prime', current_prime)
        current_delta = \
            (df_greek[df_greek['STRIKE'] == postion_df['Strike_Put'].iloc[0]]['PUTDEL'].reset_index(
                drop=True).iloc[0] +
             df_greek[df_greek['STRIKE'] == postion_df['Strike_Call'].iloc[0]]['DELTA'].reset_index(
                 drop=True).iloc[0])

        current_vega = abs(
            df_greek[df_greek['STRIKE'] == postion_df['Strike_Call'].iloc[0]]['VEGA'].reset_index(drop=True).iloc[
                0] +
            df_greek[df_greek['STRIKE'] == postion_df['Strike_Put'].iloc[0]]['VEGA'].reset_index(drop=True).iloc[
                0])
        current_theta = abs(
            df_greek[df_greek['STRIKE'] == postion_df['Strike_Call'].iloc[0]]['THETA'].reset_index(drop=True).iloc[
                0] +
            df_greek[df_greek['STRIKE'] == postion_df['Strike_Put'].iloc[0]]['THETA'].reset_index(drop=True).iloc[
                0])
        current_gamma = abs(
            df_greek[df_greek['STRIKE'] == postion_df['Strike_Call'].iloc[0]]['GAMMA'].reset_index(drop=True).iloc[
                0] +
            df_greek[df_greek['STRIKE'] == postion_df['Strike_Put'].iloc[0]]['GAMMA'].reset_index(drop=True).iloc[
                0])

        postion_df['Delta'] = current_delta * -100
        postion_df['Vega'] = current_vega * -100
        postion_df['Theta'] = -current_theta * -100
        postion_df['Gamma'] = current_gamma * -100

        response = shortStrangle(
            current_price, current_IV, risk_rate, 2000, postion_df['DTE'].values[0], [postion_df['DTE'].values[0]],
            [50], postion_df['Strike_Call'].values[0], postion_df['Start_prime'].values[0],
            postion_df['Strike_Put'].values[0], 0, yahoo_price_df
        )

        response['pop'] = float(response['pop'][0]) / postion_df['Multiplicator'].values[0]
        response['cvar'] = float(response['cvar']) * postion_df['Multiplicator'].values[0]
        response['exp_return'] = float(response['exp_return']) * postion_df['Multiplicator'].values[0]

        print('pop_50 ', response['pop'], 'expected_profit ', response['exp_return'], 'cvar ', response['cvar'], )
        postion_df['POP_50'] = response['pop']
        postion_df['Current_expected_return'] = response['exp_return']
        postion_df['CVAR'] = response['cvar']

        postion_df['Cost_to_close_Market_cost'] = ((postion_df['Number_pos'] * current_prime) * \
                                                   postion_df['Multiplicator'].values[0]) - postion_df['Commission']
        postion_df['Current_Margin'] = postion_df['Margin_start'].values[0]
        postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
        postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
        postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (
                postion_df['CVAR'] + postion_df['Current_PL'])
        postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
        postion_df['Leverage'] = (current_delta * 100 * current_price) / postion_df['Current_Margin']
        # new_position_df['Margin_2S_Down'] = np.max([0.1 * new_position_df['Strike'], new_position_df['Price_2s_down'] * 0.2 - (
        #             new_position_df['Price_2s_down'] - new_position_df['Strike'])]) * new_position_df['Number_pos'].abs() * 100

        print('shortPut')
        print('current_price', current_price)
        print('current_IV', current_IV)
        print('risk_rate', risk_rate)
        print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])
        print('DTE', postion_df['DTE'].values[0])
        print('Start_prime', postion_df['Start_prime'].values[0])

        P_below = stats.norm.cdf(
            (np.log(postion_df['BE_lower'].iloc[0] / current_price) / (
                    postion_df['HV_200'].iloc[0] * math.sqrt(postion_df['DAYS_remaining'].iloc[0] / 365))))

        postion_df['Current_POP_lognormal'] = 1 - P_below

        # postion_df['MC_2S_Down'] = calculate_option_price('P', postion_df['Price_2s_down'].iloc[0],
        #                           postion_df['Strike'].iloc[0], risk_rate, postion_df['HV_200'].iloc[0],
        #                           postion_df['DAYS_remaining']) * postion_df['Number_pos'].abs() * 100

        postion_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio', ]] = postion_df[[
            '%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100
        postion_df = postion_df.round(2)

        postion_df.to_csv(csv_position_df, index=False)

        current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                       'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                       'Current_expected_return',
                                       'Current_POP_lognormal', 'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE',
                                       'Leverage',
                                       'Max_profit']].T

        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )

        wight_df.columns = ['N1', 'V1', 'N2', 'V2']
        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0], \
            postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, pl, marg, pop_log

    if pos_type == 'Put Sell':
        quotes_df = market_data(postion_df['Contract'].iloc[0], KEY)
        print(quotes_df)
        postion_df['DAYS_remaining'] = (datetime.datetime.strptime(postion_df['Exp_date'].iloc[0],
                                                                   '%Y-%m-%d') - datetime.datetime.now()).days
        postion_df['DAYS_elapsed_TDE'] = postion_df['DTE'] - postion_df['DAYS_remaining']
        postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE']
        postion_df['Prime_now'] = quotes_df['ask']
        postion_df['Cost_to_close_Market_cost'] = (postion_df['Number_pos'] * quotes_df['ask']) * 100
        postion_df['Current_Margin'] = np.max(
            [0.2 * current_price - (current_price - postion_df['Strike']), 0.1 * postion_df['Strike']]) * postion_df[
                                           'Number_pos'].abs() * 100
        postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
        postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
        postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (
                postion_df['Max_Risk'] + postion_df['Current_PL'])
        postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
        postion_df['Leverage'] = (quotes_df['delta'] * 100 * current_price) / postion_df['Current_Margin']
        postion_df['Margin_2S_Down'] = np.max([0.1 * postion_df['Strike'], postion_df['Price_2s_down'] * 0.2 - (
                postion_df['Price_2s_down'] - postion_df['Strike'])]) * postion_df['Number_pos'].abs() * 100

        pop_50, expected_profit, avg_dtc = get_proba_50_put(current_price, yahoo_price_df, postion_df['Strike'].iloc[0],
                                                            postion_df['Start_prime'].iloc[0],
                                                            quotes_df['iv'].iloc[0] * 100,
                                                            postion_df['DAYS_remaining'].iloc[0], risk_rate)
        print('pop_50 ', pop_50, 'expected_profit ', expected_profit, 'avg_dtc ', avg_dtc, )
        postion_df['POP_50'] = pop_50
        postion_df['Current_expected_return'] = expected_profit
        postion_df['DTE_Target'] = avg_dtc[0]

        P_below = stats.norm.cdf(
            (np.log(postion_df['BE_lower'].iloc[0] / current_price) / (
                    postion_df['HV_200'].iloc[0] * math.sqrt(postion_df['DAYS_remaining'].iloc[0] / 365))))

        postion_df['Current_POP_lognormal'] = 1 - P_below

        postion_df['MC_2S_Down'] = calculate_option_price('P', postion_df['Price_2s_down'].iloc[0],
                                                          postion_df['Strike'].iloc[0], risk_rate,
                                                          postion_df['HV_200'].iloc[0],
                                                          postion_df['DAYS_remaining']) * postion_df[
                                       'Number_pos'].abs() * 100

        postion_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio', ]] = postion_df[[
            '%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100
        postion_df = postion_df.round(2)

        postion_df.to_csv(csv_position_df)

        current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                       'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                       'Current_expected_return',
                                       'Current_POP_lognormal', 'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE',
                                       'Leverage',
                                       'MC_2S_Down', 'Margin_2S_Down']].T

        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )

        wight_df.columns = ['N1', 'V1', 'N2', 'V2']
        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0], \
            postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, pl, marg, pop_log

    if pos_type == 'ITM Calendar':
        # calculate historical volatility
        log_returns = np.log(yahoo_price_df['Close'] / yahoo_price_df['Close'].shift(1))
        hv = log_returns.rolling(window=252).std() * np.sqrt(252)
        hv = hv[-1]

        quotes_df_long = market_data(postion_df['Contract_long'].iloc[0], KEY)
        quotes_df_short = market_data(postion_df['Contract_short'].iloc[0], KEY)
        postion_df['DAYS_remaining_short'] = (datetime.datetime.strptime(postion_df['Exp_date_short'].iloc[0],
                                                                         '%Y-%m-%d') - datetime.datetime.now()).days
        postion_df['DAYS_remaining_long'] = (datetime.datetime.strptime(postion_df['Exp_date_long'].iloc[0],
                                                                        '%Y-%m-%d') - datetime.datetime.now()).days
        postion_df['DAYS_elapsed_TDE'] = postion_df['DTE'] - postion_df['DAYS_remaining_short']
        postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE']
        postion_df['Prime_now'] = quotes_df_long['mid'] - quotes_df_short['mid']
        postion_df['Cost_to_close_Market_cost'] = (postion_df['Number_pos'] * postion_df['Prime_now']) * 100
        postion_df['Current_Margin'] = postion_df['Open_cost']
        postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
        postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
        postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (
                postion_df['Max_Risk'] + postion_df['Current_PL'])
        postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
        postion_df['Leverage'] = ((quotes_df_long['delta'] + quotes_df_short['delta']) * 100 * current_price) / \
                                 postion_df['Current_Margin']

        proba_50, avg_dtc = get_proba_50_calendar(current_price, yahoo_price_df,
                                                  postion_df['Strike_long'].iloc[0],
                                                  postion_df['Start_prime_long'].iloc[0],
                                                  postion_df['Strike_short'].iloc[0],
                                                  postion_df['Start_prime_short'].iloc[0],
                                                  quotes_df_short['iv'].iloc[0] * 100,
                                                  quotes_df_long['iv'].iloc[0] * 100,
                                                  postion_df['DAYS_remaining_short'].iloc[0],
                                                  postion_df['DAYS_remaining_long'].iloc[0], risk_rate)

        expected_return = expected_return_calc(quotes_df_short['iv'].iloc[0], quotes_df_long['iv'].iloc[0],
                                               current_price,
                                               hv, postion_df['DAYS_remaining_short'].iloc[0],
                                               postion_df['DAYS_remaining_long'].iloc[0],
                                               postion_df['Strike_long'].iloc[0], postion_df['Strike_short'].iloc[0],
                                               postion_df['Start_prime_long'].iloc[0],
                                               postion_df['Start_prime_short'].iloc[0])

        print('proba_50 ', proba_50, 'expected_return ', expected_return, 'avg_dtc ', avg_dtc, )
        postion_df['POP_50'] = proba_50
        postion_df['Current_expected_return'] = expected_return
        postion_df['DTE_Target'] = avg_dtc[0]

        P_below = stats.norm.cdf(
            (np.log(postion_df['BE_lower'].iloc[0] / current_price) / (
                    postion_df['HV_200'].iloc[0] * math.sqrt(postion_df['DAYS_remaining_short'].iloc[0] / 365))))

        postion_df['Current_POP_lognormal'] = 1 - P_below

        # postion_df['MC_2S_Down'] = calculate_option_price('P', postion_df['Price_2s_down'].iloc[0],
        #                           postion_df['Strike'].iloc[0], risk_rate, postion_df['HV_200'].iloc[0],
        #                           postion_df['DAYS_remaining_short']) * postion_df['Number_pos'].abs() * 100

        postion_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio', ]] = postion_df[[
            '%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100
        postion_df = postion_df.round(2)

        postion_df.to_csv(csv_position_df)

        current_position = postion_df[
            ['DAYS_remaining_short', 'DAYS_remaining_long', 'DAYS_elapsed_TDE', '%_days_elapsed',
             'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL', 'Current_expected_return',
             'Current_POP_lognormal', 'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE', 'Leverage',
             ]].T

        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )

        wight_df.columns = ['N1', 'V1', 'N2', 'V2']
        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0], \
            postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, pl, marg, pop_log




def barchart_selection(short_df, long_df, side, short_dte, long_dte, tick, rate, percentage_array, multiplier):
    short_df = short_df[short_df['Type'] == side][:-1]
    short_df.columns = short_df.columns.str.lower()
    short_df['strike'] = short_df['strike'].str.replace('-', '.')
    short_df['iv'] = short_df['iv'].str.replace('%', '')
    short_df[['strike', 'last', 'iv']] = short_df[['strike', 'last', 'iv']].astype('float')
    short_df['side'] = side
    short_df['count'] = -1
    short_df['prime'] = short_df['last']
    short_df['days_to_exp'] = short_dte


    long_df = long_df[long_df['Type'] == side][:-1]
    long_df.columns = long_df.columns.str.lower()
    long_df['strike'] = long_df['strike'].str.replace('-', '.')
    long_df['iv'] = long_df['iv'].str.replace('%', '')
    long_df[['strike', 'last', 'iv']] = long_df[['strike', 'last', 'iv']].astype('float')
    long_df['side'] = side
    long_df['prime'] = long_df['last']
    long_df['count'] = 1
    long_df['days_to_exp'] = long_dte



    yahoo_data = yf.download(tick, progress=False)['2018-01-01':]
    underlying = yahoo_data['Close'].iloc[-1]
    long_df['underlying'] = underlying
    short_df['underlying'] = underlying


    std_short = yahoo_data['Close'][-(short_dte + 1):].std()
    std_long = yahoo_data['Close'][-(long_dte + 1):].std()

    short_df = short_df[short_df['strike'] >= (underlying - (std_short*0.25))]
    short_df = short_df[short_df['strike'] <= (underlying + (std_short*0.25))]

    long_df = long_df[long_df['strike'] >= underlying - (std_long*0.5)]
    long_df = long_df[long_df['strike'] <= underlying]

    print('short_df')
    print(short_df)
    print('long_df')
    print(long_df)

    # Список для хранения всех комбинаций DataFrame
    combined_dfs = []

    # Перебираем все строки из df1 и df2
    for i, row1 in short_df.iterrows():
        for j, row2 in long_df.iterrows():
            # Создаём DataFrame из двух строк: одна из df1 и одна из df2
            combined_df = pd.concat([row1.to_frame().T, row2.to_frame().T], axis=0)
            combined_dfs.append(combined_df)

    return_df = pd.DataFrame()

    print('combined_dfs')
    print(combined_dfs)
    for df in combined_dfs:
        # print('num', num)
        print('df')
        print(df)

        df['symbol'] = tick
        df['rate'] = rate
        df['percentage_array'] = percentage_array
        df['multiplier'] = multiplier

        min_dte = df['days_to_exp'].min()
        response_df = pd.DataFrame()

        for num, row in df.iterrows():

            response = pd.DataFrame(
                solo_position_calc_new(row, yahoo_data, row['iv'],
                                       row['days_to_exp'], min_dte, min_dte,
                                       row['underlying'], row['strike'],
                                       row['prime']))
            response[['cvar', 'exp_return']] = (response[['cvar', 'exp_return']] * np.abs(row['count']) * row['multiplier'])
            response_df = pd.concat([response_df, response])

        response_df = response_df.mean()
        response_df['Short_Strike'] = df[df['count'] == -1]['strike'].values[0]
        response_df['Long_Strike'] = df[df['count'] == 1]['strike'].values[0]

        response_df = response_df.round(4)
        return_df = pd.concat([return_df, response_df.to_frame().T], axis=0)
    print('return_df')
    print(return_df)
    return_df['top_score'] = return_df['pop'] * return_df['exp_return']
    best_df = return_df[return_df['top_score'] == return_df['top_score'].max()]
    return return_df, best_df