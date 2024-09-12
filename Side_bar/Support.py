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
from .popoption.ShortPut import shortPut
from scipy.stats import norm
from .popoption.poptions import *
from .popoption.ShortCall import shortCall
from .popoption.ShortStrangle import shortStrangle
from .popoption.PutCalendar import putCalendar
from .option_strategist import strategist_greeks
from .databentos import get_databento


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


def solo_position_calc(row, yahoo_stock, sigma, days_to_expiration, days_to_expiration_min, closing_days_array,
                       underlying, strike, prime):
    trials = 3_000
    pos_type = row['Position_side'].iloc[0]
    short_long = row['Number_pos_short'].iloc[0]
    print('pos_type', pos_type)
    rate = row['Rate'].iloc[0]
    # closing_days_array = row['closing_days_array']
    percentage_array = row['Percentage_Array'].iloc[0]

    if '-' in row['Symbol'].iloc[0] or '=' in row['Symbol'].iloc[0]:
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

    if pos_type == 'PUT':
        if short_long > 0:
            response = longPut(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                               [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
        else:
            response = shortPut(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
    if pos_type == 'CALL':
        if short_long > 0:
            response = longCall(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
        else:
            response = shortCall(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                 [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)

    if pos_type == 'STOCK':
        if short_long > 0:
            response = longStock(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                 [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
        else:
            response = shortStock(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                  [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)

    return response


def solo_position_calc_covered(row, yahoo_stock, sigma, days_to_expiration, days_to_expiration_min, closing_days_array,
                               underlying, strike, prime):
    trials = 3_000
    pos_type = row['Position_side'].iloc[0]
    short_long = row['Number_pos'].iloc[0]
    print('pos_type', pos_type)
    rate = row['Rate'].iloc[0]
    # closing_days_array = row['closing_days_array']
    percentage_array = row['Percentage_Array'].iloc[0]

    if '-' in row['Symbol'].iloc[0] or '=' in row['Symbol'].iloc[0]:
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

    if pos_type == 'PUT':
        if short_long > 0:
            response = longPut(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                               [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
        else:
            response = shortPut(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
    if pos_type == 'CALL':
        if short_long > 0:
            response = longCall(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
        else:
            response = shortCall(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                 [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)

    if pos_type == 'STOCK':
        if short_long > 0:
            response = longStock(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                 [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)
        else:
            response = shortStock(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
                                  [closing_days_array], [percentage_array], strike, prime, yahoo_stock, instr_type)

    return response


def emulate_position(input_new_df, path, path_bento, risk_rate):
    new_position_df = pd.DataFrame()
    pos_type = input_new_df['Position_type'].values[0]
    ticker = input_new_df['Symbol'].values[0]
    ticker_b = input_new_df['Symbol Bento'].values[0]
    print('/', ticker, '/')
    print(ticker)
    print(input_new_df['Symbol'].values)
    yahoo_price_df = yf.download(ticker)
    print(yahoo_price_df)
    start_price = yahoo_price_df['Close'].iloc[-1]
    KEY = credential.password['marketdata_KEY']
    log_returns = np.log(yahoo_price_df["Close"] / yahoo_price_df["Close"].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    hv = log_returns.rolling(window=200).std() * np.sqrt(365)
    hv = hv.iloc[-1]
    current_price = yahoo_price_df['Close'].iloc[-1]

    if pos_type == 'F. Covered':
        input_new_df['DTE'] = (input_new_df['Exp_date'] - input_new_df['Start_date']).values[0].days
        input_new_df['Open_cost'] = (input_new_df['Number_pos'].iloc[0] * input_new_df['Prime'].values[
            0]) * input_new_df['Multiplicator'].values[0]
        input_new_df['HV_200'] = hv
        input_new_df['Margin_start'] = input_new_df['Margin']
        input_new_df['DAYS_remaining'] = (input_new_df['Exp_date'].iloc[0] - datetime.datetime.now().date()).days

        min_dte = input_new_df['DTE'].iloc[0]

        print(input_new_df, yahoo_price_df, input_new_df['IV'].iloc[0], input_new_df['DTE'].iloc[0], min_dte,
              min_dte, input_new_df['Underlying'].iloc[0],
              input_new_df['Strike'].iloc[0], input_new_df['Prime'].iloc[0])
        opt_side_response = pd.DataFrame(
            solo_position_calc_covered(input_new_df, yahoo_price_df, input_new_df['IV'].iloc[0],
                                       input_new_df['DTE'].iloc[0], min_dte, min_dte,
                                       input_new_df['Underlying'].iloc[0], input_new_df['Strike'].iloc[0],
                                       input_new_df['Prime'].iloc[0]))
        opt_side_response[['cvar', 'exp_return']] = opt_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        input_new_df['Position_side'] = 'STOCK'

        stock_side_response = pd.DataFrame(
            solo_position_calc_covered(input_new_df, yahoo_price_df, input_new_df['IV'].iloc[0],
                                       input_new_df['DTE'].iloc[0], min_dte, min_dte,
                                       input_new_df['Underlying_stock'].iloc[0], input_new_df['Strike'].iloc[0],
                                       input_new_df['Prime'].iloc[0]))
        stock_side_response[['cvar', 'exp_return']] = stock_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        print('opt_side_response')
        print(opt_side_response)
        print('stock_side_response')
        print(stock_side_response)

        input_new_df['POP_50'] = (opt_side_response['pop'] + stock_side_response['pop']) / 2
        input_new_df['Current_expected_return'] = (opt_side_response['exp_return'] + stock_side_response[
            'exp_return']) / 2
        input_new_df['CVAR'] = (opt_side_response['cvar'] + stock_side_response['cvar']) / 2

        input_new_df = input_new_df.round(3)

        return input_new_df[['POP_50', 'Current_expected_return', 'CVAR']]

    if pos_type == 'F. Diagonal':
        input_new_df['Start_prime'] = input_new_df['Prime_short'] - input_new_df['Prime_long']
        input_new_df['underlying_long'] = input_new_df['Start_underlying_long']
        input_new_df['underlying_short'] = input_new_df['Start_underlying_short']
        input_new_df['DTE_short'] = (input_new_df['Exp_date_short'] - input_new_df['Start_date']).values[0].days
        input_new_df['DTE_long'] = (input_new_df['Exp_date_long'] - input_new_df['Start_date']).values[0].days
        input_new_df['Open_cost'] = (input_new_df['Number_pos_long'].iloc[0] * input_new_df['Start_prime'].values[
            0]) * input_new_df['Multiplicator'].values[0]
        input_new_df['HV_200'] = hv
        input_new_df['Margin_start'] = input_new_df['Margin']
        input_new_df['DAYS_remaining'] = (input_new_df['Exp_date_short'].iloc[0] - datetime.datetime.now().date()).days

        print(input_new_df['DTE_short'].iloc[0])
        print(input_new_df['DTE_long'].iloc[0])
        min_dte = np.min([input_new_df['DTE_short'].iloc[0], input_new_df['DTE_long'].iloc[0]])
        min_closing_days_array = min_dte

        print('long_side_response')
        print(input_new_df, yahoo_price_df, input_new_df['IV_LONG'].iloc[0], input_new_df['DTE_long'].iloc[0], min_dte,
              min_closing_days_array, input_new_df['Start_underlying_long'].iloc[0],
              input_new_df['Strike_long'].iloc[0], input_new_df['Prime_long'].iloc[0])
        long_side_response = pd.DataFrame(
            solo_position_calc(input_new_df, yahoo_price_df, input_new_df['IV_LONG'].iloc[0],
                               input_new_df['DTE_long'].iloc[0], min_dte, min_closing_days_array,
                               input_new_df['Start_underlying_long'].iloc[0], input_new_df['Strike_long'].iloc[0],
                               input_new_df['Prime_long'].iloc[0]))
        long_side_response[['cvar', 'exp_return']] = long_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos_long'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        short_side_response = pd.DataFrame(
            solo_position_calc(input_new_df, yahoo_price_df, input_new_df['IV_SHORT'].iloc[0],
                               input_new_df['DTE_short'].iloc[0], min_dte, min_closing_days_array,
                               input_new_df['Start_underlying_short'].iloc[0], input_new_df['Strike_short'].iloc[0],
                               input_new_df['Prime_short'].iloc[0]))
        short_side_response[['cvar', 'exp_return']] = short_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos_short'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        print('long_side_response')
        print(long_side_response)
        print('short_side_response')
        print(short_side_response)

        input_new_df['POP_50'] = (short_side_response['pop'] + long_side_response['pop']) / 2
        input_new_df['Current_expected_return'] = (short_side_response['exp_return'] + long_side_response[
            'exp_return']) / 2
        input_new_df['CVAR'] = (short_side_response['cvar'] + long_side_response['cvar']) / 2

        input_new_df = input_new_df.round(3)

        return input_new_df[['POP_50', 'Current_expected_return', 'CVAR']]

    if pos_type == 'F. Strangle':
        input_new_df['Start_prime'] = input_new_df['Prime_Call'] + input_new_df['Prime_Put']
        input_new_df['DTE'] = (input_new_df['Exp_date'] - input_new_df['Start_date']).values[0].days

        input_new_df['Open_cost'] = (input_new_df['Number_pos'].iloc[0] * input_new_df['Start_prime'].values[
            0]) * input_new_df['Multiplicator'].values[0]

        min_dte = np.min([input_new_df['DTE'].iloc[0], input_new_df['DTE'].iloc[0]])
        min_closing_days_array = min_dte

        input_new_df['Number_pos_short'] = -1

        input_new_df['Position_side'] = 'PUT'
        put_side_response = pd.DataFrame(
            solo_position_calc(input_new_df, yahoo_price_df, input_new_df['IV_Put'].iloc[0],
                               input_new_df['DTE'].iloc[0], input_new_df['DTE'].values[0],
                               input_new_df['DTE'].values[0],
                               input_new_df['Start_underlying'].iloc[0], input_new_df['Strike_Put'].iloc[0],
                               input_new_df['Prime_Put'].iloc[0]))
        put_side_response[['cvar', 'exp_return']] = put_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        input_new_df['Position_side'] = 'CALL'
        call_side_response = pd.DataFrame(
            solo_position_calc(input_new_df, yahoo_price_df, input_new_df['IV_Call'].iloc[0],
                               input_new_df['DTE'].iloc[0], input_new_df['DTE'].values[0],
                               input_new_df['DTE'].values[0],
                               input_new_df['Start_underlying'].iloc[0], input_new_df['Strike_Call'].iloc[0],
                               input_new_df['Prime_Call'].iloc[0]))
        call_side_response[['cvar', 'exp_return']] = call_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        input_new_df['POP_50'] = (put_side_response['pop'] + call_side_response['pop']) / 2
        input_new_df['Current_expected_return'] = (put_side_response['exp_return'] + call_side_response[
            'exp_return']) / 2
        input_new_df['CVAR'] = (put_side_response['cvar'] + call_side_response['cvar']) / 2

        input_new_df = input_new_df.round(3)

        return input_new_df[['POP_50', 'Current_expected_return', 'CVAR']]


def create_new_postion(input_new_df, path, path_bento, risk_rate):
    new_position_df = pd.DataFrame()
    pos_type = input_new_df['Position_type'].values[0]
    ticker = input_new_df['Symbol'].values[0]
    ticker_b = input_new_df['Symbol Bento'].values[0]
    print('/', ticker, '/')
    print(ticker)
    print(input_new_df['Symbol'].values)
    yahoo_price_df = yf.download(ticker)
    print(yahoo_price_df)
    start_price = yahoo_price_df['Close'].iloc[-1]
    KEY = credential.password['marketdata_KEY']
    log_returns = np.log(yahoo_price_df["Close"] / yahoo_price_df["Close"].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    hv = log_returns.rolling(window=200).std() * np.sqrt(365)
    hv = hv.iloc[-1]
    current_price = yahoo_price_df['Close'].iloc[-1]

    if pos_type == 'F. Put':
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
        sigma_df = get_databento.get_bento_data(ticker_b, current_price, new_position_df['DTE'].values[0],
                                                input_new_df['Strike_o_p'].iloc[0], 'P', path_bento)
        current_IV = sigma_df['sigma'].iloc[0] * 100
        new_position_df['Open_cost'] = (new_position_df['Number_pos'].iloc[0] * new_position_df['Start_prime'].values[
            0]) * new_position_df['Multiplicator'].values[0]
        new_position_df['Start_delta'] = input_new_df['Delta_o_p']
        new_position_df['HV_200'] = hv
        new_position_df['Price_2s_down'] = start_price - (
                2 * start_price * hv * (math.sqrt(new_position_df['DTE'].iloc[0] / 365)))
        new_position_df['Max_profit'] = new_position_df['Open_cost'].abs()
        # max_risk = calculate_option_price('P', new_position_df['Price_2s_down'].values[0], new_position_df['Strike'].values[0], risk_rate, current_IV, new_position_df['DTE'].values[0])
        new_position_df['Max_Risk'] = np.max([new_position_df['Strike'].iloc[0] * 0.1, (
                new_position_df['Price_2s_down'].iloc[0] - (
                new_position_df['Price_2s_down'].iloc[0] - new_position_df['Strike'].iloc[0])) * 0.2]) * \
                                      new_position_df['Open_cost'].abs() * 100
        new_position_df['BE_lower'] = new_position_df['Strike'] - new_position_df['Start_prime']
        new_position_df['RR_RATIO'] = new_position_df['Max_profit'] / new_position_df['Max_Risk']
        new_position_df['MP/DTE'] = new_position_df['Max_profit'] / new_position_df['DTE']
        new_position_df['Profit_Target'] = new_position_df['Max_profit'] * 0.5 - (new_position_df['Commission'])

        response = shortPut(
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

        new_position_df['POP_50'] = response['pop'] * 100
        new_position_df['Current_expected_return'] = response['exp_return']
        new_position_df['CVAR'] = response['cvar']

        new_position_df['DAYS_remaining'] = (new_position_df['Exp_date'].iloc[0] - datetime.datetime.now().date()).days
        new_position_df['DAYS_elapsed_TDE'] = new_position_df['DTE'] - new_position_df['DAYS_remaining']
        new_position_df['%_days_elapsed'] = new_position_df['DAYS_elapsed_TDE'] / new_position_df['DTE']

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
            (np.log(new_position_df['BE_lower'].iloc[0] / current_price) / (
                    new_position_df['HV_200'].iloc[0] * math.sqrt(new_position_df['DAYS_remaining'].iloc[0] / 365))))

        new_position_df['Current_POP_lognormal'] = 1 - P_below

        new_position_df['MC_2S_Down'] = calculate_option_price('P', new_position_df['Price_2s_down'].iloc[0],
                                                               new_position_df['Strike'].iloc[0], risk_rate,
                                                               new_position_df['HV_200'].iloc[0],
                                                               new_position_df['DAYS_remaining']) * new_position_df[
                                            'Number_pos'].abs() * 100

        new_position_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio', ]] = \
            new_position_df[[
                '%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100
        postion_df = new_position_df.round(2)

        new_position_df.to_csv(f"{path}{ticker}_{new_position_df['Start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                               index=False)

    if pos_type == 'F. Covered':
        input_new_df['DTE'] = (input_new_df['Exp_date'] - input_new_df['Start_date']).values[0].days

        input_new_df['Open_cost'] = (input_new_df['Number_pos'].iloc[0] * input_new_df['Prime'].values[
            0]) * input_new_df['Multiplicator'].values[0]
        input_new_df['HV_200'] = hv
        input_new_df['Margin_start'] = input_new_df['Margin']
        input_new_df['DAYS_remaining'] = (input_new_df['Exp_date'].iloc[0] - datetime.datetime.now().date()).days

        min_dte = input_new_df['DTE'].iloc[0]

        print(input_new_df, yahoo_price_df, input_new_df['IV'].iloc[0], input_new_df['DTE'].iloc[0], min_dte,
              min_dte, input_new_df['Underlying'].iloc[0],
              input_new_df['Strike'].iloc[0], input_new_df['Prime'].iloc[0])
        opt_side_response = pd.DataFrame(
            solo_position_calc_covered(input_new_df, yahoo_price_df, input_new_df['IV'].iloc[0],
                                       input_new_df['DTE'].iloc[0], min_dte, min_dte,
                                       input_new_df['Underlying'].iloc[0], input_new_df['Strike'].iloc[0],
                                       input_new_df['Prime'].iloc[0]))
        opt_side_response[['cvar', 'exp_return']] = opt_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        input_new_df['Position_side'] = 'STOCK'

        stock_side_response = pd.DataFrame(
            solo_position_calc_covered(input_new_df, yahoo_price_df, input_new_df['IV'].iloc[0],
                                       input_new_df['DTE'].iloc[0], min_dte, min_dte,
                                       input_new_df['Underlying_stock'].iloc[0], input_new_df['Strike'].iloc[0],
                                       input_new_df['Prime'].iloc[0]))
        stock_side_response[['cvar', 'exp_return']] = stock_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        print('opt_side_response')
        print(opt_side_response)
        print('stock_side_response')
        print(stock_side_response)

        input_new_df['POP_50'] = (opt_side_response['pop'] + stock_side_response['pop']) / 2
        input_new_df['Current_expected_return'] = (opt_side_response['exp_return'] + stock_side_response[
            'exp_return']) / 2
        input_new_df['CVAR'] = (opt_side_response['cvar'] + stock_side_response['cvar']) / 2

        print('input_new_df')
        print(input_new_df)

        # sigma_df_short = get_databento.get_bento_data(ticker, ticker_b, current_price, new_position_df['DTE_short'].values[0],
        #                                        new_position_df['Strike_short'].iloc[0], 'C', path_bento)
        # sigma_df_long = get_databento.get_bento_data(ticker, ticker_b, current_price, new_position_df['DTE_long'].values[0],
        #                                         new_position_df['Strike_long'].iloc[0], 'C', path_bento)
        # current_IV = (sigma_df_short['iv'].iloc[0] + sigma_df_long['iv'].iloc[0])/2 * 100

        input_new_df['Price_2s_down'] = start_price - (
                2 * start_price * hv * (math.sqrt(input_new_df['DTE'].iloc[0] / 365)))
        input_new_df['Max_profit'] = input_new_df['Open_cost'].abs()
        # max_risk = calculate_option_price('P', new_position_df['Price_2s_down'].values[0], new_position_df['Strike'].values[0], risk_rate, current_IV, new_position_df['DTE'].values[0])
        # new_position_df['Max_Risk'] = np.max([new_position_df['Strike'].iloc[0] * 0.1, (
        #             new_position_df['Price_2s_down'].iloc[0] - (
        #                 new_position_df['Price_2s_down'].iloc[0] - new_position_df['Strike'].iloc[0])) * 0.2]) * \
        #                               new_position_df['Open_cost'].abs() * 100
        # new_position_df['BE_higher'] = new_position_df['Strike'] + new_position_df['Start_prime']
        # new_position_df['RR_RATIO'] = new_position_df['Max_profit'] / new_position_df['Max_Risk']
        input_new_df['MP/DTE'] = input_new_df['Max_profit'] / input_new_df['DTE']
        input_new_df['Profit_Target'] = input_new_df['Max_profit'] * 0.5 - (input_new_df['Commission'])

        local_side = 'p'
        if input_new_df['Position_side'].iloc[0] == 'CALL':
            local_side = 'c'

        value_opt, delta_opt, gamma_opt, theta_opt, vega_opt, rho_opt = _gbs(local_side, input_new_df[
            'Underlying'].iloc[0], input_new_df['Strike'].iloc[0], input_new_df['DTE'].iloc[
                                                                                 0] / 365, 0.04, 0.04,
                                                                             input_new_df['IV'].iloc[
                                                                                 0] / 100)

        delta_opt_bs, gamma_opt_bs, theta_opt_bs, vega_opt_bs = get_black_scholes_greeks(local_side, input_new_df[
            'Underlying'].iloc[0], input_new_df['Strike'].iloc[0], risk_rate, input_new_df['IV'].iloc[
                                                                                             0] / 100, input_new_df[
                                                                                             'DTE'].iloc[
                                                                                             0])

        input_new_df['delta'] = ((delta_opt * input_new_df['Number_pos'].iloc[0]) - 1)
        input_new_df['gamma'] = (gamma_opt * input_new_df['Number_pos'].iloc[0])
        input_new_df['theta'] = (theta_opt * input_new_df['Number_pos'].iloc[0])
        input_new_df['vega'] = (vega_opt * input_new_df['Number_pos'].iloc[0])
        input_new_df['rho'] = (rho_opt * input_new_df['Number_pos'].iloc[0])

        input_new_df['theta_bs'] = (theta_opt_bs * input_new_df['Number_pos'].iloc[0])
        input_new_df['vega_bs'] = (vega_opt_bs * input_new_df['Number_pos'].iloc[0])

        input_new_df[['theta_bs', 'vega_bs']] = input_new_df[['theta_bs', 'vega_bs']] * input_new_df['Multiplicator'].iloc[0]
        #
        current_prime = input_new_df['Prime'] * input_new_df['Number_pos']
        print('current_prime', current_prime)

        input_new_df['DAYS_remaining'] = (input_new_df['Exp_date'].iloc[0] - datetime.datetime.now().date()).days
        input_new_df['DAYS_elapsed_TDE'] = input_new_df['DTE'] - input_new_df['DAYS_remaining']
        input_new_df['%_days_elapsed'] = input_new_df['DAYS_elapsed_TDE'] / input_new_df['DTE']

        input_new_df['Prime_now'] = current_prime
        input_new_df['Cost_to_close_Market_cost'] = input_new_df['Open_cost'] - input_new_df['Commission']

        input_new_df['Current_Margin'] = input_new_df['Margin_start'].values[0]
        input_new_df['Current_PL'] = input_new_df['Cost_to_close_Market_cost'] - input_new_df['Open_cost']
        input_new_df['Current_ROI'] = input_new_df['Current_PL'] / input_new_df['Current_Margin']

        input_new_df['PL_TDE'] = input_new_df['Current_PL'] / input_new_df['DAYS_elapsed_TDE']

        input_new_df[['%_days_elapsed', 'Current_ROI']] = input_new_df[['%_days_elapsed', 'Current_ROI']] * 100

        input_new_df = input_new_df.round(4)

        input_new_df.to_csv(f"{path}{ticker}_{input_new_df['Start_date'].values[0].strftime('%Y-%m-%d')}.csv",
                            index=False)

    if pos_type == 'F. Diagonal':

        input_new_df['Start_prime'] = input_new_df['Prime_short'] - input_new_df['Prime_long']
        input_new_df['underlying_long'] = input_new_df['Start_underlying_long']
        input_new_df['underlying_short'] = input_new_df['Start_underlying_short']
        input_new_df['DTE_short'] = (input_new_df['Exp_date_short'] - input_new_df['Start_date']).values[0].days
        input_new_df['DTE_long'] = (input_new_df['Exp_date_long'] - input_new_df['Start_date']).values[
            0].days
        input_new_df['Open_cost'] = (input_new_df['Number_pos_long'].iloc[0] * input_new_df['Start_prime'].values[
            0]) * input_new_df['Multiplicator'].values[0]
        input_new_df['HV_200'] = hv
        input_new_df['Margin_start'] = input_new_df['Margin']
        input_new_df['DAYS_remaining'] = (input_new_df['Exp_date_short'].iloc[0] - datetime.datetime.now().date()).days

        print(input_new_df['DTE_short'].iloc[0])
        print(input_new_df['DTE_long'].iloc[0])
        min_dte = np.min([input_new_df['DTE_short'].iloc[0], input_new_df['DTE_long'].iloc[0]])
        min_closing_days_array = min_dte

        print('long_side_response')
        print(input_new_df, yahoo_price_df, input_new_df['IV_LONG'].iloc[0], input_new_df['DTE_long'].iloc[0], min_dte,
              min_closing_days_array, input_new_df['Start_underlying_long'].iloc[0],
              input_new_df['Strike_long'].iloc[0], input_new_df['Prime_long'].iloc[0])
        long_side_response = pd.DataFrame(
            solo_position_calc(input_new_df, yahoo_price_df, input_new_df['IV_LONG'].iloc[0],
                               input_new_df['DTE_long'].iloc[0], min_dte, min_closing_days_array,
                               input_new_df['Start_underlying_long'].iloc[0], input_new_df['Strike_long'].iloc[0],
                               input_new_df['Prime_long'].iloc[0]))
        long_side_response[['cvar', 'exp_return']] = long_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos_long'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        short_side_response = pd.DataFrame(
            solo_position_calc(input_new_df, yahoo_price_df, input_new_df['IV_SHORT'].iloc[0],
                               input_new_df['DTE_short'].iloc[0], min_dte, min_closing_days_array,
                               input_new_df['Start_underlying_short'].iloc[0], input_new_df['Strike_short'].iloc[0],
                               input_new_df['Prime_short'].iloc[0]))
        short_side_response[['cvar', 'exp_return']] = short_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos_short'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        print('long_side_response')
        print(long_side_response)
        print('short_side_response')
        print(short_side_response)

        input_new_df['POP_50'] = (short_side_response['pop'] + short_side_response['pop']) / 2
        input_new_df['Current_expected_return'] = (short_side_response['exp_return'] + short_side_response[
            'exp_return']) / 2
        input_new_df['CVAR'] = (short_side_response['cvar'] + short_side_response['cvar']) / 2

        print('input_new_df')
        print(input_new_df)

        # sigma_df_short = get_databento.get_bento_data(ticker, ticker_b, current_price, new_position_df['DTE_short'].values[0],
        #                                        new_position_df['Strike_short'].iloc[0], 'C', path_bento)
        # sigma_df_long = get_databento.get_bento_data(ticker, ticker_b, current_price, new_position_df['DTE_long'].values[0],
        #                                         new_position_df['Strike_long'].iloc[0], 'C', path_bento)
        # current_IV = (sigma_df_short['iv'].iloc[0] + sigma_df_long['iv'].iloc[0])/2 * 100

        input_new_df['Price_2s_down'] = start_price - (
                2 * start_price * hv * (math.sqrt(input_new_df['DTE_short'].iloc[0] / 365)))
        input_new_df['Max_profit'] = input_new_df['Open_cost'].abs()
        # max_risk = calculate_option_price('P', new_position_df['Price_2s_down'].values[0], new_position_df['Strike'].values[0], risk_rate, current_IV, new_position_df['DTE'].values[0])
        # new_position_df['Max_Risk'] = np.max([new_position_df['Strike'].iloc[0] * 0.1, (
        #             new_position_df['Price_2s_down'].iloc[0] - (
        #                 new_position_df['Price_2s_down'].iloc[0] - new_position_df['Strike'].iloc[0])) * 0.2]) * \
        #                               new_position_df['Open_cost'].abs() * 100
        # new_position_df['BE_higher'] = new_position_df['Strike'] + new_position_df['Start_prime']
        # new_position_df['RR_RATIO'] = new_position_df['Max_profit'] / new_position_df['Max_Risk']
        input_new_df['MP/DTE'] = input_new_df['Max_profit'] / input_new_df['DTE_short']
        input_new_df['Profit_Target'] = input_new_df['Max_profit'] * 0.5 - (input_new_df['Commission'])

        local_side = 'p'
        if input_new_df['Position_side'].iloc[0] == 'CALL':
            local_side = 'c'

        value_short, delta_short, gamma_short, theta_short, vega_short, rho_short = _gbs(local_side, input_new_df[
            'Start_underlying_short'].iloc[0], input_new_df['Strike_short'].iloc[0], input_new_df['DTE_short'].iloc[
                                                                                             0] / 365, 0.04, 0.04,
                                                                                         input_new_df['IV_SHORT'].iloc[
                                                                                             0] / 100)
        value_long, delta_long, gamma_long, theta_long, vega_long, rho_long = _gbs(local_side, input_new_df[
            'Start_underlying_long'].iloc[0], input_new_df['Strike_long'].iloc[0], input_new_df['DTE_long'].iloc[
                                                                                       0] / 365, 0.04, 0.04,
                                                                                   input_new_df['IV_LONG'].iloc[
                                                                                       0] / 100)

        delta_short_bs, gamma_short_bs, theta_short_bs, vega_short_bs = get_black_scholes_greeks(local_side, input_new_df[
            'Start_underlying_short'].iloc[0], input_new_df['Strike_short'].iloc[0], risk_rate, input_new_df[
                                                                                                     'IV_SHORT'].iloc[
                                                                                                     0] / 100,
                                                                                                 input_new_df[
                                                                                                     'DTE_short'].iloc[
                                                                                                     0])
        delta_long_bs, gamma_long_bs, theta_long_bs, vega_long_bs = get_black_scholes_greeks(local_side, input_new_df[
            'Start_underlying_long'].iloc[0], input_new_df['Strike_long'].iloc[0], risk_rate, input_new_df[
                                                                                                 'IV_LONG'].iloc[
                                                                                                 0] / 100, input_new_df[
                                                                                                 'DTE_long'].iloc[
                                                                                                 0])

        input_new_df['delta'] = ((delta_short * input_new_df['Number_pos_short'].iloc[0]) + (
                delta_long * input_new_df['Number_pos_long'].iloc[0])) / 2
        input_new_df['gamma'] = ((gamma_short * input_new_df['Number_pos_short'].iloc[0]) + (
                gamma_long * input_new_df['Number_pos_long'].iloc[0])) / 2
        input_new_df['theta'] = ((theta_short * input_new_df['Number_pos_short'].iloc[0]) + (
                theta_long * input_new_df['Number_pos_long'].iloc[0])) / 2
        input_new_df['vega'] = ((vega_short * input_new_df['Number_pos_short'].iloc[0]) + (
                vega_long * input_new_df['Number_pos_long'].iloc[0])) / 2
        input_new_df['rho'] = ((rho_short * input_new_df['Number_pos_short'].iloc[0]) + (
                rho_long * input_new_df['Number_pos_long'].iloc[0])) / 2

        input_new_df['theta_bs'] = ((theta_short_bs * input_new_df['Number_pos_short'].iloc[0]) + (
                theta_long_bs * input_new_df['Number_pos_long'].iloc[0])) / 2
        input_new_df['vega_bs'] = ((vega_short_bs * input_new_df['Number_pos_short'].iloc[0]) + (
                vega_long_bs * input_new_df['Number_pos_long'].iloc[0])) / 2

        input_new_df[['theta_bs', 'vega_bs']] = input_new_df[['theta_bs', 'vega_bs']] * input_new_df['Multiplicator'].iloc[0]
        #
        current_prime = input_new_df['Prime_long'] + (input_new_df['Prime_short'] * input_new_df['Number_pos_short'])
        print('current_prime', current_prime)
        # current_delta = sigma_df_long['delta'].iloc[0] + (sigma_df_short['delta'] * new_position_df['Number_pos_short'])
        # current_vega = sigma_df_long['vega'].iloc[0] + (sigma_df_short['vega'] * new_position_df['Number_pos_short'])
        # current_theta = sigma_df_long['theta'].iloc[0] + (sigma_df_short['theta'] * new_position_df['Number_pos_short'])
        # current_gamma = sigma_df_long['gamma'].iloc[0] + (sigma_df_short['gamma'] * new_position_df['Number_pos_short'])

        # new_position_df['Delta'] = current_delta
        # new_position_df['Vega'] = current_vega
        # new_position_df['Theta'] = current_theta
        # new_position_df['Gamma'] = current_gamma

        # print('pop_50 ', response['pop'], 'expected_profit ', response['exp_return'], 'cvar ', response['cvar'], )

        input_new_df['DAYS_remaining'] = (input_new_df['Exp_date_short'].iloc[0] - datetime.datetime.now().date()).days
        input_new_df['DAYS_elapsed_TDE'] = input_new_df['DTE_short'] - input_new_df['DAYS_remaining']
        input_new_df['%_days_elapsed'] = input_new_df['DAYS_elapsed_TDE'] / input_new_df['DTE_short']

        input_new_df['Prime_now'] = current_prime
        input_new_df['Cost_to_close_Market_cost'] = input_new_df['Open_cost'] - input_new_df['Commission']

        input_new_df['Current_Margin'] = input_new_df['Margin_start'].values[0]
        input_new_df['Current_PL'] = input_new_df['Cost_to_close_Market_cost'] - input_new_df['Open_cost']
        input_new_df['Current_ROI'] = input_new_df['Current_PL'] / input_new_df['Current_Margin']
        # new_position_df['Current_RR_ratio'] = (new_position_df['Max_profit'] - new_position_df['Current_PL']) / (
        #         new_position_df['CVAR'] + new_position_df['Current_PL'])
        input_new_df['PL_TDE'] = input_new_df['Current_PL'] / input_new_df['DAYS_elapsed_TDE']

        input_new_df[['%_days_elapsed', 'Current_ROI']] = input_new_df[['%_days_elapsed', 'Current_ROI']] * 100

        input_new_df = input_new_df.round(4)

        input_new_df.to_csv(f"{path}{ticker}_{input_new_df['Start_date'].values[0].strftime('%Y-%m-%d')}.csv",
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

        input_new_df['Start_prime'] = input_new_df['Prime_Call'] + input_new_df['Prime_Put']
        input_new_df['DTE'] = (input_new_df['Exp_date'] - input_new_df['Start_date']).values[0].days

        input_new_df['Open_cost'] = (input_new_df['Number_pos'].iloc[0] * input_new_df['Start_prime'].values[
            0]) * input_new_df['Multiplicator'].values[0]

        input_new_df['HV_200'] = hv
        input_new_df['Price_2s_down'] = start_price - (
                2 * start_price * hv * (math.sqrt(input_new_df['DTE'].iloc[0] / 365)))
        input_new_df['Max_profit'] = input_new_df['Open_cost'].abs()



        put_price = calculate_option_price('P', input_new_df['Price_2s_down'].values[0],
                                           input_new_df['Strike_Put'].values[0], risk_rate, input_new_df['IV_Put'].values[0],
                                           input_new_df['DTE'].values[0])
        call_price = calculate_option_price('C', input_new_df['Price_2s_down'].values[0],
                                            input_new_df['Strike_Call'].values[0], risk_rate, input_new_df['IV_Call'].values[0],
                                            input_new_df['DTE'].values[0])
        max_risk = np.max([put_price * abs(input_new_df['Number_pos'].values[0]) *
                           input_new_df['Multiplicator'].values[0] - input_new_df['Max_profit'].values[0],
                           call_price * abs(input_new_df['Number_pos'].values[0]) *
                           input_new_df['Multiplicator'].values[0] - input_new_df['Max_profit'].values[0]])
        input_new_df['Max_Risk'] = max_risk
        input_new_df['BE_lower'] = input_new_df['Strike_Put'] - input_new_df['Start_prime']
        input_new_df['RR_RATIO'] = input_new_df['Max_profit'] / input_new_df['Max_Risk']
        input_new_df['MP/DTE'] = input_new_df['Max_profit'] / input_new_df['DTE']
        input_new_df['Profit_Target'] = input_new_df['Max_profit'] * 0.5 - (input_new_df['Commission'])

        input_new_df['Number_pos_short'] = -1

        input_new_df['Position_side'] = 'PUT'
        put_side_response = pd.DataFrame(
            solo_position_calc(input_new_df, yahoo_price_df, input_new_df['IV_Put'].iloc[0],
                               input_new_df['DTE'].iloc[0], input_new_df['DTE'].values[0], input_new_df['DTE'].values[0],
                               input_new_df['Start_underlying'].iloc[0], input_new_df['Strike_Put'].iloc[0],
                               input_new_df['Prime_Put'].iloc[0]))
        put_side_response[['cvar', 'exp_return']] = put_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]

        input_new_df['Position_side'] = 'CALL'
        call_side_response = pd.DataFrame(
            solo_position_calc(input_new_df, yahoo_price_df, input_new_df['IV_Call'].iloc[0],
                               input_new_df['DTE'].iloc[0], input_new_df['DTE'].values[0], input_new_df['DTE'].values[0],
                               input_new_df['Start_underlying'].iloc[0], input_new_df['Strike_Call'].iloc[0],
                               input_new_df['Prime_Call'].iloc[0]))
        call_side_response[['cvar', 'exp_return']] = call_side_response[['cvar', 'exp_return']] * np.abs(
            input_new_df['Number_pos'].iloc[0]) * input_new_df['Multiplicator'].iloc[0]


        input_new_df['POP_50'] = (put_side_response['pop'] + call_side_response['pop']) / 2
        input_new_df['Current_expected_return'] = (put_side_response['exp_return'] + call_side_response[
            'exp_return']) / 2
        input_new_df['CVAR'] = (put_side_response['cvar'] + call_side_response['cvar']) / 2



        value_put, delta_put, gamma_put, theta_put, vega_put, rho_put = _gbs('p', input_new_df[
            'Start_underlying'].iloc[0], input_new_df['Strike_Put'].iloc[0], input_new_df['DTE'].iloc[
                                                                                             0] / 365, 0.04, 0.04,
                                                                                         input_new_df['IV_Put'].iloc[
                                                                                             0] / 100)
        value_call, delta_call, gamma_call, theta_call, vega_call, rho_call = _gbs('c', input_new_df[
            'Start_underlying'].iloc[0], input_new_df['Strike_Call'].iloc[0], input_new_df['DTE'].iloc[
                                                                                       0] / 365, 0.04, 0.04,
                                                                                   input_new_df['IV_Call'].iloc[
                                                                                       0] / 100)

        delta_put_bs, gamma_put_bs, theta_put_bs, vega_put_bs = get_black_scholes_greeks('p', input_new_df[
            'Start_underlying'].iloc[0], input_new_df['Strike_Put'].iloc[0], risk_rate, input_new_df[
                                                                                                     'IV_Put'].iloc[
                                                                                                     0] / 100,
                                                                                                 input_new_df[
                                                                                                     'DTE'].iloc[
                                                                                                     0])
        delta_call_bs, gamma_call_bs, theta_call_bs, vega_call_bs = get_black_scholes_greeks('c', input_new_df[
            'Start_underlying'].iloc[0], input_new_df['Strike_Call'].iloc[0], risk_rate, input_new_df[
                                                                                                 'IV_Call'].iloc[
                                                                                                 0] / 100, input_new_df[
                                                                                                 'DTE'].iloc[
                                                                                                 0])


        input_new_df['delta'] = ((delta_put * input_new_df['Number_pos'].iloc[0]) + (
                delta_call * input_new_df['Number_pos'].iloc[0])) / 2
        input_new_df['gamma'] = ((gamma_put * input_new_df['Number_pos'].iloc[0]) + (
                gamma_call * input_new_df['Number_pos'].iloc[0])) / 2
        input_new_df['theta'] = ((theta_put * input_new_df['Number_pos'].iloc[0]) + (
                theta_call * input_new_df['Number_pos'].iloc[0])) / 2
        input_new_df['vega'] = ((vega_put * input_new_df['Number_pos'].iloc[0]) + (
                vega_call * input_new_df['Number_pos'].iloc[0])) / 2
        input_new_df['rho'] = ((rho_put * input_new_df['Number_pos'].iloc[0]) + (
                rho_call * input_new_df['Number_pos'].iloc[0])) / 2

        input_new_df['theta_bs'] = ((theta_put_bs * input_new_df['Number_pos'].iloc[0]) + (
                theta_call_bs * input_new_df['Number_pos'].iloc[0])) / 2
        input_new_df['vega_bs'] = ((vega_put_bs * input_new_df['Number_pos'].iloc[0]) + (
                vega_call_bs * input_new_df['Number_pos'].iloc[0])) / 2

        input_new_df[['theta_bs', 'vega_bs']] = input_new_df[['theta_bs', 'vega_bs']] * input_new_df['Multiplicator'].iloc[0]
        #
        current_prime = (input_new_df['Prime_Call'] + input_new_df['Prime_Put']) * input_new_df['Number_pos_short']


        input_new_df['DAYS_remaining'] = (input_new_df['Exp_date'].iloc[0] - datetime.datetime.now().date()).days
        input_new_df['DAYS_elapsed_TDE'] = input_new_df['DTE'] - input_new_df['DAYS_remaining']
        input_new_df['%_days_elapsed'] = input_new_df['DAYS_elapsed_TDE'] / input_new_df['DTE']


        input_new_df['Prime_now'] = current_prime
        input_new_df['Cost_to_close_Market_cost'] = ((input_new_df['Number_pos'] * current_prime) * \
                                                        input_new_df['Multiplicator'].values[0]) - input_new_df[
                                                           'Commission']
        input_new_df['Current_Margin'] = input_new_df['Margin'].values[0]
        input_new_df['Current_PL'] = input_new_df['Cost_to_close_Market_cost'] - input_new_df['Open_cost']
        input_new_df['Current_ROI'] = input_new_df['Current_PL'] / input_new_df['Current_Margin']
        input_new_df['Current_RR_ratio'] = (input_new_df['Max_profit'] - input_new_df['Current_PL']) / (
                input_new_df['CVAR'] + input_new_df['Current_PL'])
        input_new_df['PL_TDE'] = input_new_df['Current_PL'] / input_new_df['DAYS_elapsed_TDE']
        # input_new_df['Leverage'] = (current_delta * 100 * current_price) / input_new_df['Current_Margin']
        # new_position_df['Margin_2S_Down'] = np.max([0.1 * new_position_df['Strike'], new_position_df['Price_2s_down'] * 0.2 - (
        #             new_position_df['Price_2s_down'] - new_position_df['Strike'])]) * new_position_df['Number_pos'].abs() * 100

        P_below = stats.norm.cdf(
            (np.log(input_new_df['BE_lower'].iloc[0] / current_price) / (
                    input_new_df['HV_200'].iloc[0] * math.sqrt(input_new_df['DAYS_remaining'].iloc[0] / 365))))

        input_new_df['Current_POP_lognormal'] = 1 - P_below

        # new_position_df['MC_2S_Down'] = calculate_option_price('P', new_position_df['Price_2s_down'].iloc[0],
        #                                                   new_position_df['Strike'].iloc[0], risk_rate,
        #                                                   new_position_df['HV_200'].iloc[0],
        #                                                   new_position_df['DAYS_remaining']) * new_position_df[
        #                                'Number_pos'].abs() * 100

        input_new_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio', ]] = \
            input_new_df[[
                '%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100

        input_new_df.to_csv(f"{path}{ticker}_{input_new_df['Start_date'].values[0].strftime('%Y-%m-%d')}.csv",
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


def update_postion_strangle(csv_position_df, pos_type, risk_rate, path_bento, input_update_df):
    print('update_postion')
    postion_df = pd.read_csv(csv_position_df)
    # postion_df = pd.concat([postion_df, input_update_df], axis=1)
    postion_df = pd.concat([input_update_df, postion_df], axis=1)
    postion_df = postion_df.loc[:, ~postion_df.columns.duplicated()]
    ticker = postion_df['Symbol'].iloc[0]
    ticker_b = postion_df['Symbol Bento'].iloc[0]
    print('tickerrrrrr', ticker)
    yahoo_price_df = yf.download(ticker)
    print('yahoo_price_df')
    print(yahoo_price_df)

    log_returns = np.log(yahoo_price_df["Close"] / yahoo_price_df["Close"].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    hv = log_returns.rolling(window=200).std() * np.sqrt(365)
    hv = hv.iloc[-1]
    current_price = yahoo_price_df['Close'].iloc[-1]
    postion_df['Cur_date'] = datetime.datetime.now().date()
    postion_df[['Exp_date', 'Cur_date']] = postion_df[['Exp_date', 'Cur_date']].apply(pd.to_datetime)
    print('88888')
    print(postion_df)

    postion_df['DTE_Current'] = (postion_df['Exp_date'] - postion_df['Cur_date']).dt.days

    postion_df['DAYS_remaining'] = (postion_df['Exp_date'].iloc[0] - datetime.datetime.now()).days
    postion_df['DAYS_elapsed_TDE'] = postion_df['DTE'] - postion_df['DAYS_remaining']
    postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE']
    print(postion_df['Number_pos'].values[0])

    local_side = 'p'
    if postion_df['Position_side'].iloc[0] == 'CALL':
        local_side = 'c'

    dte = (postion_df['Exp_date'].iloc[0] - datetime.datetime.now()).days


    value_put, delta_put, gamma_put, theta_put, vega_put, rho_put = _gbs('p', postion_df[
        'underlying_Current'].iloc[0], postion_df['Strike_Put'].iloc[0], dte / 365, 0.04, 0.04,
                                                                         postion_df['IV_Put_Current'].iloc[
                                                                             0] / 100)
    value_call, delta_call, gamma_call, theta_call, vega_call, rho_call = _gbs('c', postion_df[
        'underlying_Current'].iloc[0], postion_df['Strike_Call'].iloc[0], dte / 365, 0.04, 0.04,
                                                                               postion_df['IV_Call_Current'].iloc[
                                                                                   0] / 100)

    delta_put_bs, gamma_put_bs, theta_put_bs, vega_put_bs = get_black_scholes_greeks('p', postion_df[
        'underlying_Current'].iloc[0], postion_df['Strike_Put'].iloc[0], risk_rate, postion_df[
                                                                                         'IV_Put_Current'].iloc[
                                                                                         0] / 100,
                                                                                     dte)
    delta_call_bs, gamma_call_bs, theta_call_bs, vega_call_bs = get_black_scholes_greeks('c', postion_df[
        'underlying_Current'].iloc[0], postion_df['Strike_Call'].iloc[0], risk_rate, postion_df[
                                                                                             'IV_Call_Current'].iloc[
                                                                                             0] / 100, dte)

    postion_df['delta'] = ((delta_put * postion_df['Number_pos'].iloc[0]) + (
            delta_call * postion_df['Number_pos'].iloc[0])) / 2
    postion_df['gamma'] = ((gamma_put * postion_df['Number_pos'].iloc[0]) + (
            gamma_call * postion_df['Number_pos'].iloc[0])) / 2
    postion_df['theta'] = ((theta_put * postion_df['Number_pos'].iloc[0]) + (
            theta_call * postion_df['Number_pos'].iloc[0])) / 2
    postion_df['vega'] = ((vega_put * postion_df['Number_pos'].iloc[0]) + (
            vega_call * postion_df['Number_pos'].iloc[0])) / 2
    postion_df['rho'] = ((rho_put * postion_df['Number_pos'].iloc[0]) + (
            rho_call * postion_df['Number_pos'].iloc[0])) / 2

    postion_df['theta_bs'] = ((theta_put_bs * postion_df['Number_pos'].iloc[0]) + (
            theta_call_bs * postion_df['Number_pos'].iloc[0])) / 2
    postion_df['vega_bs'] = ((vega_put_bs * postion_df['Number_pos'].iloc[0]) + (
            vega_call_bs * postion_df['Number_pos'].iloc[0])) / 2

    postion_df[['theta_bs', 'vega_bs']] = postion_df[['theta_bs', 'vega_bs']] * postion_df['Multiplicator'].iloc[
        0]
    #

    current_delta = postion_df['delta'].iloc[0]

    print('shortPutF')
    print('current_price', current_price)
    print('risk_rate', risk_rate)
    print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])

    min_dte = np.min([postion_df['DTE'].iloc[0], postion_df['DTE'].iloc[0]])
    min_closing_days_array = min_dte

    print('long_side_response')
    postion_df['Number_pos_short'] = -1

    postion_df['Position_side'] = 'PUT'
    put_side_response = pd.DataFrame(
        solo_position_calc(postion_df, yahoo_price_df, postion_df['IV_Put_Current'].iloc[0],
                           postion_df['DTE_Current'].iloc[0], postion_df['DTE_Current'].values[0], postion_df['DTE_Current'].values[0],
                           postion_df['underlying_Current'].iloc[0], postion_df['Strike_Put'].iloc[0],
                           postion_df['Prime_put_Current'].iloc[0]))
    put_side_response[['cvar', 'exp_return']] = put_side_response[['cvar', 'exp_return']] * np.abs(
        postion_df['Number_pos'].iloc[0]) * postion_df['Multiplicator'].iloc[0]

    postion_df['Position_side'] = 'CALL'
    call_side_response = pd.DataFrame(
        solo_position_calc(postion_df, yahoo_price_df, postion_df['IV_Call_Current'].iloc[0],
                           postion_df['DTE_Current'].iloc[0], postion_df['DTE_Current'].values[0], postion_df['DTE_Current'].values[0],
                           postion_df['underlying_Current'].iloc[0], postion_df['Strike_Call'].iloc[0],
                           postion_df['Prime_call_Current'].iloc[0]))
    call_side_response[['cvar', 'exp_return']] = call_side_response[['cvar', 'exp_return']] * np.abs(
        postion_df['Number_pos'].iloc[0]) * postion_df['Multiplicator'].iloc[0]

    postion_df['POP_50'] = (put_side_response['pop'] + call_side_response['pop']) / 2
    postion_df['Current_expected_return'] = (put_side_response['exp_return'] + call_side_response[
        'exp_return']) / 2
    postion_df['CVAR'] = (put_side_response['cvar'] + call_side_response['cvar']) / 2



    print('postion_df')
    print(postion_df)

    current_prime = ((postion_df['Number_pos'].iloc[0] * postion_df['Prime_put_Current']) + (
            postion_df['Prime_call_Current'] * postion_df['Number_pos'].abs())) * postion_df['Multiplicator']

    postion_df['Cost_to_close_Market_cost'] = current_prime
    postion_df['Current_Margin'] = postion_df['Margin'].values[0]
    postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
    postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
    postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (
            postion_df['CVAR'] + postion_df['Current_PL'])
    postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
    postion_df['Leverage'] = (current_delta * 100 * current_price) / postion_df['Current_Margin']
    # postion_df['Margin_2S_Down'] = np.max([0.1 * postion_df['Strike'], postion_df['Price_2s_down'] * 0.2 - (postion_df['Price_2s_down'] - postion_df['Strike'])]) * postion_df['Number_pos'].abs() * 100

    print('shortPut')
    print('current_price', current_price)
    # print('current_IV', current_IV)
    print('risk_rate', risk_rate)
    print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])
    # print('DTE', postion_df['DTE'].values[0])
    # print('Strike', postion_df['Strike'].values[0])
    print('Start_prime', postion_df['Start_prime'].values[0])


    postion_df[['%_days_elapsed', 'Current_ROI', 'Current_RR_ratio', ]] = postion_df[['%_days_elapsed', 'Current_ROI',
                                                                                      'Current_RR_ratio']] * 100

    postion_df.to_csv(csv_position_df, index=False)
    print('postion_df DONE!!!!')

    current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                   'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                   'Current_expected_return',
                                   'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE', 'Leverage',
                                   'Max_profit']].T

    current_position['Values'] = current_position[0]
    current_position['Name'] = current_position.index.values.tolist()
    current_position = current_position[['Name', 'Values']]
    current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
    current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

    wight_df = pd.concat([current_position1, current_position2], axis=1, )

    wight_df.columns = ['N1', 'V1', 'N2', 'V2']
    pl, marg = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0]

    postion_df = postion_df.round(4)
    wight_df = wight_df.round(4)

    return wight_df, pl, marg

def update_postion_dia(csv_position_df, pos_type, risk_rate, path_bento, input_update_df):
    print('update_postion')
    postion_df = pd.read_csv(csv_position_df)
    # postion_df = pd.concat([postion_df, input_update_df], axis=1)
    postion_df = pd.concat([input_update_df, postion_df], axis=1)
    postion_df = postion_df.loc[:, ~postion_df.columns.duplicated()]
    ticker = postion_df['Symbol'].iloc[0]
    ticker_b = postion_df['Symbol Bento'].iloc[0]
    print('tickerrrrrr', ticker)
    yahoo_price_df = yf.download(ticker)
    print('yahoo_price_df')
    print(yahoo_price_df)
    print(postion_df['Exp_date_short'])
    log_returns = np.log(yahoo_price_df["Close"] / yahoo_price_df["Close"].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    hv = log_returns.rolling(window=200).std() * np.sqrt(365)
    hv = hv.iloc[-1]
    current_price = yahoo_price_df['Close'].iloc[-1]
    postion_df['Cur_date'] = datetime.datetime.now().date()

    postion_df[['Exp_date_short', 'Exp_date_long', 'Cur_date']] = postion_df[
        ['Exp_date_short', 'Exp_date_long', 'Cur_date']].apply(pd.to_datetime)

    postion_df['DTE_short_Current'] = (postion_df['Exp_date_short'] - postion_df['Cur_date']).dt.days
    postion_df['DTE_long_Current'] = (postion_df['Exp_date_long'] - postion_df['Cur_date']).dt.days
    print('88888')
    print(postion_df['DTE_short'])
    postion_df['DAYS_remaining'] = (postion_df['Exp_date_short'].iloc[0] - datetime.datetime.now()).days
    postion_df['DAYS_elapsed_TDE'] = postion_df['DTE_short'] - postion_df['DAYS_remaining']
    postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE_short']
    print(postion_df['Number_pos_long'].values[0])

    local_side = 'p'
    if postion_df['Position_side'].iloc[0] == 'CALL':
        local_side = 'c'

    dte_short = (postion_df['Exp_date_short'].iloc[0] - datetime.datetime.now()).days
    dte_long = (postion_df['Exp_date_long'].iloc[0] - datetime.datetime.now()).days

    value_short, delta_short, gamma_short, theta_short, vega_short, rho_short = _gbs(local_side, postion_df[
        'underlying_short_Current'].iloc[0], postion_df['Strike_short'].iloc[0], dte_short / 365, 0.04, 0.04,
                                                                                     postion_df[
                                                                                         'IV_SHORT_Current'].iloc[
                                                                                         0] / 100)
    value_long, delta_long, gamma_long, theta_long, vega_long, rho_long = _gbs(local_side, postion_df[
        'underlying_long_Current'].iloc[0], postion_df['Strike_long'].iloc[0], dte_long / 365, 0.04, 0.04,
                                                                               postion_df['IV_LONG_Current'].iloc[
                                                                                   0] / 100)

    delta_short_bs, gamma_short_bs, theta_short_bs, vega_short_bs = get_black_scholes_greeks(local_side, postion_df[
        'underlying_short_Current'].iloc[0], postion_df['Strike_short'].iloc[0], risk_rate, postion_df[
                                                                                                 'IV_SHORT_Current'].iloc[
                                                                                                 0] / 100, dte_short)
    delta_long_bs, gamma_long_bs, theta_long_bs, vega_long_bs = get_black_scholes_greeks(local_side, postion_df[
        'underlying_long_Current'].iloc[0], postion_df['Strike_long'].iloc[0], risk_rate, postion_df[
                                                                                             'IV_LONG_Current'].iloc[
                                                                                             0] / 100, dte_long)

    print('delta_short', delta_short)
    print('gamma_short', gamma_short)
    print('theta_short', theta_short)
    print('vega_short', vega_short)
    print('delta_long', delta_long)
    print('gamma_long', gamma_long)
    print('theta_long', theta_long)
    print('vega_long', vega_long)

    postion_df['delta'] = ((delta_short * postion_df['Number_pos_short'].iloc[0]) + (
            delta_long * postion_df['Number_pos_long'].iloc[0])) / 2
    postion_df['gamma'] = ((gamma_short * postion_df['Number_pos_short'].iloc[0]) + (
            gamma_long * postion_df['Number_pos_long'].iloc[0])) / 2
    postion_df['theta'] = ((theta_short * postion_df['Number_pos_short'].iloc[0]) + (
            theta_long * postion_df['Number_pos_long'].iloc[0])) / 2
    postion_df['vega'] = ((vega_short * postion_df['Number_pos_short'].iloc[0]) + (
            vega_long * postion_df['Number_pos_long'].iloc[0])) / 2
    postion_df['rho'] = ((rho_short * postion_df['Number_pos_short'].iloc[0]) + (
            rho_long * postion_df['Number_pos_long'].iloc[0])) / 2
    #

    postion_df['theta_bs'] = ((theta_short_bs * postion_df['Number_pos_short'].iloc[0]) + (
            theta_long_bs * postion_df['Number_pos_long'].iloc[0])) / 2
    postion_df['vega_bs'] = ((vega_short_bs * postion_df['Number_pos_short'].iloc[0]) + (
            vega_long_bs * postion_df['Number_pos_long'].iloc[0])) / 2

    postion_df[['theta_bs', 'vega_bs']] = postion_df[['theta_bs', 'vega_bs']] * postion_df['Multiplicator'].iloc[0]

    current_delta = postion_df['delta'].iloc[0]

    print('shortPutF')
    print('current_price', current_price)
    print('risk_rate', risk_rate)
    print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])

    print(postion_df['DTE_short'].iloc[0])
    print(postion_df['DTE_long'].iloc[0])
    min_dte = np.min([postion_df['DTE_short'].iloc[0], postion_df['DTE_long'].iloc[0]])
    min_closing_days_array = min_dte

    print('long_side_response')
    print(postion_df, yahoo_price_df, postion_df['IV_LONG_Current'].iloc[0], postion_df['DTE_long'].iloc[0], min_dte,
          min_closing_days_array, postion_df['Start_underlying_long'].iloc[0], postion_df['Strike_long'].iloc[0],
          postion_df['Prime_long'].iloc[0])
    long_side_response = pd.DataFrame(
        solo_position_calc(postion_df, yahoo_price_df, postion_df['IV_LONG_Current'].iloc[0],
                           postion_df['DTE_long_Current'].iloc[0], min_dte,
                           min_closing_days_array,
                           postion_df['underlying_long_Current'].iloc[0],
                           postion_df['Strike_long'].iloc[0],
                           postion_df['Prime_long_Current'].iloc[0]))
    long_side_response[['cvar', 'exp_return']] = long_side_response[['cvar', 'exp_return']] * np.abs(
        postion_df['Number_pos_long'].iloc[0]) * postion_df['Multiplicator'].iloc[0]

    short_side_response = pd.DataFrame(
        solo_position_calc(postion_df, yahoo_price_df, postion_df['IV_SHORT_Current'].iloc[0],
                           postion_df['DTE_short'].iloc[0], min_dte, min_closing_days_array,
                           postion_df['underlying_short_Current'].iloc[0], postion_df['Strike_short'].iloc[0],
                           postion_df['Prime_short_Current'].iloc[0]))
    short_side_response[['cvar', 'exp_return']] = short_side_response[['cvar', 'exp_return']] * np.abs(
        postion_df['Number_pos_short'].iloc[0]) * postion_df['Multiplicator'].iloc[0]

    print('long_side_response')
    print(long_side_response)
    print('short_side_response')
    print(short_side_response)

    postion_df['POP_50'] = (short_side_response['pop'] + short_side_response['pop']) / 2
    postion_df['Current_expected_return'] = (short_side_response['exp_return'] + short_side_response[
        'exp_return']) / 2
    postion_df['CVAR'] = (short_side_response['cvar'] + short_side_response['cvar']) / 2

    print('postion_df')
    print(postion_df)

    current_prime = ((postion_df['Number_pos_long'].iloc[0] * postion_df['Prime_long_Current']) - (
            postion_df['Prime_short_Current'] * postion_df['Number_pos_short'].abs())) * postion_df['Multiplicator']

    postion_df['Cost_to_close_Market_cost'] = current_prime
    postion_df['Current_Margin'] = postion_df['Margin'].values[0]
    postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
    postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
    postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (
            postion_df['CVAR'] + postion_df['Current_PL'])
    postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
    postion_df['Leverage'] = (current_delta * 100 * current_price) / postion_df['Current_Margin']
    # postion_df['Margin_2S_Down'] = np.max([0.1 * postion_df['Strike'], postion_df['Price_2s_down'] * 0.2 - (postion_df['Price_2s_down'] - postion_df['Strike'])]) * postion_df['Number_pos'].abs() * 100

    print('shortPut')
    print('current_price', current_price)
    # print('current_IV', current_IV)
    print('risk_rate', risk_rate)
    print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])
    # print('DTE', postion_df['DTE'].values[0])
    # print('Strike', postion_df['Strike'].values[0])
    print('Start_prime', postion_df['Start_prime'].values[0])


    postion_df[['%_days_elapsed', 'Current_ROI', 'Current_RR_ratio', ]] = postion_df[['%_days_elapsed', 'Current_ROI',
                                                                                      'Current_RR_ratio']] * 100

    postion_df.to_csv(csv_position_df, index=False)
    print('postion_df DONE!!!!')

    current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                   'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                   'Current_expected_return',
                                   'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE', 'Leverage',
                                   'Max_profit']].T

    current_position['Values'] = current_position[0]
    current_position['Name'] = current_position.index.values.tolist()
    current_position = current_position[['Name', 'Values']]
    current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
    current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

    wight_df = pd.concat([current_position1, current_position2], axis=1, )

    wight_df.columns = ['N1', 'V1', 'N2', 'V2']
    pl, marg = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0]

    postion_df = postion_df.round(4)
    wight_df = wight_df.round(4)

    return wight_df, pl, marg


def update_postion_cover(csv_position_df, pos_type, risk_rate, path_bento, input_update_df):
    postion_df = pd.read_csv(csv_position_df)
    # postion_df = pd.concat([postion_df, input_update_df], axis=1)
    postion_df = pd.concat([input_update_df, postion_df], axis=1)
    postion_df = postion_df.loc[:, ~postion_df.columns.duplicated()]
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
    postion_df['Cur_date'] = datetime.datetime.now().date()
    postion_df[['Exp_date', 'Cur_date']] = postion_df[['Exp_date', 'Cur_date']].apply(pd.to_datetime)
    postion_df['DTE'] = (postion_df['Exp_date'] - postion_df['Cur_date']).dt.days
    print('88888')
    postion_df['DAYS_remaining'] = (postion_df['Exp_date'].iloc[0] - datetime.datetime.now()).days
    postion_df['DAYS_elapsed_TDE'] = postion_df['DTE'] - postion_df['DAYS_remaining']
    postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE']
    print(postion_df['Number_pos'].values[0])

    local_side = 'p'
    if postion_df['Position_side'].iloc[0] == 'CALL':
        local_side = 'c'

    print('local_side', local_side)
    print('Underlying_Current', postion_df['Underlying_Current'].iloc[0])
    print('Strike', postion_df['Strike'].iloc[0])
    print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0] / 365)
    print('IV_Current', postion_df['IV_Current'].iloc[0] / 100)

    value_opt, delta_opt, gamma_opt, theta_opt, vega_opt, rho_opt = _gbs(local_side, postion_df[
        'Underlying_Current'].iloc[0], postion_df['Strike'].iloc[0], postion_df['DAYS_remaining'].iloc[
                                                                             0] / 365, 0.04, 0.04,
                                                                         postion_df['IV_Current'].iloc[
                                                                             0] / 100)

    delta_opt_bs, gamma_opt_bs, theta_opt_bs, vega_opt_bs = get_black_scholes_greeks(local_side, postion_df[
        'Underlying_Current'].iloc[0], postion_df['Strike'].iloc[0], risk_rate, postion_df[
                                                                                                 'IV_Current'].iloc[
                                                                                                 0] / 100, postion_df['DAYS_remaining'].iloc[
                                                                             0])


    print('theta_opt', theta_opt)
    print('vega_opt', vega_opt)
    print('gamma_opt', gamma_opt)
    print('Number_pos', postion_df['Number_pos'].iloc[0])

    postion_df['delta'] = ((delta_opt * postion_df['Number_pos'].iloc[0]) - 1)
    postion_df['gamma'] = (gamma_opt * postion_df['Number_pos'].iloc[0])
    postion_df['theta'] = (theta_opt * postion_df['Number_pos'].iloc[0])
    postion_df['vega'] = (vega_opt * postion_df['Number_pos'].iloc[0])
    postion_df['rho'] = (rho_opt * postion_df['Number_pos'].iloc[0])
    #

    postion_df['theta_bs'] = (theta_opt_bs * postion_df['Number_pos'].iloc[0])
    postion_df['vega_bs'] = (vega_opt_bs * postion_df['Number_pos'].iloc[0])

    postion_df[['theta_bs', 'vega_bs']] = postion_df[['theta_bs', 'vega_bs']] * postion_df['Multiplicator'].iloc[0]

    current_delta = postion_df['delta'].iloc[0]

    print('shortPutF')
    print('current_price', current_price)
    print('risk_rate', risk_rate)
    print('DAYS_remaining', postion_df['DAYS_remaining'].iloc[0])

    min_dte = postion_df['DTE'].iloc[0]
    min_closing_days_array = min_dte

    opt_side_response = pd.DataFrame(
        solo_position_calc_covered(postion_df, yahoo_price_df, postion_df['IV_Current'].iloc[0],
                                   postion_df['DTE'].iloc[0], min_dte, min_dte,
                                   postion_df['Underlying_Current'].iloc[0], postion_df['Strike'].iloc[0],
                                   postion_df['Prime_Current'].iloc[0]))
    opt_side_response[['cvar', 'exp_return']] = opt_side_response[['cvar', 'exp_return']] * np.abs(
        postion_df['Number_pos'].iloc[0]) * postion_df['Multiplicator'].iloc[0]

    postion_df['Position_side'] = 'STOCK'

    stock_side_response = pd.DataFrame(
        solo_position_calc_covered(postion_df, yahoo_price_df, postion_df['IV_Current'].iloc[0],
                                   postion_df['DTE'].iloc[0], min_dte, min_dte,
                                   postion_df['Underlying_stock_Current'].iloc[0], postion_df['Strike'].iloc[0],
                                   postion_df['Prime_Current'].iloc[0]))
    stock_side_response[['cvar', 'exp_return']] = stock_side_response[['cvar', 'exp_return']] * np.abs(
        postion_df['Number_pos'].iloc[0]) * postion_df['Multiplicator'].iloc[0]

    postion_df['POP_50'] = (stock_side_response['pop'] + opt_side_response['pop']) / 2
    postion_df['Current_expected_return'] = (stock_side_response['exp_return'] + opt_side_response[
        'exp_return']) / 2
    postion_df['CVAR'] = (stock_side_response['cvar'] + opt_side_response['cvar']) / 2

    print('postion_df')
    print(postion_df)

    current_prime = (postion_df['Number_pos'].iloc[0] * postion_df['Prime_Current']) * postion_df['Multiplicator']

    postion_df['Cost_to_close_Market_cost'] = current_prime
    postion_df['Current_Margin'] = postion_df['Margin'].values[0]
    postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
    postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
    postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (
            postion_df['CVAR'] + postion_df['Current_PL'])
    postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
    postion_df['Leverage'] = (current_delta * 100 * current_price) / postion_df['Current_Margin']
    # postion_df['Margin_2S_Down'] = np.max([0.1 * postion_df['Strike'], postion_df['Price_2s_down'] * 0.2 - (postion_df['Price_2s_down'] - postion_df['Strike'])]) * postion_df['Number_pos'].abs() * 100

    postion_df[['%_days_elapsed', 'Current_ROI', 'Current_RR_ratio', ]] = postion_df[['%_days_elapsed', 'Current_ROI',
                                                                                      'Current_RR_ratio']] * 100
    postion_df = postion_df.round(4)

    postion_df.to_csv(csv_position_df, index=False)
    print('postion_df DONE!!!!')

    current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                   'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                   'Current_expected_return',
                                   'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE', 'Leverage',
                                   'Max_profit']].T

    current_position['Values'] = current_position[0]
    current_position['Name'] = current_position.index.values.tolist()
    current_position = current_position[['Name', 'Values']]
    current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
    current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

    wight_df = pd.concat([current_position1, current_position2], axis=1, )

    wight_df.columns = ['N1', 'V1', 'N2', 'V2']
    pl, marg = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0]

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


def return_postion(csv_position_df, pos_type, risk_rate):
    postion_df = pd.read_csv(csv_position_df)
    print('postion_df')
    print(postion_df)
    print(postion_df.columns.tolist())
    KEY = credential.password['marketdata_KEY']
    ticker = postion_df['Symbol'].iloc[0]
    yahoo_price_df = yf.download(ticker)

    if pos_type == 'F. Strangle':
        current_position = postion_df[
            ['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed', 'Cost_to_close_Market_cost',
             'Current_Margin', 'Current_PL', 'Current_expected_return', 'CVAR',
             'POP_50', 'Current_ROI', 'PL_TDE', 'Max_profit']].T
        # 'Current_RR_ratio',
        # current_position = [['Symbol', 'Start_date', 'Exp_date', 'Strike', 'Number_pos', 'Start_prime', 'Multiplicator', 'Start_price', 'Dividend', 'Commission', 'Margin_start', 'DTE', 'Open_cost', 'Start_delta', 'HV_200', 'Price_2s_down', 'Max_profit', 'Max_Risk', 'BE_lower', 'RR_RATIO', 'MP/DTE', 'Profit_Target', 'Expected_Return', 'DTE_Target', 'POP_Monte_start_50', 'Plan_ROC ', 'ROC_DAY_target']]
        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )
        wight_df.columns = ['N1', 'V1', 'N2', 'V2']

        greeks_position = postion_df[['delta', 'vega', 'theta', 'gamma', 'vega_bs', 'theta_bs']].T

        greeks_position['Values'] = greeks_position[0]
        greeks_position['Name'] = greeks_position.index.values.tolist()
        greeks_position = greeks_position[['Name', 'Values']]
        greeks_position1 = greeks_position[int(len(greeks_position) / 2):].reset_index(drop=True)
        greeks_position2 = greeks_position[:int(len(greeks_position) / 2)].reset_index(drop=True)

        greeks_df = pd.concat([greeks_position1, greeks_position2], axis=1, )
        greeks_df.columns = ['N1', 'V1', 'N2', 'V2']
        greeks_df = greeks_df.round(4)
        wight_df = wight_df.round(2)

        pl, marg = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0]

        return wight_df, greeks_df, pl, marg,  # greeks_df,

    if pos_type == 'F. Diagonal':
        current_position = postion_df[
            ['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed', 'Cost_to_close_Market_cost',
             'Current_Margin', 'Current_PL', 'Current_expected_return', 'CVAR',
             'POP_50', 'Current_ROI', 'PL_TDE', 'Max_profit']].T
        # 'Current_RR_ratio',
        # current_position = [['Symbol', 'Start_date', 'Exp_date', 'Strike', 'Number_pos', 'Start_prime', 'Multiplicator', 'Start_price', 'Dividend', 'Commission', 'Margin_start', 'DTE', 'Open_cost', 'Start_delta', 'HV_200', 'Price_2s_down', 'Max_profit', 'Max_Risk', 'BE_lower', 'RR_RATIO', 'MP/DTE', 'Profit_Target', 'Expected_Return', 'DTE_Target', 'POP_Monte_start_50', 'Plan_ROC ', 'ROC_DAY_target']]
        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )
        wight_df.columns = ['N1', 'V1', 'N2', 'V2']

        greeks_position = postion_df[['delta', 'vega', 'theta', 'gamma', 'vega_bs', 'theta_bs']].T

        greeks_position['Values'] = greeks_position[0]
        greeks_position['Name'] = greeks_position.index.values.tolist()
        greeks_position = greeks_position[['Name', 'Values']]
        greeks_position1 = greeks_position[int(len(greeks_position) / 2):].reset_index(drop=True)
        greeks_position2 = greeks_position[:int(len(greeks_position) / 2)].reset_index(drop=True)

        greeks_df = pd.concat([greeks_position1, greeks_position2], axis=1, )
        greeks_df.columns = ['N1', 'V1', 'N2', 'V2']
        greeks_df = greeks_df.round(4)
        wight_df = wight_df.round(2)

        pl, marg = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0]

        return wight_df, greeks_df, pl, marg,  # greeks_df,

    if pos_type == 'F. Put':
        current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                       'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                       'Current_expected_return', 'CVAR',
                                       'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE',
                                       'Max_profit']].T
        # current_position = [['Symbol', 'Start_date', 'Exp_date', 'Strike', 'Number_pos', 'Start_prime', 'Multiplicator', 'Start_price', 'Dividend', 'Commission', 'Margin_start', 'DTE', 'Open_cost', 'Start_delta', 'HV_200', 'Price_2s_down', 'Max_profit', 'Max_Risk', 'BE_lower', 'RR_RATIO', 'MP/DTE', 'Profit_Target', 'Expected_Return', 'DTE_Target', 'POP_Monte_start_50', 'Plan_ROC ', 'ROC_DAY_target']]
        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )
        wight_df.columns = ['N1', 'V1', 'N2', 'V2']

        greeks_position = postion_df[['Delta', 'Vega', 'Theta', 'Gamma']].T

        greeks_position['Values'] = greeks_position[0]
        greeks_position['Name'] = greeks_position.index.values.tolist()
        greeks_position = greeks_position[['Name', 'Values']]
        greeks_position1 = greeks_position[int(len(greeks_position) / 2):].reset_index(drop=True)
        greeks_position2 = greeks_position[:int(len(greeks_position) / 2)].reset_index(drop=True)

        greeks_df = pd.concat([greeks_position1, greeks_position2], axis=1, )
        greeks_df.columns = ['N1', 'V1', 'N2', 'V2']
        greeks_df = greeks_df.round(3)

        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0], \
            postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, greeks_df, pl, marg, pop_log

    if pos_type == 'F. Call':
        current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                       'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                       'Current_expected_return', 'CVAR',
                                       'Current_POP_lognormal', 'POP_50', 'Current_ROI', 'Current_RR_ratio',
                                       'PL_TDE', 'Max_profit']].T
        # current_position = [['Symbol', 'Start_date', 'Exp_date', 'Strike', 'Number_pos', 'Start_prime', 'Multiplicator', 'Start_price', 'Dividend', 'Commission', 'Margin_start', 'DTE', 'Open_cost', 'Start_delta', 'HV_200', 'Price_2s_down', 'Max_profit', 'Max_Risk', 'BE_lower', 'RR_RATIO', 'MP/DTE', 'Profit_Target', 'Expected_Return', 'DTE_Target', 'POP_Monte_start_50', 'Plan_ROC ', 'ROC_DAY_target']]
        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )
        wight_df.columns = ['N1', 'V1', 'N2', 'V2']

        greeks_position = postion_df[['Delta', 'Vega', 'Theta', 'Gamma']].T

        greeks_position['Values'] = greeks_position[0]
        greeks_position['Name'] = greeks_position.index.values.tolist()
        greeks_position = greeks_position[['Name', 'Values']]
        greeks_position1 = greeks_position[int(len(greeks_position) / 2):].reset_index(drop=True)
        greeks_position2 = greeks_position[:int(len(greeks_position) / 2)].reset_index(drop=True)

        greeks_df = pd.concat([greeks_position1, greeks_position2], axis=1, )
        greeks_df.columns = ['N1', 'V1', 'N2', 'V2']

        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0], \
            postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, greeks_df, pl, marg, pop_log

    if pos_type == 'F. Strangle':
        current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                                       'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL',
                                       'Current_expected_return', 'CVAR',
                                       'Current_POP_lognormal', 'POP_50', 'Current_ROI', 'Current_RR_ratio',
                                       'PL_TDE', 'Max_profit']].T
        # current_position = [['Symbol', 'Start_date', 'Exp_date', 'Strike', 'Number_pos', 'Start_prime', 'Multiplicator', 'Start_price', 'Dividend', 'Commission', 'Margin_start', 'DTE', 'Open_cost', 'Start_delta', 'HV_200', 'Price_2s_down', 'Max_profit', 'Max_Risk', 'BE_lower', 'RR_RATIO', 'MP/DTE', 'Profit_Target', 'Expected_Return', 'DTE_Target', 'POP_Monte_start_50', 'Plan_ROC ', 'ROC_DAY_target']]
        current_position['Values'] = current_position[0]
        current_position['Name'] = current_position.index.values.tolist()
        current_position = current_position[['Name', 'Values']]
        current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
        current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

        wight_df = pd.concat([current_position1, current_position2], axis=1, )
        wight_df.columns = ['N1', 'V1', 'N2', 'V2']

        greeks_position = postion_df[['Delta', 'Vega', 'Theta', 'Gamma']].T

        greeks_position['Values'] = greeks_position[0]
        greeks_position['Name'] = greeks_position.index.values.tolist()
        greeks_position = greeks_position[['Name', 'Values']]
        greeks_position1 = greeks_position[int(len(greeks_position) / 2):].reset_index(drop=True)
        greeks_position2 = greeks_position[:int(len(greeks_position) / 2)].reset_index(drop=True)

        greeks_df = pd.concat([greeks_position1, greeks_position2], axis=1, )
        greeks_df.columns = ['N1', 'V1', 'N2', 'V2']

        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0], \
            postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, greeks_df, pl, marg, pop_log
