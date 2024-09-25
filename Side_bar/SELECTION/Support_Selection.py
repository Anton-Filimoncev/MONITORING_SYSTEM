import pandas as pd
import numpy as np
import scipy.stats as stats
import time
import datetime
import yfinance as yf
import math
import requests
import credential
import tqdm
import mibian
from scipy.stats import norm
from pathlib import Path
from .popoption.ShortPut import shortPut
from .popoption.ShortCall import shortCall
from .popoption.ShortStrangle import shortStrangle
from .popoption.PutCalendar import putCalendar
from .popoption.CallCalendar import callCalendar
from .popoption.PutCalendar_template import putCalendar_template
from .popoption.CallCalendar_template import callCalendar_template
from .popoption.RiskReversal import riskReversal
from .popoption.LongPut import longPut
from .popoption.LongCall import longCall
from .popoption.Fut_DA_BUT import fut_DA_BUT
from .popoption.FutRation_1_1_2 import futRatio_1_1_2
from .Relative_PRICE import *
from sklearn.linear_model import LinearRegression
# from optlib.optlib.gbs import *


def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))



def implied_volatility(row, side, type_contract):
    P = row['bid']
    S = row['underlyingPrice']
    E = row['strike']
    T = row['days_to_exp']/365
    r = 0.04
    sigma = 0.01

    if type_contract == 'F':
        try:
            #    option_type = "p" or "c"
            #    fs          = price of underlying
            #    x           = strike
            #    t           = time to expiration
            #    v           = implied volatility
            #    r           = risk free rate
            #    q           = dividend payment
            #    b           = cost of carry
            if side == 'C':
                return amer_implied_vol_76("c", S, E, T, r, P)
            if side == 'P':
                return amer_implied_vol_76("p", S, E, T, r, P)
        except:
            return 0

    else:
        while sigma < 1:
            try:
                d_1 = float(float((math.log(S/E)+(r+(sigma**2)/2)*T))/float((sigma*(math.sqrt(T)))))
                d_2 = float(float((math.log(S/E)+(r-(sigma**2)/2)*T))/float((sigma*(math.sqrt(T)))))

                if side == 'C':
                    P_implied = float(S*norm.cdf(d_1) - E*math.exp(-r*T)*norm.cdf(d_2))

                if side == 'P':
                    P_implied = float(norm.cdf(-d_2)*E*math.exp(-r*T) - norm.cdf(-d_1)*S)

                if P-(P_implied) < 0.001:
                    return sigma * 100

                sigma +=0.001
            except:
                return 0

        return 0

def greek_calc(input_df, side, type_contract):
    delta_list = []
    iv_list = []

    tqdm_params = {
        'total': len(input_df),
        'miniters': 1,
        'unit': 'it',
        'unit_scale': True,
        'unit_divisor': 1024,
    }

    with tqdm.tqdm(**tqdm_params) as pb:
        for number_row, row in input_df.iterrows():
            current_price = row['underlyingPrice']
            strike = row['strike']
            days_to_exp = row['days_to_exp']
            option_price = row['bid']

            # print('volatility calc...')
            if side == 'C':
                # c = mibian.BS([current_price, strike, 1.5, days_to_exp], callPrice=option_price)
                # volatility = c.impliedVolatility
                # print('volatility1', volatility)
                volatility = implied_volatility(row, side, type_contract)


            elif side == 'P':
                # c = mibian.BS([current_price, strike, 1.5, days_to_exp], putPrice=option_price)
                # volatility = c.impliedVolatility
                # print('volatility1', volatility)

                volatility = implied_volatility(row, side, type_contract)


            iv_list.append(volatility)
            # BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], volatility=x, callPrice=y, putPrice=z)
            # print('delta calc...')
            c = mibian.BS([current_price, strike, 1, days_to_exp], volatility=volatility)

            if side == 'C':
                delta_list.append(c.callDelta)

            elif side == 'P':
                delta_list.append(c.putDelta)

            pb.update(1)

    input_df['delta'] = delta_list
    input_df['iv'] = iv_list
    return input_df

def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


def volatility_calc(stock_yahoo, period):
    # ======= HIST volatility calculated ===========
    try:
        # TRADING_DAYS = 252
        returns = np.log(stock_yahoo / stock_yahoo.shift(1))
        returns.fillna(0, inplace=True)
        volatility = returns.rolling(window=period).std() * np.sqrt(365)
        hist_vol = volatility.iloc[-1]
    except:
        hist_vol = 0

    return hist_vol


def get_exp_move(tick, stock_yahoo, period):
    print('---------------------------')
    print('------------- Getting HV --------------')
    print('---------------------------')

    stock_yahoo_solo = stock_yahoo['Close']
    hist_vol = volatility_calc(stock_yahoo_solo, period)

    return hist_vol


def get_yahoo_price(ticker):
    yahoo_data = yf.download(ticker, progress=False)['2018-01-01':]
    return yahoo_data


def get_strangle(tick, rate, percentage_array, days_to_expiration, closing_days_array, quotes):

    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    sum_df = pd.DataFrame()
    hv = get_exp_move(tick, yahoo_stock, 244)

    if '-' in tick or '=' in tick:
        quotes_put = quotes[quotes['instrument_class'] == 'P'].reset_index(drop=True)
        delta_5_put = nearest_equal_abs(quotes_put['delta'].dropna().abs().values.tolist(), 0.05)
        delta_10_put = nearest_equal_abs(quotes_put['delta'].dropna().abs().values.tolist(), 0.1)
        delta_15_put = nearest_equal_abs(quotes_put['delta'].dropna().abs().values.tolist(), 0.15)

        print(quotes)

        quotes_call = quotes[quotes['instrument_class'] == 'C'].reset_index(drop=True)
        delta_5_call = nearest_equal_abs(quotes_call['delta'].dropna().abs().values.tolist(), 0.05)
        delta_10_call = nearest_equal_abs(quotes_call['delta'].dropna().abs().values.tolist(), 0.1)
        delta_15_call = nearest_equal_abs(quotes_call['delta'].dropna().abs().values.tolist(), 0.15)

        print('quotes_call greek_calc')
        print(quotes_call)

        finish_put, finish_call = get_rel_price_76(quotes_put, quotes_call, underlying, days_to_expiration)
        print('finish_put')
        print(finish_call)
        print(finish_call.index.tolist())
        max_put_skew_strike = finish_put[finish_put == finish_put.max()].index.tolist()[0][0]
        print('max_put_skew_strike',max_put_skew_strike)
        print('finish_call')
        print(finish_call)

        skew_delta_put = quotes_put[quotes_put['strike'] == max_put_skew_strike].reset_index(drop=True)['delta'].abs()[0]
        skew_delta_call = nearest_equal_abs(quotes_call['delta'].dropna().abs().values.tolist(), skew_delta_put)
        print('skew_delta_put', skew_delta_put)
        print('skew_delta_call', skew_delta_call)


    else:
        quotes_put = quotes[quotes['side'] == 'put'].reset_index(drop=True)
        delta_5_put = nearest_equal_abs(quotes_put['delta'].dropna().abs().values.tolist(), 0.05)
        delta_10_put = nearest_equal_abs(quotes_put['delta'].dropna().abs().values.tolist(), 0.1)
        delta_15_put = nearest_equal_abs(quotes_put['delta'].dropna().abs().values.tolist(), 0.15)

        quotes_call = quotes[quotes['side'] == 'call'].reset_index(drop=True)
        delta_5_call = nearest_equal_abs(quotes_call['delta'].dropna().abs().values.tolist(), 0.05)
        delta_10_call = nearest_equal_abs(quotes_call['delta'].dropna().abs().values.tolist(), 0.1)
        delta_15_call = nearest_equal_abs(quotes_call['delta'].dropna().abs().values.tolist(), 0.15)

        finish_put, finish_call = get_rel_price(tick)
        closest_dte = nearest_equal_abs(finish_put.columns.astype('int').values.tolist(), days_to_expiration)
        finish_put = finish_put[f'{closest_dte}']
        finish_call = finish_call[f'{closest_dte}']
        print('finish_put')

        max_put_skew_strike = finish_put[finish_put == finish_put.max()].index.tolist()[0]
        print('max_put_skew_strike',max_put_skew_strike)
        print('finish_call')
        print(finish_call)

        skew_delta_put = quotes_put[quotes_put['strike'] == max_put_skew_strike].reset_index(drop=True)['delta'].abs()[0]
        skew_delta_call = nearest_equal_abs(quotes_call['delta'].dropna().abs().values.tolist(), skew_delta_put)
        print('skew_delta_put', skew_delta_put)
        print('skew_delta_call', skew_delta_call)

    delta_list = [[delta_5_put, delta_5_call], [delta_10_put, delta_10_call], [delta_15_put, delta_15_call], [skew_delta_put, skew_delta_call]]


    for deltas in delta_list:
        delta_put = deltas[0]
        delta_call = deltas[1]
        print('delta_put', delta_put)
        print('delta_call', delta_call)
        quotes_call_row = quotes_call[quotes_call['delta'].abs() == delta_call].reset_index(drop=True).iloc[0]
        quotes_put_row = quotes_put[quotes_put['delta'].abs() == delta_put].reset_index(drop=True).iloc[0]
        print('quotes_call_row')
        print(quotes_call_row)
        print('quotes_put_row')
        print(quotes_put_row)
        print(quotes_put_row.iloc[0])
        sigma_call = quotes_call_row['iv']
        sigma_put = quotes_put_row['iv']
        call_strike = float(quotes_call_row['strike'])
        call_price = float(quotes_call_row['bid'])
        put_strike = float(quotes_put_row['strike'])
        put_price = float(quotes_put_row['bid'])

        print('sigma_call', sigma_call)
        print('sigma_put', sigma_put)
        print('call_strike', call_strike)
        print('call_price', call_price)
        print('put_strike', put_strike)
        print('put_price', put_price)

        strangle_data = shortStrangle(underlying, (sigma_call + sigma_put) / 2, rate, trials, days_to_expiration,
                                      [closing_days_array], [percentage_array], call_strike,
                                      call_price, put_strike, put_price, yahoo_stock)
        strangle_data = pd.DataFrame(strangle_data)
        strangle_data['Strike Call'] = [call_strike]
        strangle_data['Strike Put'] = [put_strike]
        sum_df = pd.concat([sum_df, strangle_data])

    return sum_df

def get_f_strangle(tick, rate, percentage_array, days_to_expiration, closing_days_array, sigma_call, sigma_put, call_price,
                   put_price, call_strike, put_strike):

    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    hv = get_exp_move(tick, yahoo_stock)

    strangle_data = shortStrangle(underlying, (sigma_call + sigma_put) / 2, rate, trials, days_to_expiration,
                                  [closing_days_array], [percentage_array], call_strike,
                                  call_price, put_strike, put_price, yahoo_stock)

    strangle_data = pd.DataFrame(strangle_data)

    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration / 365)

    return strangle_data, exp_move_hv

def get_short(tick, rate, days_to_expiration, closing_days_array, percentage_array,
              position_type, quotes, instr_type):
    print(quotes)
    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    sum_df = pd.DataFrame()

    hv = get_exp_move(tick, yahoo_stock, 244)

    if position_type == 'Put':
        quotes = quotes[quotes['instrument_class'] == 'P'].reset_index(drop=True)
        quotes = quotes[quotes['strike'] <= underlying * 1].reset_index(drop=True)
        quotes = quotes[quotes['strike'] >= underlying * 0.9].reset_index(drop=True)
        for num, quote_row in quotes.iterrows():
            print(quote_row)
            sigma = quote_row['iv'] * 100
            short_strike = quote_row['strike']
            short_price = quote_row['bid']
            short_data = shortPut(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration,[closing_days_array],
                                  [percentage_array],
                                  short_strike, short_price, yahoo_stock, instr_type)
            short_data = pd.DataFrame(short_data)
            short_data['Strike'] = [short_strike]
            sum_df = pd.concat([sum_df, short_data])

    if position_type == 'Call':
        quotes = quotes[quotes['instrument_class'] == 'C'].reset_index(drop=True)
        quotes = quotes[quotes['strike'] <= underlying * 1.1].reset_index(drop=True)
        quotes = quotes[quotes['strike'] >= underlying * 1].reset_index(drop=True)
        for num, quote_row in quotes.iterrows():
            print(quote_row)
            sigma = quote_row['iv'] * 100
            short_strike = quote_row['strike']
            short_price = quote_row['bid']
            short_data = shortCall(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration,[closing_days_array],
                                   [percentage_array],
                                   short_strike, short_price, yahoo_stock, instr_type)
            short_data = pd.DataFrame(short_data)
            short_data['Strike'] = [short_strike]
            sum_df = pd.concat([sum_df, short_data])

    nearest_atm_strike = nearest_equal_abs(quotes['strike'].astype('float'), underlying)
    iv = quotes[quotes['strike'] == nearest_atm_strike]['iv'].values.tolist()[0]
    print('nearest_strike', nearest_atm_strike)
    print('current_iv', iv)

    sum_df['top_score'] = sum_df['pop'] * sum_df['exp_return']
    best_df = sum_df[sum_df['top_score'] == sum_df['top_score'].max()]

    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration / 365)
    exp_move_iv = iv * underlying * math.sqrt(days_to_expiration / 365)

    return sum_df, best_df, exp_move_hv, exp_move_iv

def get_long(tick, rate, days_to_expiration, closing_days_array, percentage_array,
              position_type, quotes, instr_type):
    print(quotes)
    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    sum_df = pd.DataFrame()

    hv = get_exp_move(tick, yahoo_stock, 244)

    if position_type == 'Put':
        quotes = quotes[quotes['instrument_class'] == 'P'].reset_index(drop=True)
        quotes = quotes[quotes['strike'] <= underlying * 1.2].reset_index(drop=True)
        quotes = quotes[quotes['strike'] >= underlying * 0.8].reset_index(drop=True)
        for num, quote_row in quotes.iterrows():
            print(quote_row)
            sigma = quote_row['iv'] * 100
            short_strike = quote_row['strike']
            short_price = quote_row['bid']
            short_data = longPut(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration,[closing_days_array],
                                  [percentage_array],
                                  short_strike, short_price, yahoo_stock, instr_type)
            short_data = pd.DataFrame(short_data)
            short_data['Strike'] = [short_strike]
            sum_df = pd.concat([sum_df, short_data])

    if position_type == 'Call':
        quotes = quotes[quotes['instrument_class'] == 'C'].reset_index(drop=True)
        quotes = quotes[quotes['strike'] <= underlying * 1.2].reset_index(drop=True)
        quotes = quotes[quotes['strike'] >= underlying * 0.8].reset_index(drop=True)
        for num, quote_row in quotes.iterrows():
            print(quote_row)
            sigma = quote_row['iv'] * 100
            short_strike = quote_row['strike']
            short_price = quote_row['bid']
            short_data = longCall(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration,[closing_days_array],
                                   [percentage_array],
                                   short_strike, short_price, yahoo_stock, instr_type)
            short_data = pd.DataFrame(short_data)
            short_data['Strike'] = [short_strike]
            sum_df = pd.concat([sum_df, short_data])

    nearest_atm_strike = nearest_equal_abs(quotes['strike'].astype('float'), underlying)
    iv = quotes[quotes['strike'] == nearest_atm_strike]['iv'].values.tolist()[0]
    print('nearest_strike', nearest_atm_strike)
    print('current_iv', iv)

    sum_df['top_score'] = sum_df['pop'] * sum_df['exp_return']
    best_df = sum_df[sum_df['top_score'] == sum_df['top_score'].max()]

    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration / 365)
    exp_move_iv = iv * underlying * math.sqrt(days_to_expiration / 365)

    return sum_df, best_df, exp_move_hv, exp_move_iv

def get_f_short(tick, rate, days_to_expiration, closing_days_array, percentage_array,
              position_type, sigma, short_strike, short_price):

    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    sum_df = pd.DataFrame()

    hv = get_exp_move(tick, yahoo_stock)

    if position_type == 'Put':
        # quotes = quotes[quotes['side'] == 'put'].reset_index(drop=True)
        # quotes = quotes[quotes['strike'] <= underlying * 1].reset_index(drop=True)
        # quotes = quotes[quotes['strike'] >= underlying * 0.9].reset_index(drop=True)
        # for num, quote_row in quotes.iterrows():
        short_data = shortPut(underlying, sigma, rate, trials, days_to_expiration, [closing_days_array],
                              [percentage_array],
                              short_strike, short_price, yahoo_stock)
        short_data = pd.DataFrame(short_data)

    if position_type == 'Call':

        short_data = shortCall(underlying, sigma, rate, trials, days_to_expiration, [closing_days_array],
                               [percentage_array],
                               short_strike, short_price, yahoo_stock)
        short_data = pd.DataFrame(short_data)

    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration / 365)

    return short_data, exp_move_hv

def get_calendar_diagonal(tick, rate, days_to_expiration_long, days_to_expiration_short, closing_days_array,
                          percentage_array, position_type, quotes_short, quotes_long, short_count, long_count, position_options):
    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    sum_df = pd.DataFrame()

    hv = get_exp_move(tick, yahoo_stock)

    if position_type == 'put':
        quotes_short = quotes_short[quotes_short['side'] == 'put'].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] <= underlying * position_options['short_strike_limit_to']].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] >= underlying * position_options['short_strike_limit_from']].reset_index(drop=True)

        quotes_long = quotes_long[quotes_long['side'] == 'put'].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] <= underlying * position_options['long_strike_limit_to']].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] >= underlying * position_options['long_strike_limit_from']].reset_index(drop=True)
        for num, quotes_short_row in quotes_short.iterrows():
            for num_long, quotes_long_row in quotes_long.iterrows():
                sigma_short = quotes_short_row['iv'] * 100
                short_strike = quotes_short_row['strike']
                short_price = quotes_short_row['bid']
                sigma_long = quotes_long_row['iv'] * 100
                long_strike = quotes_long_row['strike']
                long_price = quotes_long_row['ask']

                if position_options['structure'] == 'calendar':
                    if quotes_short_row['strike'] == quotes_long_row['strike']:
                        print('underlying', underlying)
                        print('short_count', short_count)
                        print('long_count', long_count)
                        print('rate', rate)
                        print('short bid', quotes_short_row['bid'])
                        print('short_price', short_price)
                        print('long ask', quotes_long_row['ask'])
                        print('long_price', long_price)
                        print('long_strike', long_strike)
                        print('short_strike', short_strike)
                        print('sigma_long', sigma_long)
                        print('sigma_short', sigma_short)
                        print('days_to_expiration_short', days_to_expiration_short)
                        print('days_to_expiration_long', days_to_expiration_long)
                        print('closing_days_array', closing_days_array)
                        print('percentage_array', percentage_array)
                        print('position_options', position_options)

                        calendar_diagonal_data, max_profit, percentage_type = putCalendar_template(underlying, sigma_short, sigma_long, rate, trials,
                                                             days_to_expiration_short, days_to_expiration_long,
                                                             [closing_days_array],
                                                             [percentage_array], long_strike, long_price, short_strike,
                                                             short_price, yahoo_stock, short_count, long_count, position_options)
                        print('calendar_diagonal_data', calendar_diagonal_data)
                        calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                        calendar_diagonal_data['Strike_Short'] = [short_strike]
                        calendar_diagonal_data['Strike_Long'] = [long_strike]
                        sum_df = pd.concat([sum_df, calendar_diagonal_data])
                else:
                    calendar_diagonal_data, max_profit, percentage_type = putCalendar_template(underlying, sigma_short,
                                                                                               sigma_long, rate, trials,
                                                                                               days_to_expiration_short,
                                                                                               days_to_expiration_long,
                                                                                               [closing_days_array],
                                                                                               [percentage_array],
                                                                                               long_strike, long_price,
                                                                                               short_strike,
                                                                                               short_price, yahoo_stock,
                                                                                               short_count, long_count,
                                                                                               position_options)
                    print('calendar_diagonal_data', calendar_diagonal_data)
                    calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                    calendar_diagonal_data['Strike_Short'] = [short_strike]
                    calendar_diagonal_data['Strike_Long'] = [long_strike]
                    sum_df = pd.concat([sum_df, calendar_diagonal_data])

    if position_type == 'call':
        quotes_short = quotes_short[quotes_short['side'] == 'call'].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] <= underlying * position_options['short_strike_limit_to']].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] >= underlying * position_options['short_strike_limit_from']].reset_index(drop=True)

        quotes_long = quotes_long[quotes_long['side'] == 'call'].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] <= underlying * position_options['long_strike_limit_to']].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] >= underlying * position_options['long_strike_limit_from']].reset_index(drop=True)

        # if position_options['structure'] == 'calendar':
        #     quotes_long =

        for num, quotes_short_row in quotes_short.iterrows():
            for num_long, quotes_long_row in quotes_long.iterrows():
                sigma_short = quotes_short_row['iv'] * 100
                short_strike = quotes_short_row['strike']
                short_price = quotes_short_row['bid'] * short_count

                sigma_long = quotes_long_row['iv'] * 100
                long_strike = quotes_long_row['strike']
                long_price = quotes_long_row['ask'] * long_count

                print('short bid', quotes_short_row['bid'])
                print('short_price', short_price)
                print('long ask', quotes_long_row['ask'])
                print('long_price', long_price)
                print('long_strike', long_strike)
                print('short_strike', short_strike)
                print('long_strike', sigma_long)
                print('short_strike', sigma_short)
                print('days_to_expiration_short', days_to_expiration_short)
                print('days_to_expiration_long', days_to_expiration_long)

                if position_options['structure'] == 'calendar':
                    if quotes_short_row['strike'] == quotes_long_row['strike']:
                        calendar_diagonal_data, max_profit, percentage_type = callCalendar_template(underlying, sigma_short, sigma_long, rate, trials,
                                                             days_to_expiration_short, days_to_expiration_long,
                                                             [closing_days_array],
                                                             [percentage_array], long_strike, long_price, short_strike,
                                                             short_price, yahoo_stock,
                                                             short_count, long_count, position_options)

                        calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                        calendar_diagonal_data['Strike_Short'] = [short_strike]
                        calendar_diagonal_data['Strike_Long'] = [long_strike]
                        sum_df = pd.concat([sum_df, calendar_diagonal_data])
                else:
                    calendar_diagonal_data, max_profit, percentage_type = callCalendar_template(underlying, sigma_short,
                                                                                                sigma_long, rate,
                                                                                                trials,
                                                                                                days_to_expiration_short,
                                                                                                days_to_expiration_long,
                                                                                                [closing_days_array],
                                                                                                [percentage_array],
                                                                                                long_strike, long_price,
                                                                                                short_strike,
                                                                                                short_price,
                                                                                                yahoo_stock,
                                                                                                short_count, long_count,
                                                                                                position_options)

                    calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                    calendar_diagonal_data['Strike_Short'] = [short_strike]
                    calendar_diagonal_data['Strike_Long'] = [long_strike]
                    sum_df = pd.concat([sum_df, calendar_diagonal_data])

    nearest_atm_strike = nearest_equal_abs(quotes_short['strike'].astype('float'), underlying)
    iv = quotes_short[quotes_short['strike'] == nearest_atm_strike]['iv'].values.tolist()[0]
    print('nearest_strike', nearest_atm_strike)
    print('current_iv', iv)
    sum_df['top_score'] = sum_df['pop'] * sum_df['exp_return']
    best_df = sum_df[sum_df['top_score'] == sum_df['top_score'].max()]
    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration_short / 365)
    exp_move_iv = iv * underlying * math.sqrt(days_to_expiration_short / 365)

    return sum_df, best_df, exp_move_hv, exp_move_iv, max_profit, percentage_type


def get_calendar_diagonal_input(ticker, rate, days_to_expiration_7, days_to_expiration_30, days_to_expiration_90,
                             percentage_array, quotes_90, position_type, quotes_30, short_count, long_count, quotes_7):
    yahoo_stock = get_yahoo_price(ticker)
    underlying = yahoo_stock['Close'].iloc[-1]
    sum_df = pd.DataFrame()

    if '-' in ticker or '=' in ticker:
        side = 'Call'
        if position_type == 'P':
            side = 'Put'
        print(str(Path(__file__).parents[1]))
        quotes_7 = pd.read_csv(f'{str(Path(__file__).parents[1])}/DATA/{ticker}_1.csv')
        quotes_7.columns = [i.lower() for i in quotes_7.columns.tolist()]
        quotes_7['days_to_exp'] = days_to_expiration_7
        quotes_7['underlyingPrice'] = underlying

        quotes_7 = quotes_7[quotes_7['type'] == side].reset_index(drop=True)
        print('quotes_7')
        quotes_7['strike'] = quotes_7['strike'].str.replace('P', '').str.replace('C', '').str.split(',').str.join('').astype('float')
        quotes_7 = greek_calc(quotes_7, position_type, 'F')
        quotes_7['side'] = side.lower()

        # quotes 30
        quotes_30 = pd.read_csv(f'{str(Path(__file__).parents[1])}/DATA/{ticker}_2.csv')
        quotes_30.columns = [i.lower() for i in quotes_30.columns.tolist()]
        quotes_30['days_to_exp'] = days_to_expiration_30
        quotes_30['underlyingPrice'] = underlying
        quotes_30 = quotes_30[quotes_30['type'] == side].reset_index(drop=True)
        print('quotes_30')
        quotes_30['strike'] = quotes_30['strike'].str.replace('P', '').str.replace('C', '').str.split(',').str.join('').astype('float')
        print(quotes_30)
        quotes_30 = greek_calc(quotes_30, position_type, 'F')
        quotes_30['side'] = side.lower()
        print('quotes_30')
        print(quotes_30)
        # quotes 90
        quotes_90 = pd.read_csv(f'{str(Path(__file__).parents[1])}/DATA/{ticker}_2.csv')
        quotes_90.columns = [i.lower() for i in quotes_90.columns.tolist()]
        quotes_90['days_to_exp'] = days_to_expiration_90
        quotes_90['underlyingPrice'] = underlying
        quotes_90 = quotes_90[quotes_90['type'] == side].reset_index(drop=True)
        print('quotes_30')
        quotes_90['strike'] = quotes_90['strike'].str.replace('P', '').str.replace('C', '').str.split(',').str.join('').astype('float')
        print(quotes_90)
        quotes_90 = greek_calc(quotes_90, position_type, 'F')
        quotes_90['side'] = side.lower()


    pair = [[quotes_7, quotes_30, days_to_expiration_7, days_to_expiration_30], [quotes_30, quotes_90, days_to_expiration_30, days_to_expiration_90]]

    for pair_quotes in pair:
        short_quote = pair_quotes[0]
        long_quote = pair_quotes[1]
        days_to_expiration_short = pair_quotes[2]
        days_to_expiration_long = pair_quotes[3]

        trials = 5000
        hv = get_exp_move(ticker, yahoo_stock, days_to_expiration_short)
        print('hv', hv)
        exp_move_hv = hv * underlying * math.sqrt(days_to_expiration_short / 365)
        exp_move_price = underlying + exp_move_hv

        print('underlying', underlying)
        print('exp_move_hv', exp_move_hv)
        print('exp_move_price', exp_move_price)


        if position_type == 'P':
            short_quote = short_quote[short_quote['strike'] >= int(underlying - exp_move_hv)]
            short_quote = short_quote[short_quote['strike'] <= int(underlying)]

            short_quote = short_quote[short_quote['side'] == 'put']
            long_quote = long_quote[long_quote['side'] == 'put']

            for strike in short_quote['strike'].values.tolist():
                try:
                    print(short_quote[short_quote['strike'] == strike])
                    print(long_quote[long_quote['strike'] == strike])
                    short_local = short_quote[short_quote['strike'] == strike].reset_index(drop=True).iloc[0]
                    long_local = long_quote[long_quote['strike'] == strike].reset_index(drop=True).iloc[0]

                    sigma_short = short_local['iv']
                    short_strike = short_local['strike']
                    short_price = short_local['bid'] * short_count

                    sigma_long = long_local['iv']
                    long_strike = long_local['strike']
                    long_price = long_local['ask'] * long_count

                    calendar_diagonal_data, profit_for_percent, percentage_type = putCalendar(underlying, sigma_short, sigma_long, rate, trials,
                                             days_to_expiration_short, days_to_expiration_long, [days_to_expiration_short], [percentage_array],
                                             long_strike, long_price, short_strike, short_price, yahoo_stock, short_count,
                                             long_count)
                    print("calendar_diagonal_data Put: ", calendar_diagonal_data)
                    calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                    calendar_diagonal_data['Strike_Short'] = [short_strike]
                    calendar_diagonal_data['Strike_Long'] = [long_strike]
                    calendar_diagonal_data['days_to_expiration_short'] = [days_to_expiration_short]
                    calendar_diagonal_data['days_to_expiration_long'] = [days_to_expiration_long]
                    sum_df = pd.concat([sum_df, calendar_diagonal_data])

                except:
                    pass

        if position_type == 'C':
            short_quote = short_quote[short_quote['strike'] >= int(underlying)]
            short_quote = short_quote[short_quote['strike'] <= int(underlying + exp_move_hv)]
            short_quote = short_quote[short_quote['side'] == 'call']
            long_quote = long_quote[long_quote['side'] == 'call']

            for strike in short_quote['strike'].values.tolist():
                try:
                    print(short_quote[short_quote['strike'] == strike])
                    print(long_quote[long_quote['strike'] == strike])
                    short_local = short_quote[short_quote['strike'] == strike].reset_index(drop=True).iloc[0]
                    long_local = long_quote[long_quote['strike'] == strike].reset_index(drop=True).iloc[0]

                    sigma_short = short_local['iv']
                    short_strike = short_local['strike']
                    short_price = short_local['bid'] * short_count

                    sigma_long = long_local['iv']
                    long_strike = long_local['strike']
                    long_price = long_local['ask'] * long_count

                    print('sigma_short', sigma_short)
                    print('sigma_long', sigma_long)
                    print('rate', rate)
                    print('days_to_expiration_short', days_to_expiration_short)
                    print('days_to_expiration_long', days_to_expiration_long)
                    print('percentage_array', percentage_array)
                    print('long_strike', long_strike)

                    print('long_price', long_price)
                    print('short_strike', short_strike)
                    print('short_price', short_price)
                    print('short_count', short_count)
                    print('long_count', long_count)


                    calendar_diagonal_data, profit_for_percent, percentage_type = callCalendar(underlying, sigma_short,
                                                                  sigma_long, rate, trials, days_to_expiration_short,
                                                                  days_to_expiration_long,[days_to_expiration_short],
                                                                  [percentage_array],long_strike,
                                                                  long_price, short_strike, short_price, yahoo_stock,
                                                                  short_count, long_count)


                    print("calendar_diagonal_data Call: ", calendar_diagonal_data)
                    calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                    calendar_diagonal_data['Strike_Short'] = [short_strike]
                    calendar_diagonal_data['Strike_Long'] = [long_strike]
                    calendar_diagonal_data['days_to_expiration_short'] = [days_to_expiration_short]
                    calendar_diagonal_data['days_to_expiration_long'] = [days_to_expiration_long]
                    sum_df = pd.concat([sum_df, calendar_diagonal_data])
                except:
                    pass

    return sum_df


def get_risk_reversal(tick, sigma, rate, days_to_expiration, closing_days_array,
                      percentage_array, long_strike, long_price, short_strike, short_price):
    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    risk_reversal_data = riskReversal(underlying, sigma, rate, trials, days_to_expiration,
                                      [closing_days_array], [percentage_array], long_strike, long_price, short_strike,
                                      short_price, yahoo_stock)

    print("risk_reversal_data: ", risk_reversal_data)

    return pd.DataFrame(risk_reversal_data)


# def get_long(ticker, rate, days_to_expiration, closing_days_array, percentage_array,
#                                   side, quotes, instr_type):
#     yahoo_stock = get_yahoo_price(ticker)
#     underlying = yahoo_stock['Close'].iloc[-1]
#     trials = 2000
#
#     sum_df = pd.DataFrame()
#
#     if side == 'P':
#         if '-' in ticker or '=' in ticker:
#             quotes['days_to_exp'] = days_to_expiration
#             quotes['underlyingPrice'] = underlying
#             quotes = quotes[quotes['instrument_class'] == side].reset_index(drop=True)
#             print('quotes111')
#
#         else:
#             quotes = quotes[quotes['side'] == 'put'].reset_index(drop=True)
#
#         quotes = quotes[quotes['strike'] <= underlying * 1.2].reset_index(drop=True)
#         quotes = quotes[quotes['strike'] >= underlying * 0.8].reset_index(drop=True)
#
#         for num, quote_row in quotes.iterrows():
#             print(quote_row)
#             sigma = quote_row['iv']
#             long_strike = quote_row['strike']
#             long_price = quote_row['bid']
#             long_data = longPut(underlying, sigma, rate, trials, days_to_expiration, [closing_days_array],
#                                   [percentage_array], long_strike, long_price, yahoo_stock, instr_type)
#             long_data = pd.DataFrame(long_data)
#             long_data['Strike'] = [long_strike]
#             sum_df = pd.concat([sum_df, long_data])
#
#     if side == 'C':
#         if '-' in ticker or '=' in ticker:
#             quotes['days_to_exp'] = days_to_expiration
#             quotes['underlyingPrice'] = underlying
#             quotes = quotes[quotes['instrument_class'] == side].reset_index(drop=True)
#         else:
#             quotes = quotes[quotes['side'] == 'call'].reset_index(drop=True)
#
#         quotes = quotes[quotes['strike'] <= underlying * 1.2].reset_index(drop=True)
#         quotes = quotes[quotes['strike'] >= underlying * 0.8].reset_index(drop=True)
#
#         for num, quote_row in quotes.iterrows():
#             print(quote_row)
#             sigma = quote_row['iv']
#             long_strike = quote_row['strike']
#             long_price = quote_row['bid']
#             long_data = longCall(underlying, sigma, rate, trials, days_to_expiration, [closing_days_array],
#                                   [percentage_array], long_strike, long_price, yahoo_stock, instr_type)
#             long_data = pd.DataFrame(long_data)
#             long_data['Strike'] = [long_strike]
#             sum_df = pd.concat([sum_df, long_data])
#
#     return sum_df

def get_f_long(tick, rate, days_to_expiration, closing_days_array, percentage_array,
                                  position_type, sigma, long_strike, long_price):
    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    if position_type == 'Put':
        long_data = longPut(underlying, sigma, rate, trials, days_to_expiration, [closing_days_array],
                            [percentage_array],
                            long_strike, long_price, yahoo_stock)
        print("Long Put: ", long_data)

    if position_type == 'Call':
        long_data = longCall(underlying, sigma, rate, trials, days_to_expiration, [closing_days_array],
                             [percentage_array],
                             long_strike, long_price, yahoo_stock)
        print("Long Call: ", long_data)

    return pd.DataFrame(long_data)

def get_f_da_but(ticker, rate, days_to_expiration_1, days_to_expiration_2_long, closing_days_array, percentage_array,
                       long_1_sigma, long_2_sigma, short_sigma, long_1_prime, long_2_prime, short_prime, long_1_strike,
                      long_2_strike, short_strike, long_1_count, long_2_count, short_count):

    yahoo_stock = get_yahoo_price(ticker)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    hv = get_exp_move(ticker, yahoo_stock)


    da_bat_data = fut_DA_BUT(underlying, long_1_sigma, long_2_sigma, short_sigma, rate, trials,
     days_to_expiration_1, days_to_expiration_2_long, [closing_days_array], [percentage_array], long_1_strike,
     long_2_strike, short_strike, long_1_prime, long_2_prime, short_prime, yahoo_stock, long_1_count,
     long_2_count, short_count)


    da_bat_data = pd.DataFrame(da_bat_data)
    print('da_bat_data', da_bat_data)

    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration_1 / 365)

    return da_bat_data

def get_f_ratio_112(ticker, long_sigma, short_1_sigma, short_2_sigma, rate, days_to_expiration,
                          closing_days_array, percentage_array, long_strike, short_1_strike, short_2_strike,
                          long_prime, short_1_prime, short_2_prime,  long_count, short_1_count, short_2_count):

    yahoo_stock = get_yahoo_price(ticker)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000

    hv = get_exp_move(ticker, yahoo_stock)


    da_ratio_112_data = futRatio_1_1_2(underlying, long_sigma, short_1_sigma, short_2_sigma, rate, trials,
                days_to_expiration, [closing_days_array], [percentage_array], long_strike,
                short_1_strike, short_2_strike, long_prime, short_1_prime,  short_2_prime, yahoo_stock, long_count,
                short_1_count, short_2_count)

    print('da_ratio_112_data', da_ratio_112_data)
    da_ratio_112_data = pd.DataFrame(da_ratio_112_data)


    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration / 365)

    return da_ratio_112_data


def get_ratio_112(ticker, rate,  days_to_expiration, closing_days_array, percentage_array, quotes,  long_count,
                  short_1_count, short_2_count, start_side, instr_type):
    yahoo_stock = get_yahoo_price(ticker)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 5000
    sum_df = pd.DataFrame()
    hv = get_exp_move(ticker, yahoo_stock, 244)

    if '-' in ticker or '=' in ticker:
        quotes['days_to_exp'] = days_to_expiration
        quotes['underlyingPrice'] = underlying

        # quotes = quotes[quotes['instrument_class'] == start_side].reset_index(drop=True)
        print('quotes111')
        # quotes['strike'] = quotes['strike'].str.replace('P', '').str.replace('C', '').str.split(',').str.join('').astype('float')
        print(quotes)

        long_df = quotes[quotes['delta'].abs() >= 0.25].reset_index(drop=True)
        long_df = long_df[long_df['delta'].abs() <= 0.30].reset_index(drop=True)

        # short_1_df = quotes[quotes['delta'].abs() >= 0.25].reset_index(drop=True)
        # short_1_df = short_1_df[short_1_df['delta'].abs() <= 0.45].reset_index(drop=True)

        short_2_df = quotes[quotes['delta'].abs() >= 0.05].reset_index(drop=True)
        short_2_df = short_2_df[short_2_df['delta'].abs() <= 0.15].reset_index(drop=True)
    else:
        # side = 'call'
        # if start_side == 'P':
        #     side = 'put'
        quotes = quotes[quotes['side'] == start_side].reset_index(drop=True)
        long_df = quotes[quotes['delta'].abs() >= 0.25].reset_index(drop=True)
        long_df = long_df[long_df['delta'].abs() <= 0.45].reset_index(drop=True)

        # short_1_df = quotes[quotes['delta'].abs() >= 0.25].reset_index(drop=True)
        # short_1_df = short_1_df[short_1_df['delta'].abs() <= 0.45].reset_index(drop=True)
        print('short_2_df')
        short_2_df = quotes[quotes['delta'].abs() >= 0.05].reset_index(drop=True)
        short_2_df = short_2_df[short_2_df['delta'].abs() <= 0.20].reset_index(drop=True)
        # делаем импорт и подбор разных вариантов страйков. Лонг 1 - от 45 до 25.
        # Шорт 1 разной ширины от лонга от 1 до 3 страйков. Шорт два от 20 до 5 дельта.

    print('short_2_df')
    print(short_2_df)
    print('long_df')
    print(long_df)
    for num, long_row in long_df.iterrows():
        if start_side == 'P':
            short_1_local = quotes[quotes['strike'] < float(long_row['strike'])]
            short_1_local = short_1_local.sort_values('strike', ascending=False)[:3]

        if start_side == 'C':
            short_1_local = quotes[quotes['strike'] > float(long_row['strike'])]
            # print(float(long_row['strike']))
            # print(short_1_local)
            short_1_local = short_1_local.sort_values('strike', ascending=True)[:3]
            # print('start_side',start_side)
            # print(short_1_local)

        print('11111111111111111')
        print(long_df)
        print(short_1_local)
        print(short_2_df)

        for num1, short_1_row in short_1_local.iterrows():
            print('short_1_local')
            print(short_1_local)
            for num2, short_2_row in short_2_df.iterrows():
                print('short_2_row')
                print(short_2_row)
                if float(short_1_row['strike']) != float(short_2_row['strike']):
                    long_strike = float(long_row['strike'])
                    short_1_strike = float(short_1_row['strike'])
                    short_2_strike = float(short_2_row['strike'])

                    long_prime = float(long_row['ask'])
                    short_1_prime = float(short_1_row['bid'])
                    short_2_prime = float(short_2_row['bid'])

                    long_sigma = float(long_row['iv'])
                    short_1_sigma = float(short_1_row['iv'])
                    short_2_sigma = float(short_2_row['iv'])

                    print('long_strike', long_strike)
                    print('short_1_strike', short_1_strike)
                    print('short_2_strike', short_2_strike)
                    print('long_prime', long_prime)
                    print('short_1_prime', short_1_prime)
                    print('short_2_prime', short_2_prime)

                    da_ratio_112_data = futRatio_1_1_2(underlying, long_sigma, short_1_sigma, short_2_sigma, rate, trials,
                                                       days_to_expiration, [closing_days_array], [percentage_array],
                                                       long_strike,
                                                       short_1_strike, short_2_strike, long_prime, short_1_prime, short_2_prime,
                                                       yahoo_stock, long_count,
                                                       short_1_count, short_2_count, start_side, instr_type)


                    print('da_ratio_112_data', da_ratio_112_data)
                    ratio_data = pd.DataFrame(da_ratio_112_data)
                    ratio_data['Strike Long'] = [long_strike]
                    ratio_data['Strike Short 1'] = [short_1_strike]
                    ratio_data['Strike Short 2'] = [short_2_strike]
                    sum_df = pd.concat([sum_df, ratio_data])



    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration / 365)

    return sum_df

def get_covered(tick, quotes, rate, percentage_array ):
    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 3000

    sum_df = pd.DataFrame()

    hv = get_exp_move(tick, yahoo_stock)


    quotes_put = quotes[quotes['side'] == 'put'].reset_index(drop=True)
    quotes_put = quotes_put[quotes_put['strike'] <= underlying * 1.05].reset_index(drop=True)
    quotes_put = quotes_put[quotes_put['strike'] >= underlying].reset_index(drop=True)

    for num, quotes_short_row in quotes_put.iterrows():

        sigma_short = quotes_short_row['iv'] * 100
        short_strike = quotes_short_row['strike']
        short_price = quotes_short_row['bid']
        sigma_long = quotes_long_row['iv'] * 100
        long_strike = quotes_long_row['strike']
        long_price = quotes_long_row['ask']

        if position_options['structure'] == 'calendar':
            if quotes_short_row['strike'] == quotes_long_row['strike']:
                print('underlying', underlying)
                print('short_count', short_count)
                print('long_count', long_count)
                print('rate', rate)
                print('short bid', quotes_short_row['bid'])
                print('short_price', short_price)
                print('long ask', quotes_long_row['ask'])
                print('long_price', long_price)
                print('long_strike', long_strike)
                print('short_strike', short_strike)
                print('sigma_long', sigma_long)
                print('sigma_short', sigma_short)
                print('days_to_expiration_short', days_to_expiration_short)
                print('days_to_expiration_long', days_to_expiration_long)
                print('closing_days_array', closing_days_array)
                print('percentage_array', percentage_array)
                print('position_options', position_options)

                calendar_diagonal_data, max_profit, percentage_type = putCalendar_template(underlying, sigma_short, sigma_long, rate, trials,
                                                     days_to_expiration_short, days_to_expiration_long,
                                                     [closing_days_array],
                                                     [percentage_array], long_strike, long_price, short_strike,
                                                     short_price, yahoo_stock, short_count, long_count, position_options)
                print('calendar_diagonal_data', calendar_diagonal_data)
                calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                calendar_diagonal_data['Strike_Short'] = [short_strike]
                calendar_diagonal_data['Strike_Long'] = [long_strike]
                sum_df = pd.concat([sum_df, calendar_diagonal_data])
        else:
            calendar_diagonal_data, max_profit, percentage_type = putCalendar_template(underlying, sigma_short,
                                                                                       sigma_long, rate, trials,
                                                                                       days_to_expiration_short,
                                                                                       days_to_expiration_long,
                                                                                       [closing_days_array],
                                                                                       [percentage_array],
                                                                                       long_strike, long_price,
                                                                                       short_strike,
                                                                                       short_price, yahoo_stock,
                                                                                       short_count, long_count,
                                                                                       position_options)
            print('calendar_diagonal_data', calendar_diagonal_data)
            calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
            calendar_diagonal_data['Strike_Short'] = [short_strike]
            calendar_diagonal_data['Strike_Long'] = [long_strike]
            sum_df = pd.concat([sum_df, calendar_diagonal_data])

    if position_type == 'call':
        quotes_short = quotes_short[quotes_short['side'] == 'call'].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] <= underlying * position_options['short_strike_limit_to']].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] >= underlying * position_options['short_strike_limit_from']].reset_index(drop=True)

        quotes_long = quotes_long[quotes_long['side'] == 'call'].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] <= underlying * position_options['long_strike_limit_to']].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] >= underlying * position_options['long_strike_limit_from']].reset_index(drop=True)

        # if position_options['structure'] == 'calendar':
        #     quotes_long =

        for num, quotes_short_row in quotes_short.iterrows():
            for num_long, quotes_long_row in quotes_long.iterrows():
                sigma_short = quotes_short_row['iv'] * 100
                short_strike = quotes_short_row['strike']
                short_price = quotes_short_row['bid'] * short_count

                sigma_long = quotes_long_row['iv'] * 100
                long_strike = quotes_long_row['strike']
                long_price = quotes_long_row['ask'] * long_count

                print('short bid', quotes_short_row['bid'])
                print('short_price', short_price)
                print('long ask', quotes_long_row['ask'])
                print('long_price', long_price)
                print('long_strike', long_strike)
                print('short_strike', short_strike)
                print('long_strike', sigma_long)
                print('short_strike', sigma_short)
                print('days_to_expiration_short', days_to_expiration_short)
                print('days_to_expiration_long', days_to_expiration_long)

                if position_options['structure'] == 'calendar':
                    if quotes_short_row['strike'] == quotes_long_row['strike']:
                        calendar_diagonal_data, max_profit, percentage_type = callCalendar_template(underlying, sigma_short, sigma_long, rate, trials,
                                                             days_to_expiration_short, days_to_expiration_long,
                                                             [closing_days_array],
                                                             [percentage_array], long_strike, long_price, short_strike,
                                                             short_price, yahoo_stock,
                                                             short_count, long_count, position_options)

                        calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                        calendar_diagonal_data['Strike_Short'] = [short_strike]
                        calendar_diagonal_data['Strike_Long'] = [long_strike]
                        sum_df = pd.concat([sum_df, calendar_diagonal_data])
                else:
                    calendar_diagonal_data, max_profit, percentage_type = callCalendar_template(underlying, sigma_short,
                                                                                                sigma_long, rate,
                                                                                                trials,
                                                                                                days_to_expiration_short,
                                                                                                days_to_expiration_long,
                                                                                                [closing_days_array],
                                                                                                [percentage_array],
                                                                                                long_strike, long_price,
                                                                                                short_strike,
                                                                                                short_price,
                                                                                                yahoo_stock,
                                                                                                short_count, long_count,
                                                                                                position_options)

                    calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                    calendar_diagonal_data['Strike_Short'] = [short_strike]
                    calendar_diagonal_data['Strike_Long'] = [long_strike]
                    sum_df = pd.concat([sum_df, calendar_diagonal_data])

    nearest_atm_strike = nearest_equal_abs(quotes_short['strike'].astype('float'), underlying)
    iv = quotes_short[quotes_short['strike'] == nearest_atm_strike]['iv'].values.tolist()[0]
    print('nearest_strike', nearest_atm_strike)
    print('current_iv', iv)
    sum_df['top_score'] = sum_df['pop'] * sum_df['exp_return']
    best_df = sum_df[sum_df['top_score'] == sum_df['top_score'].max()]
    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration_short / 365)
    exp_move_iv = iv * underlying * math.sqrt(days_to_expiration_short / 365)

    return sum_df, best_df, exp_move_hv, exp_move_iv, max_profit, percentage_type