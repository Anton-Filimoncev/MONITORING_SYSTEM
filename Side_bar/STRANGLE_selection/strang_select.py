import numpy as np
import pandas as pd
import requests
import datetime
from dateutil.relativedelta import relativedelta
import mibian
import math
import tqdm
from scipy.stats import norm
import yfinance as yf
import asyncio
import aiohttp
from .Black76 import black76Put
from .Black76 import black76Call
from .MonteCarlo import monteCarlo
import time
from .BlackScholes import blackScholesCall, blackScholesPut
from .MonteCarlo_RETURN import monteCarlo_exp_return

def get_exp_move(tick, stock_yahoo, period):
    print('---------------------------')
    print('------------- Getting HV --------------')
    print('---------------------------')

    stock_yahoo_solo = stock_yahoo['Close']
    hist_vol = volatility_calc(stock_yahoo_solo, period)

    return hist_vol


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

def get_yahoo_price(ticker):
    yahoo_data = yf.download(ticker, progress=False)['2018-01-01':]
    return yahoo_data

def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))

def zeta_call(stock, exercise, sigma, rate, dividend, time):
    try:
        time = time / 365
        u = ((rate - dividend) - (math.pow(sigma, 2) / 2)) / (math.pow(sigma, 2))
        l = math.sqrt(((-1) * math.pow(u, 2)) + ((2 * rate) / math.pow(sigma, 2)))
        z = math.log(exercise / stock) / (sigma * math.sqrt(time)) + l * sigma * math.sqrt(time)
        return np.min([math.pow(exercise/stock,u+l)*norm.cdf(-1*z)+math.pow(exercise/stock,u-l)*norm.cdf(-1*z+2*l*sigma*math.sqrt(time)), 1])
    except:
        return None



def bsm_debit(sim_price, strikes, rate, time_fraction,  sigma_call, sigma_put, instr_type):

    if instr_type == 'FUT':
        P_short_calls = black76Put(sim_price, strikes[0], rate, time_fraction, sigma_call)
        P_short_puts = black76Put(sim_price, strikes[1], rate, time_fraction, sigma_put)

    else:
        P_short_calls = blackScholesCall(sim_price, strikes[0], rate, time_fraction, sigma_call)
        P_short_puts = blackScholesPut(sim_price, strikes[1], rate, time_fraction, sigma_put)

    debit = P_short_calls + P_short_puts
    # debit = P_long_puts - P_short_puts

    return debit

def zeta_put(stock, exercise, sigma, rate, dividend, time):
    try:
        time = time / 365
        u = ((rate - dividend) - (math.pow(sigma, 2) / 2)) / (math.pow(sigma, 2))
        l = math.sqrt(((-1) * math.pow(u, 2)) + ((2 * rate) / math.pow(sigma, 2)))
        z = np.log(exercise / stock) / (sigma * math.sqrt(time)) + l * sigma * math.sqrt(time)
        return np.min([math.pow(exercise / stock, u + l) * norm.cdf(z) + math.pow(exercise / stock, u - l) * norm.cdf(
                z - 2 * l * sigma * math.sqrt(time)), 1])
    except:
        return None


def price_calc(side, current_price, strike, days_to_exp, volatility):

    c = mibian.BS([current_price, strike, 1, days_to_exp], volatility=volatility)

    if side == 'C':
        return c.callPrice

    elif side == 'P':
        return c.putPrice

def price_calc_76(option_type, fs, x, t, r, b, v):
    # -----------
    # Create preliminary calculations
    t__sqrt = math.sqrt(t)
    d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
    d2 = d1 - v * t__sqrt
    if option_type == "c":
        value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
    else:
        value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))

    return value

def get_rel_price_76(quotes_put, quotes_call, current_price, u_dte):

    atm_strike = nearest_equal_abs(quotes_put['strike'].astype('float').values.tolist(), current_price)
    local_df_p = quotes_put[quotes_put['strike'] >= atm_strike * 0.5]
    local_df_c = quotes_call[quotes_call['strike'] <= atm_strike * 1.5]

    atm_iv_p = local_df_p[local_df_p['strike'] == atm_strike]['iv'].values[0]
    # atm_iv_c = local_df_c[local_df_c['strike'] == atm_strike]['iv'].values[0]
    # print('atm_iv_p', atm_iv_p)
    # print('atm_iv_c', atm_iv_c)
    put_price_list = []
    call_price_list = []
    for strike in local_df_p['strike']:
        put_price_list.append(price_calc_76('p', current_price, strike, u_dte/365, 0.04, 0.04, atm_iv_p))
    # print('put_price_list', put_price_list)

    for strike in local_df_c['strike']:
        call_price_list.append(price_calc_76('c', current_price, strike, u_dte/365, 0.04, 0.04, atm_iv_p))
    # print('call_price_list', call_price_list)

    local_df_p['atm_price'] = put_price_list
    local_df_c['atm_price'] = call_price_list

    finish_call = local_df_c['bid'] - local_df_c['atm_price']
    finish_call.index = [local_df_c['strike'].values.tolist()]
    finish_put = local_df_p['bid'] - local_df_p['atm_price']
    finish_put.index = [local_df_p['strike'].values.tolist()]

    return finish_put, finish_call

async def get_market_data(session, url):
    async with session.get(url) as resp:
        market_data = await resp.json(content_type=None)
        option_chain_df = pd.DataFrame(market_data)
        # print('option_chain_df', option_chain_df)
        if len(option_chain_df) <= 1:
            unix_date = option_chain_df['prevTime'][0]
            # print('unix_date', unix_date)

        return option_chain_df


async def get_prime(exp_date_list, tick, KEY):
    option_chain_df = pd.DataFrame()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for exp in exp_date_list:
            # print(exp)  #
            # print('tick')
            # print(tick)
            exp = datetime.datetime.strftime(exp, '%Y-%m-%d')
            url = f"https://api.marketdata.app/v1/options/chain/{tick}/?token={KEY}&expiration={exp}"
            # print(url)  #
            tasks.append(asyncio.create_task(get_market_data(session, url)))

        solo_exp_chain = await asyncio.gather(*tasks)

        for chain in solo_exp_chain:
            option_chain_df = pd.concat([option_chain_df, chain])

    option_chain_df['updated'] = pd.to_datetime(option_chain_df['updated'], unit='s')
    option_chain_df['EXP_date'] = pd.to_datetime(option_chain_df['expiration'], unit='s', errors='coerce')
    option_chain_df['days_to_exp'] = (option_chain_df['EXP_date'] - option_chain_df['updated']).dt.days
    option_chain_df = option_chain_df.reset_index(drop=True)

    return option_chain_df

def get_df_chains(tick, KEY):
    # print(tick)
    url = f"https://api.marketdata.app/v1/options/expirations/{tick}?token={KEY}"
    response_exp = requests.request("GET", url).json()
    # print('url expirations')
    # print(url)

    try:
        expirations_df = pd.DataFrame(response_exp)
    except:
        expirations_df = pd.DataFrame(response_exp, index=[0])
    # print(expirations_df)
    if len(expirations_df) <= 1:
        unix_date = expirations_df['prevTime'][0]
        # print('unix_date', unix_date)
        counter = 0
        while len(expirations_df) <= 1 or counter < 1:
            url = f"https://api.marketdata.app/v1/options/expirations/{tick}?date={unix_date}&token={KEY}"
            # print(url)
            response_exp = requests.request("GET", url).json()
            start_date = datetime.datetime.fromtimestamp(unix_date)
            try:
                expirations_df = pd.DataFrame(response_exp)
                counter += 1
                break
            except:
                # print('except')
                expirations_df = pd.DataFrame(response_exp, index=[0])
                unix_date = expirations_df['prevTime'][0]
                counter += 1
            if counter > 1:
                break
    # print('test start date', start_date)
    expirations_df['expirations'] = pd.to_datetime(expirations_df['expirations'], format='%Y-%m-%d')
    expirations_df['Days_to_exp'] = (expirations_df['expirations'] - datetime.datetime.today()).dt.days - 1
    expirations_df = expirations_df[expirations_df['Days_to_exp'] > 10]
    expirations_df = expirations_df[expirations_df['Days_to_exp'] <= 355]
    # print('========================')
    # nearest_date = nearest_equal_abs(expirations_df['Days_to_exp'].values.tolist(), nearest_futures_date)
    # print(expirations_df)
    # expirations_df = expirations_df[expirations_df['Days_to_exp'] == nearest_date].reset_index(drop=True)
    option_chain_df = asyncio.run(get_prime(expirations_df['expirations'].tolist(), tick, KEY))

    return option_chain_df

def get_market_data_df(tick, KEY):

    df_chains = get_df_chains(tick, KEY)
    df_chains.to_csv('df_chains.csv', index=False)
    # df_chains = pd.read_csv('df_chains.csv')

    current_price = df_chains['underlyingPrice'].iloc[0]

    # print('df_chains')
    # print(df_chains)

    one_u_dte = df_chains['days_to_exp'][0]
    # print('one_u_dte', one_u_dte)
    local_df = df_chains[df_chains['days_to_exp'] == one_u_dte].reset_index(drop=True)
    # print(local_df)
    atm_strike = nearest_equal_abs(local_df['strike'], current_price)
    local_df = local_df[local_df['strike'] >= atm_strike * 0.5]
    local_df = local_df[local_df['strike'] <= atm_strike * 1.5]
    # print('atm_strike', atm_strike)
    local_df_p = local_df[local_df['side'] == 'put']
    local_df_p = local_df_p.drop_duplicates('strike')
    local_df_c = local_df[local_df['side'] == 'call']
    local_df_c = local_df_c.drop_duplicates('strike')
    atm_iv_p = local_df_p[local_df_p['strike'] == atm_strike]['iv'].values[0]
    atm_iv_c = local_df_c[local_df_c['strike'] == atm_strike]['iv'].values[0]
    # print('atm_iv_p', atm_iv_p)
    # print('atm_iv_c', atm_iv_c)
    put_price_list = []
    call_price_list = []
    for strike in local_df_p['strike']:
        put_price_list.append(price_calc('P', current_price, strike, one_u_dte, atm_iv_p))
    # print('put_price_list', put_price_list)

    for strike in local_df_c['strike']:
        call_price_list.append(price_calc('C', current_price, strike, one_u_dte, atm_iv_c))
    # print('call_price_list', call_price_list)

    local_df_p['atm_price'] = put_price_list
    local_df_c['atm_price'] = call_price_list

    local_df_c[f'{one_u_dte}'] = local_df_c['bid'] - local_df_c['atm_price']
    local_df_p[f'{one_u_dte}'] = local_df_p['bid'] - local_df_p['atm_price']
    finish_call = local_df_c[['strike', f'{one_u_dte}']]
    finish_put = local_df_p[['strike', f'{one_u_dte}']]

    for u_dte in df_chains['days_to_exp'].unique()[1:]:
        # print(u_dte)
        local_df = df_chains[df_chains['days_to_exp'] == u_dte].reset_index(drop=True)
        local_df = local_df[local_df['strike'] >= atm_strike * 0.5]
        local_df = local_df[local_df['strike'] <= atm_strike * 1.5]

        # print(local_df)
        atm_strike = nearest_equal_abs(local_df['strike'], current_price)
        # print('atm_strike', atm_strike)
        local_df_p = local_df[local_df['side'] == 'put']
        local_df_p = local_df_p.drop_duplicates('strike')
        local_df_c = local_df[local_df['side'] == 'call']
        local_df_c = local_df_c.drop_duplicates('strike')
        atm_iv_p = local_df_p[local_df_p['strike'] == atm_strike]['iv'].values[0]
        atm_iv_c = local_df_c[local_df_c['strike'] == atm_strike]['iv'].values[0]
        # print('atm_iv_p', atm_iv_p)
        # print('atm_iv_c', atm_iv_c)
        put_price_list = []
        call_price_list = []
        for strike in local_df_p['strike']:
            put_price_list.append(price_calc('P', current_price, strike, u_dte, atm_iv_p))
        # print('put_price_list', put_price_list)

        for strike in local_df_c['strike']:
            call_price_list.append(price_calc('C', current_price, strike, u_dte, atm_iv_c))
        # print('call_price_list', call_price_list)

        local_df_p['atm_price'] = put_price_list
        local_df_c['atm_price'] = call_price_list

        local_df_c[f'{u_dte}'] = local_df_c['bid'] - local_df_c['atm_price']
        local_df_p[f'{u_dte}'] = local_df_p['bid'] - local_df_p['atm_price']
        local_df_c = local_df_c[['strike', f'{u_dte}']]
        # print(local_df_c)
        finish_put = finish_put.merge(local_df_p[['strike', f'{u_dte}']], on='strike', how='outer')
        finish_call = finish_call.merge(local_df_c[['strike', f'{u_dte}']], on='strike', how='outer')

    # print(finish_put)
    # print(finish_call)

    return finish_put, finish_call

def get_rel_price(company):
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"

    finish_put, finish_call = get_market_data_df(company, KEY)
    finish_put = finish_put.set_index('strike')
    finish_call = finish_call.set_index('strike')

    return finish_put, finish_call

def shortStrangle(underlying,  sigma_call, sigma_put, rate, trials, days_to_expiration,
                  closing_days_array, percentage_array, call_short_strike,
                  call_short_price, put_short_strike, put_short_price, yahoo_stock, instr_type):

    # Data Verification
    # if call_short_strike < put_short_strike:
    #     raise ValueError("Call Strike cannot be less than Put Strike")

    for closing_days in closing_days_array:
        if closing_days > days_to_expiration:
            raise ValueError("Closing days cannot be beyond Days To Expiration.")

    if len(closing_days_array) != len(percentage_array):
        raise ValueError("closing_days_array and percentage_array sizes must be equal.")

    # SIMULATION
    initial_credit = call_short_price + put_short_price  # Credit received from opening trade

    percentage_array = [x / 100 for x in percentage_array]
    min_profit = [initial_credit * x for x in percentage_array]

    strikes = [call_short_strike, put_short_strike]

    # LISTS TO NUMPY ARRAYS CUZ NUMBA HATES LISTS
    strikes = np.array(strikes)
    closing_days_array = np.array(closing_days_array)
    min_profit = np.array(min_profit)

    try:
        pop, pop_error, avg_dtc, avg_dtc_error, cvar = monteCarlo(underlying, rate,  sigma_call, sigma_put, days_to_expiration,
                                                              closing_days_array, trials,
                                                              initial_credit, min_profit, strikes, bsm_debit, yahoo_stock, instr_type)
    except RuntimeError as err:
        print(err.args)

    expected_profit = monteCarlo_exp_return(underlying, rate,  sigma_call, sigma_put, days_to_expiration,
                                                closing_days_array, trials,
                                                initial_credit, min_profit, strikes, bsm_debit, yahoo_stock, instr_type)


    response = {
        "pop": pop,
        'cvar': cvar,
        'exp_return': expected_profit,
        "pop_error": pop_error,
        "avg_dtc": avg_dtc,
        "avg_dtc_error": avg_dtc_error
    }

    return response

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

        instr_type = 'FUT'

        strangle_data = shortStrangle(underlying, sigma_call, sigma_put, rate, trials, days_to_expiration,
                                      [closing_days_array], [percentage_array], call_strike,
                                      call_price, put_strike, put_price, yahoo_stock, instr_type)
        strangle_data = pd.DataFrame(strangle_data)
        strangle_data['Strike Call'] = [call_strike]
        strangle_data['Strike Put'] = [put_strike]
        sum_df = pd.concat([sum_df, strangle_data])

    return sum_df