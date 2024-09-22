import time
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import requests
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import os
import math
import yfinance as yf
import threading
import asyncio
import aiohttp
from scipy.stats import norm
import mibian


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


def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


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


def get_option_cains(contract, exp_date):
    url = f"https://api.marketdata.app/v1/options/quotes/{contract}/?token={KEY}"
    response_exp = requests.request("GET", url, timeout=30)
    chains_df = pd.DataFrame(response_exp.json())
    chains_df['updated'] = pd.to_datetime(chains_df['updated'], unit='s')
    chains_df['days_to_exp'] = (exp_date - chains_df['updated']).dt.days
    return chains_df


def zeta_call(stock, exercise, sigma, rate, dividend, time):
    try:
        time = time / 365
        u = ((rate - dividend) - (math.pow(sigma, 2) / 2)) / (math.pow(sigma, 2))
        l = math.sqrt(((-1) * math.pow(u, 2)) + ((2 * rate) / math.pow(sigma, 2)))
        z = math.log(exercise / stock) / (sigma * math.sqrt(time)) + l * sigma * math.sqrt(time)
        return np.min([math.pow(exercise/stock,u+l)*norm.cdf(-1*z)+math.pow(exercise/stock,u-l)*norm.cdf(-1*z+2*l*sigma*math.sqrt(time)), 1])
    except:
        return None

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

    # main_df, proba_df = calc_vol(to_nln_df, side)
    # print("-" * 50)
    # print(main_df)
    # return_df_put.to_excel('return_df_put.xlsx')
    # proba_df.to_excel('proba_df.xlsx')
# get_z_options()


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
