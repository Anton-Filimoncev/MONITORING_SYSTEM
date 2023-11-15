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
from .popoption.ShortCall import shortCall
from .popoption.ShortStrangle import shortStrangle
from .popoption.PutCalendar import putCalendar



def get_tick_from_csv_name(csv_position_df):
    ticker_name = csv_position_df.split("/")[-1].split(".")[0]
    return ticker_name


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
                          sigma_short, sigma_long, days_to_expiration_short, days_to_expiration_long, ):
    rate = 4.6
    closing_days_array = [days_to_expiration_short]
    percentage_array = [30]
    trials = 3000

    proba_50 = putCalendar(current_price, sigma_short, sigma_long, rate, trials, days_to_expiration_short,
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

def create_new_postion(input_new_df, path, risk_rate):
    new_position_df = pd.DataFrame()
    pos_type = input_new_df['Position_type'].values[0]
    ticker = input_new_df['Symbol'].values[0]
    yahoo_price_df = yf.download(ticker)
    start_price = yahoo_price_df['Close'].iloc[-1]
    KEY = credential.password['marketdata_KEY']
    log_returns = np.log(yahoo_price_df["Close"] / yahoo_price_df["Close"].shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    hv = log_returns.rolling(window=200).std() * np.sqrt(200)
    hv = hv.iloc[-1]

    if pos_type == 'Put Sell':

        new_position_df['Symbol'] = [ticker]
        new_position_df['Start_date'] = input_new_df['Start_date_o_p']
        new_position_df['Exp_date'] = input_new_df['Exp_date_o_p']
        new_position_df['Strike'] = input_new_df['Strike_o_p']
        new_position_df['Number_pos'] = input_new_df['Number_pos_o_p'] * -1
        new_position_df['Start_prime'] = input_new_df['Prime_o_p']
        new_position_df['Contract'] = get_contract(new_position_df, ticker)
        new_position_df['Start_price'] =  input_new_df['Start_price']
        new_position_df['Dividend'] = input_new_df['Dividend']
        new_position_df['Commission'] = input_new_df['Commission_o_p']
        new_position_df['Margin_start'] = input_new_df['Margin_o_p']
        new_position_df['DTE'] = (new_position_df['Exp_date'] - new_position_df['Start_date']).values[0].days
        new_position_df['Open_cost'] = (new_position_df['Number_pos'].iloc[0] * new_position_df['Start_prime'])*100
        new_position_df['Start_delta'] = input_new_df['Delta_o_p']
        new_position_df['HV_200'] = hv
        new_position_df['Price_2s_down'] = start_price - (2 * start_price * hv*(math.sqrt(new_position_df['DTE'].iloc[0]/365)))
        new_position_df['Max_profit'] = new_position_df['Open_cost'].abs()
        new_position_df['Max_Risk'] = np.max([new_position_df['Strike'].iloc[0] * 0.1, (new_position_df['Price_2s_down'].iloc[0] - (new_position_df['Price_2s_down'].iloc[0]-new_position_df['Strike'].iloc[0]))*0.2]) * new_position_df['Open_cost'].abs() * 100
        new_position_df['BE_lower'] = new_position_df['Strike'] - new_position_df['Start_prime']
        new_position_df['RR_RATIO'] = new_position_df['Max_profit'] / new_position_df['Max_Risk']
        new_position_df['MP/DTE'] = new_position_df['Max_profit'] / new_position_df['DTE']
        new_position_df['Profit_Target'] = new_position_df['Max_profit'] * 0.5 - (new_position_df['Commission'])

        quotes_df = market_data(new_position_df['Contract'].iloc[0], KEY)

        pop_50, expected_profit, avg_dtc = get_proba_50_put(start_price, yahoo_price_df, new_position_df['Strike'].iloc[0],
                                                            new_position_df['Start_prime'].iloc[0], quotes_df['iv'].iloc[0],
                                                            new_position_df['DTE'].iloc[0], risk_rate)

        new_position_df['Expected_Return'] = expected_profit
        new_position_df['DTE_Target'] = int(avg_dtc[0])
        new_position_df['POP_Monte_start_50'] = pop_50
        new_position_df['Plan_ROC '] = new_position_df['Profit_Target'] / new_position_df['Margin_start']
        new_position_df['ROC_DAY_target'] = new_position_df['Plan_ROC '] / new_position_df['DTE_Target']
        #
        new_position_df.to_csv(f"{path}{ticker}_{new_position_df['Start_date'].values[0].strftime('%Y-%m-%d')}.csv", index=False)

    return

def market_data(contract, KEY):
    url = f"https://api.marketdata.app/v1/options/quotes/{contract}/?&token={KEY}"
    response_exp = requests.request("GET", url, timeout=30)
    chains_df = pd.DataFrame(response_exp.json())
    return chains_df

def update_postion(csv_position_df, pos_type, risk_rate):
    postion_df = pd.read_csv(csv_position_df)
    KEY = credential.password['marketdata_KEY']
    ticker = postion_df['Symbol'].iloc[0]
    yahoo_price_df = yf.download(ticker)
    current_price = yahoo_price_df['Close'].iloc[-1]
    if pos_type == 'Put Sell':
        quotes_df = market_data(postion_df['Contract'].iloc[0], KEY)
        print(quotes_df)
        postion_df['DAYS_remaining'] = (datetime.datetime.strptime(postion_df['Exp_date'].iloc[0], '%Y-%m-%d') - datetime.datetime.now()).days
        postion_df['DAYS_elapsed_TDE'] = postion_df['DTE'] - postion_df['DAYS_remaining']
        postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE']
        postion_df['Prime_now'] = quotes_df['ask']
        postion_df['Cost_to_close_Market_cost'] = (postion_df['Number_pos'] * quotes_df['ask']) * 100
        postion_df['Current_Margin'] = np.max([0.2 * current_price - (current_price - postion_df['Strike']), 0.1 * postion_df['Strike']]) * postion_df['Number_pos'].abs() * 100
        postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
        postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
        postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (postion_df['Max_Risk'] + postion_df['Current_PL'])
        postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
        postion_df['Leverage'] = (quotes_df['delta'] * 100 * current_price) / postion_df['Current_Margin']
        postion_df['Margin_2S_Down'] = np.max([0.1 * postion_df['Strike'], postion_df['Price_2s_down'] * 0.2 - (postion_df['Price_2s_down'] - postion_df['Strike'])]) * postion_df['Number_pos'].abs() * 100

        pop_50, expected_profit, avg_dtc = get_proba_50_put(current_price, yahoo_price_df, postion_df['Strike'].iloc[0],
                                                            postion_df['Start_prime'].iloc[0], quotes_df['iv'].iloc[0]*100,
                                                            postion_df['DAYS_remaining'].iloc[0], risk_rate)
        print('pop_50 ', pop_50, 'expected_profit ', expected_profit, 'avg_dtc ', avg_dtc,)
        postion_df['POP_50'] = pop_50
        postion_df['Current_expected_return'] = expected_profit
        postion_df['DTE_Target'] = avg_dtc[0]

        P_below = stats.norm.cdf(
            (np.log(postion_df['BE_lower'].iloc[0] / current_price) / (
                    postion_df['HV_200'].iloc[0] * math.sqrt(postion_df['DAYS_remaining'].iloc[0] / 365))))

        postion_df['Current_POP_lognormal'] = 1 - P_below

        postion_df['MC_2S_Down'] = calculate_option_price('P', postion_df['Price_2s_down'].iloc[0],
                                  postion_df['Strike'].iloc[0], risk_rate, postion_df['HV_200'].iloc[0],
                                  postion_df['DAYS_remaining']) * postion_df['Number_pos'].abs() * 100

        postion_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio',]] = postion_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100
        postion_df = postion_df.round(2)

        postion_df.to_csv(csv_position_df)

        current_position = postion_df[['DAYS_remaining', 'DAYS_elapsed_TDE', '%_days_elapsed',
                           'Cost_to_close_Market_cost', 'Current_Margin', 'Current_PL', 'Current_expected_return',
                           'Current_POP_lognormal', 'POP_50', 'Current_ROI', 'Current_RR_ratio', 'PL_TDE', 'Leverage',
                           'MC_2S_Down', 'Margin_2S_Down']].T


    current_position['Values'] = current_position[0]
    current_position['Name'] = current_position.index.values.tolist()
    current_position = current_position[['Name', 'Values']]
    current_position1 = current_position[int(len(current_position) / 2):].reset_index(drop=True)
    current_position2 = current_position[:int(len(current_position) / 2)].reset_index(drop=True)

    wight_df = pd.concat([current_position1, current_position2], axis=1, )
    print(wight_df)

    wight_df.columns = ['N1', 'V1', 'N2', 'V2']
    pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0],  postion_df['Current_POP_lognormal'].iloc[0]

    return wight_df, pl, marg, pop_log

    # DAYS remaining