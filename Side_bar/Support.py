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

def get_contract_calendar(new_position_df, ticker):
    str_exp_date_long = new_position_df['Exp_date_long'].values[0].strftime('%Y-%m-%d').split('-')
    contract_price = strike_price(new_position_df['Strike_long'].values[0])
    contract_long = ticker + str_exp_date_long[0][-2:] + str_exp_date_long[1] + str_exp_date_long[2] + 'P' + contract_price

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
                          sigma_short, sigma_long, days_to_expiration_short, days_to_expiration_long, risk_rate ):
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

def get_abs_return(price_array, type_option, days_to_exp, days_to_exp_short, history_vol, current_price, strike, prime, vol_opt):
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

def expected_return_calc(vol_put_short, vol_put_long, current_price, history_vol, days_to_exp_short, days_to_exp_long, strike_put_long, strike_put_short, prime_put_long, prime_put_short):

    # print('expected_return CALCULATION ...')

    price_array = np.arange(current_price - current_price / 2, current_price + current_price, 0.2)

    short_finish = get_abs_return(price_array, 'Short', days_to_exp_short, days_to_exp_short, history_vol, current_price, strike_put_short,
                                prime_put_short,
                                vol_put_short)

    long_finish = get_abs_return(price_array, 'Long', days_to_exp_long, days_to_exp_short, history_vol, current_price, strike_put_long,
                                 prime_put_long,
                                 vol_put_long)

    expected_return = (short_finish + long_finish) * 100

    return expected_return

def create_new_postion(input_new_df, path, risk_rate):
    new_position_df = pd.DataFrame()
    pos_type = input_new_df['Position_type'].values[0]
    ticker = input_new_df['Symbol'].values[0]
    print('/', ticker, '/')
    print(ticker)
    print(input_new_df['Symbol'].values)
    yahoo_price_df = yf.download(ticker)
    print(yahoo_price_df)
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
        new_position_df['Contract_long'], new_position_df['Contract_short'] = get_contract_calendar(new_position_df, ticker)
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

def update_postion(csv_position_df, pos_type, risk_rate):
    postion_df = pd.read_csv(csv_position_df)
    KEY = credential.password['marketdata_KEY']
    ticker = postion_df['Symbol'].iloc[0]
    yahoo_price_df = yf.download(ticker)
    print('yahoo_price_df')
    print(yahoo_price_df)
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

        wight_df.columns = ['N1', 'V1', 'N2', 'V2']
        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0],  postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, pl, marg, pop_log

    if pos_type == 'ITM Calendar':
        # calculate historical volatility
        log_returns = np.log(yahoo_price_df['Close'] / yahoo_price_df['Close'].shift(1))
        hv = log_returns.rolling(window=252).std() * np.sqrt(252)
        hv = hv[-1]

        quotes_df_long = market_data(postion_df['Contract_long'].iloc[0], KEY)
        quotes_df_short = market_data(postion_df['Contract_short'].iloc[0], KEY)
        postion_df['DAYS_remaining_short'] = (datetime.datetime.strptime(postion_df['Exp_date_short'].iloc[0], '%Y-%m-%d') - datetime.datetime.now()).days
        postion_df['DAYS_remaining_long'] = (datetime.datetime.strptime(postion_df['Exp_date_long'].iloc[0],
                                                                         '%Y-%m-%d') - datetime.datetime.now()).days
        postion_df['DAYS_elapsed_TDE'] = postion_df['DTE'] - postion_df['DAYS_remaining_short']
        postion_df['%_days_elapsed'] = postion_df['DAYS_elapsed_TDE'] / postion_df['DTE']
        postion_df['Prime_now'] = quotes_df_long['mid'] - quotes_df_short['mid']
        postion_df['Cost_to_close_Market_cost'] = (postion_df['Number_pos'] * postion_df['Prime_now']) * 100
        postion_df['Current_Margin'] = postion_df['Open_cost']
        postion_df['Current_PL'] = postion_df['Cost_to_close_Market_cost'] - postion_df['Open_cost']
        postion_df['Current_ROI'] = postion_df['Current_PL'] / postion_df['Current_Margin']
        postion_df['Current_RR_ratio'] = (postion_df['Max_profit'] - postion_df['Current_PL']) / (postion_df['Max_Risk'] + postion_df['Current_PL'])
        postion_df['PL_TDE'] = postion_df['Current_PL'] / postion_df['DAYS_elapsed_TDE']
        postion_df['Leverage'] = ((quotes_df_long['delta']+quotes_df_short['delta']) * 100 * current_price) / postion_df['Current_Margin']

        proba_50, avg_dtc = get_proba_50_calendar(current_price, yahoo_price_df,
                                  postion_df['Strike_long'].iloc[0], postion_df['Start_prime_long'].iloc[0],
                                  postion_df['Strike_short'].iloc[0], postion_df['Start_prime_short'].iloc[0],
                                  quotes_df_short['iv'].iloc[0]*100, quotes_df_long['iv'].iloc[0]*100,
                                  postion_df['DAYS_remaining_short'].iloc[0],
                                  postion_df['DAYS_remaining_long'].iloc[0], risk_rate)

        expected_return = expected_return_calc(quotes_df_short['iv'].iloc[0], quotes_df_long['iv'].iloc[0], current_price,
                                               hv, postion_df['DAYS_remaining_short'].iloc[0], postion_df['DAYS_remaining_long'].iloc[0],
                                               postion_df['Strike_long'].iloc[0], postion_df['Strike_short'].iloc[0],
                                               postion_df['Start_prime_long'].iloc[0], postion_df['Start_prime_short'].iloc[0])

        print('proba_50 ', proba_50, 'expected_return ', expected_return, 'avg_dtc ', avg_dtc,)
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

        postion_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio',]] = postion_df[['%_days_elapsed', 'Current_POP_lognormal', 'Current_ROI', 'Current_RR_ratio']] * 100
        postion_df = postion_df.round(2)

        postion_df.to_csv(csv_position_df)

        current_position = postion_df[['DAYS_remaining_short', 'DAYS_remaining_long', 'DAYS_elapsed_TDE', '%_days_elapsed',
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
        pl, marg, pop_log = postion_df['Current_PL'].iloc[0], postion_df['Current_Margin'].iloc[0],  postion_df['Current_POP_lognormal'].iloc[0]

        return wight_df, pl, marg, pop_log