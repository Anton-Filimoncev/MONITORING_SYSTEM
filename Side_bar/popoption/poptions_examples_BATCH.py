import pandas as pd
import poptions
import yfinance as yf
from dateutil.relativedelta import relativedelta
import datetime

# Confused on what these variables mean? Read the README file!

# Entering existing trades: See the READMe file.

# ###############################################################

# ticker = 'GOOG'
# yahoo_stock = yf.download(ticker)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]  # или ввести значение
#
# call_long_strike = 150
# call_long_price = 28.45
# call_short_strike = 150.0
# call_short_price = 19.6
# rate = 4.9
# sigma_short = 28.7  # ATM
# sigma_long = 28.99  # ATM
# days_to_expiration_short = 409
# days_to_expiration_long = 773
# percentage_array = [30]
# closing_days_array = [409]
# trials = 2000
#
# print("Call Calendar: ", poptions.callCalendar(underlying, sigma_short, sigma_long, rate, trials, days_to_expiration_short,
#                 days_to_expiration_long, closing_days_array, percentage_array, call_long_strike,
#                 call_long_price, call_short_strike, call_short_price, yahoo_stock))

# ###############################################################
# ticker = 'AMZN'
# yahoo_stock = yf.download(ticker)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]
# long_strike = 115
# long_price = 33.55
# rate = 4.6
# sigma = 36
# days_to_expiration = 114
# multiple_array = [1]        # Multiple of debit paid that will trigger the position to close
# closing_days_array = [114]
# trials = 2000
#
# print("Long Call: ", poptions.longCall(underlying, sigma, rate, trials, days_to_expiration,
#                               closing_days_array, multiple_array, long_strike, long_price, yahoo_stock))

# ###############################################################
#
# ticker = 'CRM'
# yahoo_stock = yf.download(ticker)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]
# long_strike = 170
# long_price = 9.16
# rate = 4.6
# sigma = 36
# days_to_expiration = 256
# multiple_array = [1]        # Multiple of debit paid that will trigger the position to close
# closing_days_array = [256]
# trials = 2000
#
# print("Long Put: ", poptions.longPut(underlying, sigma, rate, trials, days_to_expiration,
#                             closing_days_array, multiple_array, long_strike, long_price, yahoo_stock))
################################################################

# ticker = 'KRE'
# yahoo_stock = yf.download(ticker)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]
# long_call_strike = 61
# long_call_price = 1.18
# short_put_strike = 38
# short_put_price = 2
# rate = 4
#
# sigma = 30   # (sigma_long_call + sigma_short_put) / 2
#
# days_to_expiration = 338
# multiple_array = [1]        # Multiple of debit paid that will trigger the position to close
# closing_days_array = [179]
# trials = 2000
#
#
# print("Risk Reversal: ", poptions.riskReversal(underlying, sigma, rate, trials, days_to_expiration,
#                               closing_days_array, multiple_array, long_call_strike, long_call_price, short_put_strike,
#                                            short_put_price, yahoo_stock))

# # ###############################################################
#
# ticker = 'GOOG'
# yahoo_stock = yf.download(ticker)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]  # или ввести значение
#
# put_long_strike = 150
# put_long_price = 28.45
# put_short_strike = 150.0
# put_short_price = 19.6
# rate = 4.9
# sigma_short = 28.7  # ATM
# sigma_long = 28.99  # ATM
# days_to_expiration_short = 409
# days_to_expiration_long = 773
# percentage_array = [30]
# closing_days_array = [409]
# trials = 2000
#
# print("Put Calendar: ", poptions.putCalendar(underlying, sigma_short, sigma_long, rate, trials, days_to_expiration_short,
#                     days_to_expiration_long, closing_days_array, percentage_array, put_long_strike,
#                     put_long_price, put_short_strike, put_short_price, yahoo_stock))

# ###############################################################

input_data = pd.read_excel('short_strangl_USO.xlsx')
ticker = 'USO'
yahoo_stock_native = yf.download(ticker)

cvar_list = []
pop_list = []

for num, row in input_data.iterrows():
    print(row)
    try:
        start_date = row['Start_Date']
        back_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') - relativedelta(years=3)).strftime('%Y-%m-%d')

        yahoo_stock = yahoo_stock_native[back_date:start_date]

        underlying = row['Current_Price']  # или ввести значение
        call_short_strike = row['Strike_Call']
        call_short_price = row['Call_price']
        put_short_strike = row['Strike_Put']
        put_short_price = row['Put_price']
        rate = 3.5
        sigma = (row['Put_iv'] + row['Call_iv']) / 2
        days_to_expiration = (row['EXP_date'] - datetime.datetime.strptime(start_date, '%Y-%m-%d')).days
        percentage_array = [50]
        closing_days_array = [days_to_expiration]
        trials = 2000

        print('call_short_strike', call_short_strike)
        print('call_short_price', call_short_price)
        print('put_short_strike', put_short_strike)
        print('put_short_price', put_short_price)
        print('sigma', sigma)
        print('days_to_expiration', days_to_expiration)

        result = poptions.shortStrangle(underlying, sigma, rate, trials, days_to_expiration,
                            closing_days_array, percentage_array, call_short_strike,
                            call_short_price, put_short_strike, put_short_price, yahoo_stock)
        print("Short Strangle: ", result)

        cvar_list.append(result['cvar'])
        pop_list.append(result['pop'])
    except:
        cvar_list.append(0)
        pop_list.append(0)


input_data['cvar'] = cvar_list
input_data['pop'] = pop_list
input_data.to_excel('data.xlsx')
# ###############################################################

# ticker = 'KRE'
# underlying = 71.72
# short_strike = 90
# short_price = 1.16
# rate = 0
# sigma = 55
# days_to_expiration = 53
# percentage_array = [50]
# closing_days_array = [36]
# trials = 2000
#
# yahoo_stock = yf.download(ticker)['2018-01-01':]
#
# print("Short Call: ", poptions.shortCall(underlying, sigma, rate, trials, days_to_expiration,
#                                 closing_days_array, percentage_array, short_strike, short_price, yahoo_stock))

################################################################

# ticker = 'SIL'
# yahoo_stock = yf.download(ticker)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]
# short_strike = 26
# short_price = 3.7
# rate = 4.6
# sigma = 33.8
# days_to_expiration = 193
# percentage_array = [50]
# closing_days_array = [193]
# trials = 2000
#
#
# print("Short Put: ", poptions.shortPut(underlying, sigma, rate, trials, days_to_expiration,
#                               closing_days_array, percentage_array, short_strike, short_price, yahoo_stock))



################################################################

# ticker = 'TFC'
# underlying = 28.23     # Current underlying price
# short_strike = 30      # Short strike price
# short_price = 3.17    # Short call price
# long_strike = 40
# long_price = 0.75
# rate = 4        # Annualized risk-free rate as a percentage (e.g. 1 year US Treasury Bill rate)
# sigma = 37.7        # Implied Volatility as a percentage
# days_to_expiration = 266     # Calendar days left till expiration
# percentage_array = [20, 30, 40]  # Percentage of maximum profit that will trigger the position to close
# closing_days_array = [100, 133, 266]       # Max calendar days passed until position is closed
# trials = 2000       # Number of independent trials
#
# yahoo_stock = yf.download(ticker)['2018-01-01':]
#
# print("Call Credit Spread: ", poptions.callCreditSpread(underlying, sigma, rate, trials, days_to_expiration,
#                                                         closing_days_array, percentage_array, short_strike,
#                                                         short_price, long_strike, long_price, yahoo_stock))

###############################################################

# underlying = 36.73
# short_strike = 28
# short_price = 0.88
# long_strike = 18
# long_price = 0.18
# rate = 0
# sigma = 71.4
# days_to_expiration = 51
# percentage_array = [75, 50]
# closing_days_array = [21, 24]
# trials = 2000
#
# print("Put Credit Spread: ", poptions.putCreditSpread(underlying, sigma, rate, trials, days_to_expiration,
#                                              closing_days_array, percentage_array, short_strike,
#                                              short_price, long_strike, long_price))
#
# ###############################################################
#
# underlying = 123
# short_strike = 120
# short_price = 6.9
# long_strike = 110
# long_price = 14.2
# rate = 0
# sigma = 29.2
# days_to_expiration = 48
# percentage_array = [20]
# closing_days_array = [48]
# trials = 2000
# #
# print("Call Debit Spread: ", poptions.callDebitSpread(underlying, sigma, rate, trials, days_to_expiration,
#                                              closing_days_array, percentage_array, short_strike,
#                                              short_price, long_strike, long_price))
#
# ###############################################################
#
# underlying = 24.87
# short_strike = 26
# short_price = 3.55
# long_strike = 28
# long_price = 4.9
# rate = 0
# sigma = 79.7
# days_to_expiration = 48
# percentage_array = [50]
# closing_days_array = [48]
# trials = 2000
#
# print("Put Debit Spread: ", poptions.putDebitSpread(underlying, sigma, rate, trials, days_to_expiration,
#                                            closing_days_array, percentage_array, short_strike,
#                                            short_price, long_strike, long_price))
#
# ###############################################################
#
# underlying = 15
# short_strike = 12.5
# short_price = 1.4
# rate = 0
# sigma = 117
# days_to_expiration = 45
# percentage_array = [50]
# closing_days_array = [21]
# trials = 2000
#
# print("Short Put: ", poptions.shortPut(underlying, sigma, rate, trials, days_to_expiration,
#                               closing_days_array, percentage_array, short_strike, short_price))
#
# ###############################################################
#
# underlying = 71.72
# short_strike = 90
# short_price = 1.16
# rate = 0
# sigma = 55
# days_to_expiration = 53
# percentage_array = [50]
# closing_days_array = [36]
# trials = 2000
#
# print("Short Call: ", poptions.shortCall(underlying, sigma, rate, trials, days_to_expiration,
#                                 closing_days_array, percentage_array, short_strike, short_price))
#

#

#
# ###############################################################
#
# underlying = 71.72
# short_strike = 90
# short_price = 1.16
# rate = 0
# sigma = 55
# days_to_expiration = 53
# percentage_array = [50]
# closing_days_array = [36]
# trials = 2000
#
# print("Covered Call: ", poptions.coveredCall(underlying, sigma, rate, trials, days_to_expiration,
#                                     closing_days_array, percentage_array, short_strike, short_price))
#
# ###############################################################
#
# underlying = 205
# rate = 0
# sigma = 68.6
# days_to_expiration = 25
# percentage_array = [50]
# closing_days_array = [25]
# trials = 2000
#
# ## PUT SIDE ###
# put_short_strike = 170
# put_short_price = 3.25
# put_long_strike = 165
# put_long_price = 2.48
#
# ## CALL SIDE ###
# call_short_strike = 250
# call_short_price = 2.82
# call_long_strike = 255
# call_long_price = 2.34
#
# print("Iron Condor: ", poptions.ironCondor(underlying, sigma, rate, trials, days_to_expiration,
#                  closing_days_array, percentage_array, put_short_strike,
#                  put_short_price, put_long_strike, put_long_price, call_short_strike,
#                  call_short_price, call_long_strike, call_long_price))
#

