import poptions
import yfinance as yf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Confused on what these variables mean? Read the README file!

# Entering existing trades: See the READMe file.

# ###############################################################

ticker = 'ZC=F'
yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]

underlying = yahoo_stock['Close'].iloc[-1]  # или ввести значение

start_underlying = 405
short_strike = 380
short_price = 29
sigma = 23
rate = 4

days_to_expiration = 46
percentage_array = [30]
closing_days_array = [days_to_expiration]
trials = 2000

print("coveredShortCall: ", poptions.coveredShortCall(underlying, start_underlying, sigma, rate, trials,
                days_to_expiration, closing_days_array, percentage_array, short_strike, short_price, yahoo_stock))
print("shortCall: ", poptions.shortCall(underlying, sigma, rate, trials,
                days_to_expiration, closing_days_array, percentage_array, short_strike, short_price, yahoo_stock))
# shortCall
# ###############################################################

# ticker = 'ES=F'
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]  # или ввести значение
#
# long_1_strike = 5190
# long_1_price = 51.75
# sigma_long_1 = 13
#
# long_2_strike = 5290
# long_2_price = 47.75
# sigma_long_2 = 12
#
# short_strike = 5240
# short_price = 25.75
# sigma_short = 14
# rate = 4
#
# days_to_expiration_1 = 2
# days_to_expiration_2_long = 37
# percentage_array = [30]
# closing_days_array = [days_to_expiration_1]
# trials = 2000
#
# long_1_count = 1
# long_2_count = 1
# short_count = 2
#
# print("FUTURES DA BUT: ", poptions.fut_DA_BUT(underlying, sigma_long_1, sigma_long_2, sigma_short, rate, trials,
#                 days_to_expiration_1, days_to_expiration_2_long, closing_days_array, percentage_array, long_1_strike,
#                 long_2_strike, short_strike, long_1_price, long_2_price,  short_price, yahoo_stock, long_1_count,
#                 long_2_count, short_count))

# ###############################################################

# ticker = 'GC=F'
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]  # или ввести значение
#
# long_strike = 2275
# long_price = 28.9
# sigma_long = 13.6
#
# short_1_strike = 2255
# short_1_price = 23.1
# sigma_short_1 = 13.5
#
# short_2_strike = 2215
# short_2_price = 14.7
# sigma_short_2 = 13.7
# rate = 4
#
# days_to_expiration = 105
# percentage_array = [50]
# closing_days_array = [105]
# trials = 2000
#
# long_count = 1
# short_1_count = 1
# short_2_count = 2
#
# print("FUTURES Ratio 1 1 2: ", poptions.futRatio_1_1_2(underlying, sigma_long, sigma_short_1, sigma_short_2, rate, trials,
#                 days_to_expiration, closing_days_array, percentage_array, long_strike,
#                 short_1_strike, short_2_strike, long_price, short_1_price,  short_2_price, yahoo_stock, long_count,
#                 short_1_count, short_2_count))

# ###############################################################

# ticker = '^SPX'
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]  # или ввести значение
#
# call_long_strike = 5375
# call_long_price = 92.7
# call_short_strike = 5375
# call_short_price = 43.9
# rate = 4
# sigma_short = 12.1
# sigma_long = 12.4
# days_to_expiration_short = 8
# days_to_expiration_long = 36
# percentage_array = [30]
# closing_days_array = [8]
# trials = 5000
#
# print("Call Calendar: ", poptions.callCalendar(underlying, sigma_short, sigma_long, rate, trials, days_to_expiration_short,
#                 days_to_expiration_long, closing_days_array, percentage_array, call_long_strike,
#                 call_short_strike, call_long_price, call_short_price, yahoo_stock))

# ###############################################################
# ticker = 'AMZN'
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
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
# ticker = 'GC=F'
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]
# long_strike = 2275
# long_price = 28.9
# rate = 4
# sigma = 13.6
#
# multiple_array = [1]        # Multiple of debit paid that will trigger the position to close
# days_to_expiration = 105
# percentage_array = [50]
# closing_days_array = [105]
# trials = 2000
#
#
# print("Long Put: ", poptions.longPut(underlying, sigma, rate, trials, days_to_expiration,
#                             closing_days_array, multiple_array, long_strike, long_price, yahoo_stock))
################################################################

# ticker = 'KRE'
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
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
# ticker = 'SPY'
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]  # или ввести значение
#
# put_long_strike = 470
# put_long_price = 14.8
# put_short_strike = 445
# put_short_price = 9 * 2
# rate = 4
# sigma_short = 16  # ATM
# sigma_long = 18  # ATM
# days_to_expiration_short = 38
# days_to_expiration_long = 101
# percentage_array = [30]
# closing_days_array = [38]
# trials = 2000
#
# print("Put Calendar: ", poptions.putCalendar(underlying, sigma_short, sigma_long, rate, trials, days_to_expiration_short,
#                     days_to_expiration_long, closing_days_array, percentage_array, put_long_strike,
#                     put_long_price, put_short_strike, put_short_price, yahoo_stock))

# ###############################################################
#
# ticker = 'DHR'
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]  # или ввести значение
#
# call_short_strike = 320
# call_short_price = 3.6
# put_short_strike = 190
# put_short_price = 4.25
# rate = 4
# sigma = 28
# days_to_expiration = 262
# percentage_array = [50]
# closing_days_array = [124]
# trials = 2000
#
# print("Short Strangle: ", poptions.shortStrangle(underlying, sigma, rate, trials, days_to_expiration,
#                     closing_days_array, percentage_array, call_short_strike,
#                     call_short_price, put_short_strike, put_short_price, yahoo_stock))

# ###############################################################

# ticker = 'ES=F'
# underlying = 5483.5
# short_strike = 6000.0
# short_price = 5.35
# rate = 4.8
# sigma = 11.22
# days_to_expiration = 90
# percentage_array = [50]
# closing_days_array = [90]
# trials = 2000
#
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
#
# print("Short Call: ", poptions.shortCall(underlying, sigma, rate, trials, days_to_expiration,
#                                 closing_days_array, percentage_array, short_strike, short_price, yahoo_stock))

################################################################

# ticker = 'CL=F'
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
#
# underlying = yahoo_stock['Close'].iloc[-1]
# short_strike = 78
# short_price = 0.47
# rate = 4
# sigma = 24.5
# days_to_expiration = 12
# percentage_array = [50]
# closing_days_array = [12]
# trials = 2000
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
# yahoo_stock = yf.download(ticker, progress=False)['2018-01-01':]
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

