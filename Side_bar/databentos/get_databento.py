import databento as db
from dateutil.relativedelta import relativedelta
from datetime import datetime, timezone, timedelta
from collections.abc import Iterable
import databento as db
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.optimize import root_scalar
from scipy.stats import norm
import math
from scipy.stats import norm
import yfinance as yf

_DEBUG = False

def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


def decode_nearest_dte():
    start_time = datetime.now()
    start_time_str = start_time.strftime('%Y-%m-%d')
    end_time = datetime.now() + relativedelta(days=35)
    end_time_str = end_time.strftime('%Y-%m-%d')

    week_num_list = ['week1', 'week2', 'week3', 'week4']

    week_days = ['1MON', '1TUE', '1WED', '1THU', '1FRI', '2MON', '2TUE', '2WED', '2THU', '2FRI', '3MON', '3TUE',
                 '3WED', '3THU', '3FRI', '4MON', '4TUE', '4WED', '4THU', '4FRI', '5MON', '5TUE', '5WED', '5THU', '5FRI']
    total_dates = {'Date': [], 'dte': []}  # 'dte':[], 'Week':[], 'Week_day':[]
    for week_day in week_days:
        m_date = pd.date_range(start_time_str, end_time_str, freq=f'WOM-{week_day}').values[0]

        total_dates['Date'].append(m_date)
        total_dates['dte'].append((pd.to_datetime(m_date) - start_time).days + 1)
    #         total_dates['Week'].append(w_name)
    #         total_dates['Week_day'].append(week_day)

    return pd.DataFrame(total_dates)


def _debug(debug_input):
    if (__name__ is "__main__") and (_DEBUG is True):
        print(debug_input)


# This class defines the Exception that gets thrown when invalid input is placed into the GBS function
class GBS_InputError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)


# This class contains the limits on inputs for GBS models
# It is not intended to be part of this module's public interface
class _GBS_Limits:
    # An GBS model will return an error if an out-of-bound input is input
    MAX32 = 2147483248.0

    MIN_T = 1.0 / 1000.0  # requires some time left before expiration
    MIN_X = 0.01
    MIN_FS = 0.01

    # Volatility smaller than 0.5% causes American Options calculations
    # to fail (Number to large errors).
    # GBS() should be OK with any positive number. Since vols less
    # than 0.5% are expected to be extremely rare, and most likely bad inputs,
    # _gbs() is assigned this limit too
    MIN_V = 0.0000005

    MAX_T = 100
    MAX_X = MAX32
    MAX_FS = MAX32

    # Asian Option limits
    # maximum TA is time to expiration for the option
    MIN_TA = 0

    # This model will work with higher values for b, r, and V. However, such values are extremely uncommon.
    # To catch some common errors, interest rates and volatility is capped to 100%
    # This reason for 1 (100%) is mostly to cause the library to throw an exceptions
    # if a value like 15% is entered as 15 rather than 0.15)
    MIN_b = -1
    MIN_r = -1

    MAX_b = 1
    MAX_r = 1
    MAX_V = 1


# ------------------------------
# This function verifies that the Call/Put indicator is correctly entered
def _test_option_type(option_type):
    if (option_type != "c") and (option_type != "p"):
        raise GBS_InputError("Invalid Input option_type ({0}). Acceptable value are: c, p".format(option_type))


# ------------------------------
# This function makes sure inputs are OK
# It throws an exception if there is a failure
def _gbs_test_inputs(option_type, fs, x, t, r, b, v):
    # -----------
    # Test inputs are reasonable
    _test_option_type(option_type)

    if (x < _GBS_Limits.MIN_X) or (x > _GBS_Limits.MAX_X):
        raise GBS_InputError(
            "Invalid Input Strike Price (X). Acceptable range for inputs is {1} to {2}".format(x, _GBS_Limits.MIN_X,
                                                                                               _GBS_Limits.MAX_X))

    if (fs < _GBS_Limits.MIN_FS) or (fs > _GBS_Limits.MAX_FS):
        raise GBS_InputError(
            "Invalid Input Forward/Spot Price (FS). Acceptable range for inputs is {1} to {2}".format(fs,
                                                                                                      _GBS_Limits.MIN_FS,
                                                                                                      _GBS_Limits.MAX_FS))

    if (t < _GBS_Limits.MIN_T) or (t > _GBS_Limits.MAX_T):
        raise GBS_InputError(
            "Invalid Input Time (T = {0}). Acceptable range for inputs is {1} to {2}".format(t, _GBS_Limits.MIN_T,
                                                                                             _GBS_Limits.MAX_T))

    if (b < _GBS_Limits.MIN_b) or (b > _GBS_Limits.MAX_b):
        raise GBS_InputError(
            "Invalid Input Cost of Carry (b = {0}). Acceptable range for inputs is {1} to {2}".format(b,
                                                                                                      _GBS_Limits.MIN_b,
                                                                                                      _GBS_Limits.MAX_b))

    if (r < _GBS_Limits.MIN_r) or (r > _GBS_Limits.MAX_r):
        raise GBS_InputError(
            "Invalid Input Risk Free Rate (r = {0}). Acceptable range for inputs is {1} to {2}".format(r,
                                                                                                       _GBS_Limits.MIN_r,
                                                                                                       _GBS_Limits.MAX_r))

    if (v < _GBS_Limits.MIN_V) or (v > _GBS_Limits.MAX_V):
        raise GBS_InputError(
            "Invalid Input Implied Volatility (V = {0}). Acceptable range for inputs is {1} to {2}".format(v,
                                                                                                           _GBS_Limits.MIN_V,
                                                                                                           _GBS_Limits.MAX_V))


# The primary class for calculating Generalized Black Scholes option prices and deltas
# It is not intended to be part of this module's public interface

# Inputs: option_type = "p" or "c", fs = price of underlying, x = strike, t = time to expiration, r = risk free rate
#         b = cost of carry, v = implied volatility
# Outputs: value, delta, gamma, theta, vega, rho
def _gbs(option_type, fs, x, t, r, b, v):
    _debug("Debugging Information: _gbs()")
    # -----------
    # Test Inputs (throwing an exception on failure)
    _gbs_test_inputs(option_type, fs, x, t, r, b, v)

    # -----------
    # Create preliminary calculations
    t__sqrt = math.sqrt(t)
    d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
    d2 = d1 - v * t__sqrt

    if option_type == "c":
        # it's a call
        _debug("     Call Option")
        value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
        delta = math.exp((b - r) * t) * norm.cdf(d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) - (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(d1) - r * x * math.exp(-r * t) * norm.cdf(d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = x * t * math.exp(-r * t) * norm.cdf(d2)
    else:
        # it's a put
        _debug("     Put Option")
        value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))
        delta = -math.exp((b - r) * t) * norm.cdf(-d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) + (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(-d1) + r * x * math.exp(-r * t) * norm.cdf(-d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = -x * t * math.exp(-r * t) * norm.cdf(-d2)

    _debug("     d1= {0}\n     d2 = {1}".format(d1, d2))
    _debug("     delta = {0}\n     gamma = {1}\n     theta = {2}\n     vega = {3}\n     rho={4}".format(delta, gamma,
                                                                                                        theta, vega,
                                                                                                        rho))

    return value, delta, gamma, theta, vega, rho


def black_76(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.05,
    is_call_option: bool = True,
) -> float:
    """
    Calculate an option price from the Black-76 model.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Option strike price.
    T : float
        Time to expiration in years.
    sigma : float
        Implied volatility, annualized.
    r : float, default 0.05
        The risk free interest rate, annualized.
    is_call_option : bool, default True
        Flag to indicate the option is a call or put.

    Returns
    -------
    float

    """
    d1 = (np.log(S / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount_factor = np.exp(-r * T)

    cp = 1 if is_call_option else -1
    return discount_factor * cp * (S * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))



def find_sigma(
    row: pd.Series,
) -> float:
    """
    Find the roots of the Black-76 model by varying sigma, implied volatility.

    This function is for use with `pandas.Dataframe.apply`. Each row should contain
    a column for "strike_price", "years_to_expiration", "instrument_class", "midprice",
    and "underlying_price",

    If the optimization fails, `numpy.nan` is returned.

    Parameters
    ----------
    row : pd.Series
        A series of data to process.

    Returns
    -------
    float | numpy.nan

    """
    # row = row.iloc[0]
    print('midprice', row["close"])
    print('underlying_price', row["underlying_price"])
    print('strike_price', row["strike"])
    print('years_to_expiration', row["years_to_expiration"])
    print('instrument_class', row["instrument_class"])
    def f(sigma: float) -> float:
        return row["close"] - black_76(
            S=row["underlying_price"],
            K=row["strike"],
            T=row["years_to_expiration"],
            sigma=sigma,
            is_call_option=row["instrument_class"] == "C",
        )

    result = root_scalar(f, x0=0.1, x1=0.5)
    if result.converged:
        return result.root
    # print(
    #     f"Could not find sigma for {row['raw_symbol']} with midprice {row['midprice']}",
    # )
    return np.nan


def get_bento_data(ticker, ticker_b, nearest_dte, side, path_bento):
    if datetime.now().weekday() != 0:
        start_date_req = (datetime.now() - relativedelta(hours=37)).strftime("%Y-%m-%dT%H:%M:%S")
        exp_start_date_req = (datetime.now() - relativedelta(days=2)).strftime("%Y-%m-%d")
    else:
        start_date_req = (datetime.now() - relativedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        exp_start_date_req = (datetime.now() - relativedelta(days=4)).strftime("%Y-%m-%d")


    underlying_price = yf.download(ticker)['Close'].iloc[-1]

    print('path_bento', path_bento)


    if nearest_dte < 35:
        weekly_symb = pd.read_excel(f'{path_bento}Weekly_symb.xlsx')[ticker_b]
        print('weekly_symb', weekly_symb)
        weekly_df = decode_nearest_dte()

        near_dte = nearest_equal_abs(weekly_df['dte'].values.tolist(), nearest_dte)
        look_index = weekly_df[weekly_df['dte'] == near_dte].index
        ticker_b = weekly_symb.iloc[look_index].values[0]

    # Create a live client
    live_client = db.Live(key="db-FRv6T7L3EUpBkE9d4vQxD8iK89Keq")
    # Subscribe with a specified start time for intraday replay
    live_client.subscribe(
        dataset="GLBX.MDP3",
        schema="ohlcv-1h",
        symbols=f"{ticker_b}.OPT",
        stype_in="parent",
        start=start_date_req,
       #end='2023-08-25T13:30'
    )
    # Now, we will open a file for writing and start streaming
    with open(f"{path_bento}{ticker_b}.dbn", "wb") as output:
        live_client.add_stream(output)
        live_client.start()
        # We will wait for 5 seconds before closing the connection
        live_client.block_for_close(timeout=5)
    # Finally, we will open the DBN file

    dbn_store = db.from_dbn(f"{path_bento}{ticker_b}.dbn")
    full_chains = dbn_store.to_df(schema="ohlcv-1h")
    print('full_chains', full_chains)
    full_chains[['symb', 'strikus']] = full_chains['symbol'].str.split(' ', n=1, expand=True)

    full_chains.reset_index(drop=True).to_excel('full_chains.xlsx')

    print(datetime.now())

    nearest_dte_symb = (datetime.now() + relativedelta(days=nearest_dte)).month

    month_code_dict = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X",
                       12: "Z"}

    nearest_dte_symb_num = month_code_dict[nearest_dte_symb]
    print('nearest_dte_symb_num')
    print(nearest_dte_symb_num)
    cur_year = str(datetime.now().year)[-1]
    print('cur_year')
    print(cur_year)
    needed_contract = ticker_b + nearest_dte_symb_num + cur_year
    needed_chain = full_chains[full_chains['symb'] == needed_contract]
    print('needed_chain')
    print(needed_chain)
    needed_chain = needed_chain.reset_index(drop=True)

    print('underlying_price11111', str(int(underlying_price)))
    len_price = len(str(int(underlying_price)))

    max_len_strike = 0
    for i in range(len(needed_chain)):
        cur_len = len(needed_chain.iloc[i]['strikus'])
        if cur_len > max_len_strike:
            max_len_strike = cur_len

    for i, row in needed_chain.iterrows():

        if len(row['strikus']) < max_len_strike or len(needed_chain['strikus'].max()) == len(
                needed_chain['strikus'].min()):
            if row['strikus'][1:len_price + 1][0] == '0':
                needed_chain.loc[i, 'strike'] = float(
                    row['strikus'][2:len_price + 2] + '.' + row['strikus'][len_price + 2:])
            else:
                needed_chain.loc[i, 'strike'] = float(
                    row['strikus'][1:len_price + 1] + '.' + row['strikus'][len_price + 1:])
        else:
            needed_chain.loc[i, 'strike'] = float(
                row['strikus'][1:len_price + 2] + '.' + row['strikus'][len_price + 2:])
        needed_chain.loc[i, 'instrument_class'] = row['strikus'][0]

    print('needed_chain')
    print(needed_chain)

    # =========================get exp date========================
    client = db.Historical(key="db-FRv6T7L3EUpBkE9d4vQxD8iK89Keq")
    trades = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[f"{needed_chain['symbol'].iloc[0]}"],
        stype_in="raw_symbol",
        schema="definition",
        start=exp_start_date_req,
    )

    # Then, convert to a DataFrame
    df = trades.to_df()
    print(df)
    needed_exp_date = df['expiration'].iloc[0]

    print('needed_exp_date', needed_exp_date)

    dte = (needed_exp_date - datetime.now(timezone.utc)).days
    print('dte', dte)

    needed_chain['years_to_expiration'] = dte / 365
    needed_chain['underlying_price'] = underlying_price

    if side != 'Strangle':
        needed_chain = needed_chain[needed_chain['instrument_class'] == side]

    iv = needed_chain.apply(find_sigma, axis=1,)
    print('iv')
    print(iv)
    needed_chain["iv"] = iv
    needed_chain = needed_chain[needed_chain["iv"] < 1]
    needed_chain = needed_chain[needed_chain["iv"] > 0]

    print('needed_chain')
    print(needed_chain)

    for i, row in needed_chain.iterrows():
        strike = row['strike']
        iv_loop = row['iv']
        local_side = row['instrument_class']
        value, delta, gamma, theta, vega, rho = _gbs(local_side.lower(), underlying_price, strike, dte / 365, 0.04, 0.04, iv_loop)

        needed_chain.loc[i, 'delta'] = delta
        needed_chain.loc[i, 'gamma'] = gamma
        needed_chain.loc[i, 'theta'] = theta
        needed_chain.loc[i, 'vega'] = vega
        needed_chain.loc[i, 'rho'] = rho


    needed_chain['bid'] = needed_chain['close']
    needed_chain['ask'] = needed_chain['close']

    needed_chain = needed_chain.drop(columns=['rtype', 'publisher_id', 'instrument_id'])

    needed_chain = needed_chain.drop_duplicates(['symbol'], keep='last').reset_index(drop=True)
    needed_chain = needed_chain.dropna()
    print('------------needed_chain-------')
    print(needed_chain)

    needed_exp_date = needed_exp_date.strftime("%Y-%m-%d")
    needed_exp_date = datetime.strptime(needed_exp_date, '%Y-%m-%d')

    return needed_exp_date, needed_chain





