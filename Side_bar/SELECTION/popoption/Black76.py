from math import log, sqrt, exp, erf
from scipy.stats import norm
from numba import njit
import math
import numpy as np
import pandas as pd


# def black_76(option_type, bsm_df):
#     # bsm_df, 'p', underlying_price, strike_price, years_to_expiration, rate, sigma
#
#     b = 0
#     # -----------
#     # Create preliminary calculations
#     t__sqrt = np.sqrt(bsm_df['dte'])
#     d1 = (np.log(bsm_df['stock_price'] / bsm_df['strike']) + (b + (bsm_df['sigma'] * bsm_df['sigma']) / 2) * bsm_df['dte']) / (bsm_df['sigma'] * t__sqrt)
#     d2 = d1 - bsm_df['sigma'] * t__sqrt
#
#     if option_type == "c":
#         value = bsm_df['stock_price'] * np.exp((b - bsm_df['rate']) * bsm_df['dte']) * norm.cdf(d1) - bsm_df['strike'] * np.exp(-bsm_df['rate'] * bsm_df['dte']) * norm.cdf(d2)
#     else:
#         value = bsm_df['strike'] * np.exp(-bsm_df['rate'] * bsm_df['dte']) * norm.cdf(-d2) - (bsm_df['stock_price'] * np.exp((b - bsm_df['rate']) * bsm_df['dte']) * norm.cdf(-d1))
#
#     return value


def black_76(option_type, bsm_df) -> float:
    is_call_option = False
    if option_type == 'c':
        is_call_option = True

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
    d1 = (np.log(bsm_df['stock_price'] / bsm_df['strike']) + (bsm_df['sigma']**2 / 2) * bsm_df['dte']) / (bsm_df['sigma'] * np.sqrt(bsm_df['dte']))
    d2 = d1 - bsm_df['sigma'] * np.sqrt(bsm_df['dte'])
    discount_factor = np.exp(-bsm_df['rate'] * bsm_df['dte'])

    cp = 1 if is_call_option else -1
    return discount_factor * cp * (bsm_df['stock_price'] * norm.cdf(cp * d1) - bsm_df['strike'] * norm.cdf(cp * d2))



def black76Put(bsm_df):
    # print('years_to_expiration', math.sqrt(years_to_expiration))
    # print('sigma', sigma)
    # print('v * t__sqrt', (sigma * math.sqrt(years_to_expiration)))
    # if years_to_expiration == 0:
    #     years_to_expiration = 0.0000001
    value = black_76('p', bsm_df)
    return value

def black76Call(bsm_df):
    # print('years_to_expiration', math.sqrt(years_to_expiration))
    # print('sigma', sigma)
    # print('v * t__sqrt', (sigma * math.sqrt(years_to_expiration)))
    # if years_to_expiration == 0:
    #     years_to_expiration = 0.0000001
    # value = black_76('c', underlying_price, strike_price, years_to_expiration, rate, sigma)
    value = black_76('c', bsm_df)
    return value

