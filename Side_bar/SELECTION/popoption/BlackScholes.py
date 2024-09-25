from numba import jit
import numpy as np
from math import log, sqrt, exp, erf
from scipy import special


def blackScholesPut(bsm_df):

    print('bsm_df')
    print(bsm_df)
    # if tt == 0 and (bsm_df['stock_price'] / bsm_df['strike'] > 1):
    #     p = 0
    # elif tt == 0 and (bsm_df['stock_price'] / bsm_df['strike'] < 1):
    #     p = bsm_df['strike'] - bsm_df['stock_price']
    # elif tt == 0 and (bsm_df['stock_price'] / bsm_df['strike'] == 1):
    #     p = 0

    d1 = (np.log(bsm_df['stock_price'] / bsm_df['strike']) + (bsm_df['rate'] + (1 / 2) * bsm_df['sigma'] ** 2) * bsm_df['dte']) / (bsm_df['sigma'] * np.sqrt(bsm_df['dte']))
    d2 = d1 - bsm_df['sigma'] * np.sqrt(bsm_df['dte'])
    c = bsm_df['stock_price'] * ((1.0 + special.erf(d1 / np.sqrt(2.0))) / 2.0) - bsm_df['strike'] * np.exp(-bsm_df['rate'] * bsm_df['dte']) * ((1.0 + special.erf(d2 / np.sqrt(2.0))) / 2.0)
    p = bsm_df['strike'] * np.exp(-bsm_df['rate'] * bsm_df['dte']) - bsm_df['stock_price'] + c

    return p


def blackScholesCall(s, k, rr, tt, sd):
    if tt == 0 and (s / k > 1):
        c = s - k
    elif tt == 0 and (s / k < 1):
        c = 0
    elif tt == 0 and (s / k == 1):
        c = 0
    else:
        d1 = (log(s / k) + (rr + (1 / 2) * sd ** 2) * tt) / (sd * sqrt(tt))
        d2 = d1 - sd * sqrt(tt)
        c = s * ((1.0 + erf(d1 / sqrt(2.0))) / 2.0) - k * exp(-rr * tt) * ((1.0 + erf(d2 / sqrt(2.0))) / 2.0)

    #     d1 = (np.log(underlyingPrice / strikePrice) + (interestRate + (volatility**2) / 2) * daysToExpiration) / a
    #     d2 = d1 - a
    #     price = underlyingPrice * norm.cdf(d1) - strikePrice * e** (-interestRate * daysToExpiration) * norm.cdf(d2)

    return c

#
# def blackScholesPut(s, k, rr, tt, sd):
#
#     if tt == 0 and (s / k > 1):
#         p = 0
#     elif tt == 0 and (s / k < 1):
#         p = k - s
#     elif tt == 0 and (s / k == 1):
#         p = 0
#     else:
#         d1 = (log(s / k) + (rr + (1 / 2) * sd ** 2) * tt) / (sd * sqrt(tt))
#         d2 = d1 - sd * sqrt(tt)
#         c = s * ((1.0 + erf(d1 / sqrt(2.0))) / 2.0) - k * exp(-rr * tt) * ((1.0 + erf(d2 / sqrt(2.0))) / 2.0)
#         p = k * exp(-rr * tt) - s + c
#
#     return p
#
#
# def blackScholesCall(s, k, rr, tt, sd):
#     if tt == 0 and (s / k > 1):
#         c = s - k
#     elif tt == 0 and (s / k < 1):
#         c = 0
#     elif tt == 0 and (s / k == 1):
#         c = 0
#     else:
#         d1 = (log(s / k) + (rr + (1 / 2) * sd ** 2) * tt) / (sd * sqrt(tt))
#         d2 = d1 - sd * sqrt(tt)
#         c = s * ((1.0 + erf(d1 / sqrt(2.0))) / 2.0) - k * exp(-rr * tt) * ((1.0 + erf(d2 / sqrt(2.0))) / 2.0)
#
#     #     d1 = (np.log(underlyingPrice / strikePrice) + (interestRate + (volatility**2) / 2) * daysToExpiration) / a
#     #     d2 = d1 - a
#     #     price = underlyingPrice * norm.cdf(d1) - strikePrice * e** (-interestRate * daysToExpiration) * norm.cdf(d2)
#
#     return c

