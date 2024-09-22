from numba import jit
from math import log, sqrt, exp, erf
import mibian as mb


def blackScholesPut(s, k, rr, tt, sd):
    if tt == 0 and (s / k > 1):
        p = 0
    elif tt == 0 and (s / k < 1):
        p = k - s
    elif tt == 0 and (s / k == 1):
        p = 0
    else:
        d1 = (log(s / k) + (rr + (1 / 2) * sd ** 2) * tt) / (sd * sqrt(tt))
        d2 = d1 - sd * sqrt(tt)
        c = s * ((1.0 + erf(d1 / sqrt(2.0))) / 2.0) - k * exp(-rr * tt) * ((1.0 + erf(d2 / sqrt(2.0))) / 2.0)
        p = k * exp(-rr * tt) - s + c

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

    return c

# sim_price, strikes[0], rate, time_fraction, sigma
sim_price = 67.56
strike = 82.5
rate = 4
time_fraction = 272
sigma = 30
r = 0

dt = 1 / 365  # 365 calendar days in a year
time_fraction_2 = dt * (time_fraction - r)
# print(time_fraction_2)

print(blackScholesCall(sim_price, strike, 0.04, time_fraction_2, sigma/100))
print(mb.BS([sim_price, strike, rate, time_fraction], volatility=sigma).callPrice)

