from numba import jit
from .MonteCarlo import monteCarlo
from .MonteCarlo_RETURN import monteCarlo_exp_return
import time
from .BlackScholes import blackScholesPut
import numpy as np


def bsm_debit(bsm_df):
    debit = bsm_df['stock_price']
    return debit


def shortStock(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
            closing_days_array, multiple_array, long_strike, long_price, yahoo_stock, instr_type):

    for closing_days in closing_days_array:
        if closing_days > days_to_expiration:
            raise ValueError("Closing days cannot be beyond Days To Expiration.")

    if len(closing_days_array) != len(multiple_array):
        raise ValueError("closing_days_array and multiple_array sizes must be equal.")

    # SIMULATION
    initial_debit = underlying  # Debit paid from opening trade
    initial_credit = initial_debit

    min_profit = [initial_debit * x for x in multiple_array]

    strikes = [long_strike]

    # LISTS TO NUMPY ARRAYS CUZ NUMBA HATES LISTS
    strikes = np.array(strikes)
    closing_days_array = np.array(closing_days_array)
    min_profit = np.array(min_profit)

    try:
        pop, pop_error, cvar = monteCarlo(underlying, rate, sigma, days_to_expiration_min,
                                                              closing_days_array, trials,
                                                              initial_credit, min_profit, strikes, bsm_debit, yahoo_stock, instr_type)
    except RuntimeError as err:
        print(err.args)

    profit_dte = np.max([days_to_expiration - days_to_expiration_min, 1])

    expected_profit = monteCarlo_exp_return(underlying, rate, sigma, profit_dte,
                                                closing_days_array, trials,
                                                initial_credit, min_profit, strikes, bsm_debit, yahoo_stock, instr_type)

    response = {
        "pop": pop,
        'cvar': cvar,
        'exp_return': expected_profit,
        "pop_error": pop_error,
    }

    return response
