from numba import jit
from .MonteCarlo import monteCarlo
from .MonteCarlo_RETURN import monteCarlo_exp_return
import time
from .BlackScholes import blackScholesCall
from .Black76 import black76Put
from .Black76 import black76Call
import numpy as np


# def bsm_debit(sim_price, strikes, rate, time_fraction, sigma, instr_type):
#     if instr_type == 'FUT':
#         P_long_calls = black76Call(sim_price, strikes[0], rate, time_fraction, sigma)
#     else:
#         P_long_calls = blackScholesCall(sim_price, strikes[0], rate, time_fraction, sigma)
#
#     credit = P_long_calls
#     debit = -credit
#
#     return debit

def bsm_debit(bsm_df):
    instr_type = bsm_df['instr_type'].iloc[0]
    if instr_type == 'FUT':
        P_long_calls = black76Call(bsm_df)
    else:
        P_long_calls = blackScholesPut(sim_price, strike, rate, time_fraction, sigma)

    debit = -P_long_calls

    return debit


def longCall(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min,
             closing_days_array, multiple_array, long_strike, long_price, yahoo_stock, instr_type):

    for closing_days in closing_days_array:
        if closing_days > days_to_expiration:
            print(closing_days)
            print(days_to_expiration)
            raise ValueError("Closing days cannot be beyond Days To Expiration.")

    if len(closing_days_array) != len(multiple_array):
        raise ValueError("closing_days_array and percentage_array sizes must be equal.")

    # SIMULATION
    initial_debit = long_price  # Debit paid from opening trade
    initial_credit = -1 * initial_debit

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