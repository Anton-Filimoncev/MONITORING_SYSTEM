from numba import jit
from .MonteCarlo import monteCarlo
from .MonteCarlo_RETURN import monteCarlo_exp_return
import time
from .BlackScholes import blackScholesPut
from .Black76 import black76Put
import numpy as np


def bsm_debit(sim_price, strikes, rate, time_fraction, sigma, instr_type):
    if instr_type == 'FUT':
        P_long_puts = black76Put(sim_price, strikes[0], rate, time_fraction, sigma)
    else:
        P_long_puts = blackScholesPut(sim_price, strikes[0], rate, time_fraction, sigma)

    credit = P_long_puts
    debit = -credit

    return debit


def longPut(underlying, sigma, rate, trials, days_to_expiration,
            closing_days_array, multiple_array, long_strike, long_price, yahoo_stock, instr_type):

    for closing_days in closing_days_array:
        if closing_days > days_to_expiration:
            raise ValueError("Closing days cannot be beyond Days To Expiration.")

    if len(closing_days_array) != len(multiple_array):
        raise ValueError("closing_days_array and multiple_array sizes must be equal.")

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
        pop, pop_error, avg_dtc, avg_dtc_error, cvar = monteCarlo(underlying, rate, sigma, days_to_expiration,
                                                              closing_days_array, trials,
                                                              initial_credit, min_profit, strikes, bsm_debit, yahoo_stock, instr_type)
    except RuntimeError as err:
        print(err.args)

    expected_profit = monteCarlo_exp_return(underlying, rate, sigma, days_to_expiration,
                                                closing_days_array, trials,
                                                initial_credit, min_profit, strikes, bsm_debit, yahoo_stock, instr_type)

    response = {
        "pop": pop,
        'cvar': cvar,
        'exp_return': expected_profit,
        "pop_error": pop_error,
        "avg_dtc": avg_dtc,
        "avg_dtc_error": avg_dtc_error
    }

    return response
