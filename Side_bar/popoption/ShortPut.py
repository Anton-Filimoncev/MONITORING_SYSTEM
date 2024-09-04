from numba import jit
# from .MonteCarlo_BACKUP import monteCarlo
# from .MonteCarlo_RETURN_BACKUP import monteCarlo_exp_return
# from .Black76_BACKUP import black76Put

from .MonteCarlo import monteCarlo
from .MonteCarlo_RETURN import monteCarlo_exp_return
from .Black76 import black76Put

from .BlackScholes import blackScholesPut
from .Black76 import black76Call
import numpy as np


def bsm_debit(bsm_df):
    instr_type = bsm_df['instr_type'].iloc[0]
    if instr_type == 'FUT':
        P_short_puts = black76Put(bsm_df)
    else:
        P_short_puts = blackScholesPut(sim_price, strike, rate, time_fraction, sigma)

    debit = P_short_puts

    return debit

# def bsm_debit(sim_price, strike, rate, time_fraction, sigma, instr_type):
#     if instr_type == 'FUT':
#         P_short_puts = black76Put(sim_price, strike, rate, time_fraction, sigma)
#     else:
#         P_short_puts = blackScholesPut(sim_price, strike, rate, time_fraction, sigma)
#
#     debit = P_short_puts
#
#     return debit

def shortPut(underlying, sigma, rate, trials, days_to_expiration, days_to_expiration_min, closing_days_array,
             percentage_array, short_strike, short_price, yahoo_stock, instr_type):

    for closing_days in closing_days_array:
        if closing_days > days_to_expiration:
            raise ValueError("Closing days cannot be beyond Days To Expiration.")

    if len(closing_days_array) != len(percentage_array):
        raise ValueError("closing_days_array and percentage_array sizes must be equal.")

    # SIMULATION
    initial_credit = short_price  # Credit received from opening trade

    percentage_array = [x / 100 for x in percentage_array]
    min_profit = [initial_credit * x for x in percentage_array]

    strikes = [short_strike]

    # LISTS TO NUMPY ARRAYS CUZ NUMBA HATES LISTS
    strikes = np.array(strikes)
    closing_days_array = np.array(closing_days_array)
    min_profit = min_profit

    print(underlying, rate, sigma, days_to_expiration_min,
                                                          closing_days_array, trials,
                                                          initial_credit, min_profit, strikes, bsm_debit, yahoo_stock, instr_type)
    # try:
    pop, pop_error,  cvar = monteCarlo(underlying, rate, sigma, days_to_expiration_min,
                                                          closing_days_array, trials,
                                                          initial_credit, min_profit, strikes, bsm_debit, yahoo_stock, instr_type)
    # except RuntimeError as err:
    #     print(err.args)

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
