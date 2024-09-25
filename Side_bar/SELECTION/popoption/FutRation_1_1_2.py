from numba import jit
from .MonteCarlo_1_1_2 import monteCarlo
from .MonteCarlo_1_1_2_RETURN import monteCarlo_return
import time
from .BlackScholes import blackScholesPut
from .BlackScholes import blackScholesCall
from .Black76_112 import black76Put
from .Black76_112 import black76Call
import numpy as np


def bsm_debit(sim_price, strikes, rate, time_fraction, sigma_long, sigma_short_1, sigma_short_2, long_count,
              short_1_count, short_2_count, side, instr_type):
    if side == 'P':
        if instr_type == 'FUT':
            P_long_puts = black76Put(sim_price, strikes[0], rate, time_fraction, sigma_long)
            P_short_1_puts = black76Put(sim_price, strikes[1], rate, time_fraction, sigma_short_1)
            P_short_2_puts = black76Put(sim_price, strikes[2], rate, time_fraction, sigma_short_2)
        else:
            P_long_puts = blackScholesPut(sim_price, strikes[0], rate, time_fraction, sigma_long)
            P_short_1_puts = blackScholesPut(sim_price, strikes[1], rate, time_fraction, sigma_short_1)
            P_short_2_puts = blackScholesPut(sim_price, strikes[2], rate, time_fraction, sigma_short_2)

    if side == 'C':
        if instr_type == 'FUT':
            P_long_puts = black76Call(sim_price, strikes[0], rate, time_fraction, sigma_long)
            P_short_1_puts = black76Call(sim_price, strikes[1], rate, time_fraction, sigma_short_1)
            P_short_2_puts = black76Call(sim_price, strikes[2], rate, time_fraction, sigma_short_2)
        else:
            P_long_puts = blackScholesCall(sim_price, strikes[0], rate, time_fraction, sigma_long)
            P_short_1_puts = blackScholesCall(sim_price, strikes[1], rate, time_fraction, sigma_short_1)
            P_short_2_puts = blackScholesCall(sim_price, strikes[2], rate, time_fraction, sigma_short_2)


    debit = (P_long_puts*long_count) - (P_short_1_puts*short_1_count) - (P_short_2_puts*short_2_count)
    # debit = P_long_puts - P_short_puts

    return debit


def futRatio_1_1_2(underlying, sigma_long, sigma_short_1, sigma_short_2, rate, trials,
                days_to_expiration, closing_days_array, percentage_array, long_strike,
                short_1_strike, short_2_strike, long_price, short_1_price,  short_2_price, yahoo_stock, long_count,
                short_1_count, short_2_count, side, instr_type):
    # Data Verification
    # if put_long_price <= put_short_price:
    #     raise ValueError("Long price cannot be less than or equal to Short price")

    # if short_strike >= long_strike:
    #     raise ValueError("Short strike cannot be greater than or equal to Long strike")

    # for closing_days in closing_days_array:
    #     if closing_days > days_to_expiration_short:
    #         raise ValueError("Closing days cannot be beyond Days To Expiration.")

    if len(closing_days_array) != len(percentage_array):
        raise ValueError("closing_days_array and percentage_array sizes must be equal.")

    # SIMULATION
    initial_debit = abs((long_price*long_count) - (short_1_price*short_1_count) - (short_2_price*short_2_count))  # Debit paid from opening trade
    # initial_credit = -1 * initial_debit
    initial_credit = (short_1_price*short_1_count) + (short_2_price*short_2_count) - (long_price*long_count)
    max_profit = initial_debit
    percentage_type = 'Initial'
    pop_from = 'Initial'
    if pop_from == 'margin':
        max_profit = (0.2 * put_short_strike) * (short_count-long_count)
        percentage_type = 'Margin'

    percentage_array = [x / 100 for x in percentage_array]
    min_profit = [max_profit * x for x in percentage_array]

    strikes = [long_strike, short_1_strike, short_2_strike]

    # LISTS TO NUMPY ARRAYS CUZ NUMBA HATES LISTS
    strikes = np.array(strikes)
    closing_days_array = np.array(closing_days_array)
    min_profit = np.array(min_profit)

    try:
        pop, pop_error, avg_dtc, avg_dtc_error, cvar = monteCarlo(underlying, rate, sigma_long, sigma_short_1, sigma_short_2,
                                                            days_to_expiration, closing_days_array, trials, initial_credit,
                                                            min_profit, strikes, bsm_debit, yahoo_stock, long_count,
                                                            short_1_count, short_2_count, side, instr_type)
    except RuntimeError as err:
        print(err.args)

    expected_profit = monteCarlo_return(underlying, rate, sigma_long, sigma_short_1, sigma_short_2,
                                                            days_to_expiration, closing_days_array, trials, initial_credit,
                                                            min_profit, strikes, bsm_debit, yahoo_stock, long_count,
                                                            short_1_count, short_2_count, side, instr_type)

    response = {
        "pop": pop,
        'cvar': cvar,
        'exp_return': expected_profit,
        "pop_error": pop_error,
        "avg_dtc": avg_dtc,
        "avg_dtc_error": avg_dtc_error
    }
    return response
