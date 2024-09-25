from numba import jit
from .MonteCarlo_da_bat import monteCarlo
from .MonteCarlo_da_bat_RETURN import monteCarlo_return
import time
from .BlackScholes import blackScholesPut
import numpy as np


def bsm_debit(sim_price, strikes, rate, time_fraction_1, time_fraction_2_long, sigma_long_1, sigma_long_2, sigma_short, long_1_count,
                long_2_count, short_count):

    P_long_1_puts = blackScholesPut(sim_price, strikes[0], rate, time_fraction_1, sigma_long_1)
    P_long_2_puts = blackScholesPut(sim_price, strikes[1], rate, time_fraction_2_long, sigma_long_2)
    P_short_puts = blackScholesPut(sim_price, strikes[2], rate, time_fraction_1, sigma_short)

    debit = (P_long_1_puts*long_1_count) + (P_long_2_puts*long_2_count) - (P_short_puts*short_count)
    # debit = P_long_puts - P_short_puts

    return debit


def fut_DA_BUT(underlying, sigma_long_1, sigma_long_2, sigma_short, rate, trials,
                days_to_expiration_1, days_to_expiration_2_long, closing_days_array, percentage_array, long_1_strike,
                long_2_strike, short_strike, long_1_price, long_2_price,  short_price, yahoo_stock, long_1_count,
                long_2_count, short_count):
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
    initial_debit = abs((long_1_price*long_1_count) + (long_2_price*long_2_count) - (short_price*short_count))  # Debit paid from opening trade
    # initial_credit = -1 * initial_debit
    initial_credit = (short_price*short_count) - (long_1_price*long_1_count) - (long_2_price*long_2_count)
    max_profit = initial_debit
    percentage_type = 'Initial'
    pop_from = 'Initial'
    if pop_from == 'margin':
        max_profit = (0.2 * put_short_strike) * (short_count-long_count)
        percentage_type = 'Margin'

    percentage_array = [x / 100 for x in percentage_array]
    min_profit = [max_profit * x for x in percentage_array]

    strikes = [long_1_strike, long_2_strike, short_strike]

    # LISTS TO NUMPY ARRAYS CUZ NUMBA HATES LISTS
    strikes = np.array(strikes)
    closing_days_array = np.array(closing_days_array)
    min_profit = np.array(min_profit)

    try:
        pop, pop_error, avg_dtc, avg_dtc_error, cvar = monteCarlo(underlying, rate, sigma_long_1, sigma_long_2, sigma_short,
                                                            days_to_expiration_1, days_to_expiration_2_long, closing_days_array, trials, initial_credit,
                                                            min_profit, strikes, bsm_debit, yahoo_stock, long_1_count,
                                                            long_2_count, short_count)
    except RuntimeError as err:
        print(err.args)

    expected_profit = monteCarlo_return(underlying, rate, sigma_long_1, sigma_long_2, sigma_short,
                            days_to_expiration_1, days_to_expiration_2_long, closing_days_array, trials, initial_credit,
                            min_profit, strikes, bsm_debit, yahoo_stock, long_1_count,
                            long_2_count, short_count)

    response = {
        "pop": pop,
        'cvar': cvar,
        'exp_return': expected_profit,
        "pop_error": pop_error,
        "avg_dtc": avg_dtc,
        "avg_dtc_error": avg_dtc_error
    }
    return response, max_profit, percentage_type
