import numpy as np
import streamlit as st
import pandas as pd
import glob
from .Support import *
import datetime

def itm_calendar():
    col1, col2, col3 = st.columns([6,1,1])
    with col1:
        st.header('ITM Calendar')
    with col2:
        risk_rate = st.number_input('Risk Rate', step=0.5, format="%.1f", min_value=1., max_value=5000., value=4.8)
    with col3:
        refresh_btn = st.button("Refresh ALL")
    if refresh_btn:
        st.success('All data is updated!')

    # download all open position
    path = 'Side_bar/side_bar_data/itm_calendar/'
    filenames = glob.glob(path + "*.csv")
    # ============================================
    # ============================================     add new position
    # ============================================
    with st.expander('Add New ITM Calendar Position'):
        col11, col12, col13, col14 = st.columns(4)
        with col11:
            start_date_o_p = st.date_input('Start date', datetime.datetime.now())
            end_date_long_stat = st.date_input('EXP date Long')
            end_date_short_stat = st.date_input('EXP date Short')
        with col12:
            ticker = st.text_input('Ticker', '')
            dividend = st.number_input('Dividend', step=0.01, format="%.2f", min_value=0., max_value=5000., value=0.01)
            # try:
            #     start_b_a_price_yahoo = yf.download(ticker)['Close'].iloc[-1]
            # except:
            start_b_a_price_yahoo = 0.
            start_b_a_price = st.number_input('Start BA Price', step=0.1, format="%.2f", min_value=0., max_value=50000., value=start_b_a_price_yahoo)
        with col13:
            prime_o_p = st.number_input('Prime', step=0.01, format="%.2f", min_value=0., max_value=5000.)
            strike_long_o_p = st.number_input('Strike Long', step=0.5, format="%.1f", min_value=1., max_value=5000., value=100.)
            strike_short_o_p = st.number_input('Strike Short', step=0.5, format="%.1f", min_value=1., max_value=5000., value=100.)
            delta_o_p = st.number_input('Delta', step=0.5, format="%.1f", min_value=1., max_value=5000., value=100.)

        with col14:
            num_pos_o_p = st.number_input('Number of positions', min_value=1, max_value=365, value=1)
            commission_o_p = st.number_input('Commission', step=0.1, format="%.1f", min_value=0., max_value=5000., value=6.04)
            try:
                start_marg = num_pos_o_p * prime_o_p * 100
                margin_o_p = st.number_input('Margin', step=0.5, format="%.1f", min_value=0., max_value=55000., value=start_marg)
            except:
                margin_o_p = st.number_input('Margin', step=0.5, format="%.1f", min_value=0., max_value=55000.,
                                             value=1200.)

        # ============================================

        if st.button("Open", type="primary"):
            input_new_df = pd.DataFrame({
                'Position_type': ['ITM Calendar'],
                'Symbol': [ticker],
                'Start_date_o_p': [start_date_o_p],
                'Exp_date_long_o_p': [end_date_long_stat],
                'Exp_date_short_o_p': [end_date_short_stat],
                'Strike_long_o_p': [strike_long_o_p],
                'Strike_short_o_p': [strike_short_o_p],
                'Number_pos_o_p': [num_pos_o_p],
                'Prime_o_p': [prime_o_p],
                'Commission_o_p': [commission_o_p],
                'Margin_o_p': [margin_o_p],
                'Delta_o_p': [delta_o_p],
                'Dividend': [dividend],
                'Start_price': [start_b_a_price],
            })
            print('input_new_df')
            print(input_new_df)

            create_new_postion(input_new_df, path, risk_rate)
            st.success('Position is OPEN waiting for $$$')



    # show all open position
    for csv_position_df in filenames[:1]:
        tick = get_tick_from_csv_name(csv_position_df)
        pos_type = 'ITM Calendar'
        postion_df, pl, marg, pop_log = update_postion(csv_position_df, pos_type, risk_rate)
        print('aaaaaaaaaaaaaaaa', pl, marg, pop_log)
        with st.expander(tick + (' .'*5) + 'PL: ' + str(pl) + (' .'*5) + 'Margin: ' + str(marg) + (' .'*5) + 'POP lognormal: ' + str(pop_log)):
            st.text(tick)
            st.dataframe(postion_df, hide_index=True, column_config=None)


