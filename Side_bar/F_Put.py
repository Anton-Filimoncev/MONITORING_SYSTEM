import numpy as np
import streamlit as st
import pandas as pd
import glob
from .Support import *
import datetime


def f_put():

    col1, col2, col3 = st.columns([6,1,1])
    with col1:
        st.header('FUTURES Put')
    with col2:
        risk_rate = st.number_input('Risk Rate', step=0.5, format="%.1f", min_value=1., max_value=5000., value=4.8)

    with col3:
        refresh_btn = st.button("Refresh ALL", type="primary")
        #     refresh_btn = True
        # else:
        #     refresh_btn = False
    print('refresh_btn222')
    print(refresh_btn)
    # download all open position
    path = 'Side_bar/side_bar_data/futures/put/'
    filenames = glob.glob(path + "*.csv")
    # ============================================
    # ============================================     add new position
    # ============================================
    with st.expander('Add New F. Put Position'):
        col11, col12, col13, col14 = st.columns(4)
        with col11:
            start_date_o_p = st.date_input('Start date', datetime.datetime.now())
            end_date_stat = st.date_input('EXP date')
        with col12:
            ticker = st.text_input('Ticker', '')
            dividend = st.number_input('Dividend', step=0.01, format="%.2f", min_value=0., max_value=5000., value=0.0)
            try:
                start_b_a_price_yahoo = yf.download(ticker)['Close'].iloc[-1]
            except:
                start_b_a_price_yahoo = 0.
            start_b_a_price = st.number_input('Start BA Price', step=0.1, format="%.2f", min_value=0., max_value=50000., value=start_b_a_price_yahoo)
        with col13:
            prime_o_p = st.number_input('Start Prime', step=0.01, format="%.2f", min_value=0., max_value=5000.)
            strike_o_p = st.number_input('Strike', step=0.5, format="%.1f", min_value=1., max_value=5000., value=100.)
            delta_o_p = st.number_input('Delta', step=0.5, format="%.1f", min_value=1., max_value=5000., value=100.)

        with col14:
            num_pos_o_p = st.number_input('Number of Positions', min_value=1, max_value=365, value=1)
            multiplicator_o_p = st.number_input('Multiplicator', min_value=1, max_value=100000, value=100)
            size_o_p = st.number_input('Size', min_value=1, max_value=100000, value=100)
            commission_o_p = st.number_input('Commission', step=0.1, format="%.1f", min_value=0., max_value=5000., value=3.4)
            margin_o_p = st.number_input('Margin', step=0.5, format="%.1f", min_value=0., max_value=55000., value=6000.)


    # ============================================

        if st.button("Open", type="primary"):
            input_new_df = pd.DataFrame({
                'Position_type': ['F. Put'],
                'Symbol': [ticker],
                'Start_date_o_p': [start_date_o_p],
                'Exp_date_o_p': [end_date_stat],
                'Strike_o_p': [strike_o_p],
                'Number_pos_o_p': [num_pos_o_p],
                'Prime_o_p': [prime_o_p],
                'Commission_o_p': [commission_o_p],
                'Margin_o_p': [margin_o_p],
                'Delta_o_p': [delta_o_p],
                'Dividend': [dividend],
                'Start_price': [start_b_a_price],
                'Multiplicator_o_p': [multiplicator_o_p],
                'Size_o_p': [size_o_p],
            })
            print('input_new_df')
            print(input_new_df)

            create_new_postion(input_new_df, path, risk_rate)
            st.success('Position is OPEN waiting for $$$')


    progress_text = "Loading..."
    my_bar = st.progress(0, text=progress_text)
    # show all open position
    print(filenames)

    for num, csv_position_df in enumerate(filenames):
        print('num', num)
        print('csv_position_df', csv_position_df)
        tick = get_tick_from_csv_name(csv_position_df)
        print('tick', tick)
        pos_type = 'F. Put'
        if refresh_btn:
            update_postion(csv_position_df, pos_type, risk_rate)
            # st.success('All data is updated!')
        # else:
        #     print('elseeeeeeeeeee')
        position_df, greeks_df, pl, marg, pop_log = return_postion(csv_position_df, pos_type, risk_rate)

        with st.expander(tick + (' .' * 5) + 'PL: ' + str(pl) + (' .' * 5) + 'Margin: ' + str(marg) + (
                ' .' * 5) + 'POP lognormal: ' + str(pop_log)):
            st.text(tick)
            st.dataframe(position_df, hide_index=True, column_config=None)
            st.dataframe(greeks_df, hide_index=True, column_config=None)

        my_bar.progress(int((100 / len(filenames)) * (num + 1)), text=progress_text)
    my_bar.empty()


