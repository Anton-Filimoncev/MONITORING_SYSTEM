import numpy as np
import streamlit as st
import pandas as pd
import glob
from .Support import *
import datetime

def put_sell():
    col1, col2 = st.columns([6,1])
    with col1:
        st.header('Put Sell')
    with col2:
        refresh_btn = st.button("Refresh ALL")
    if refresh_btn:
        st.success('All data is updated!')
    # ============================================
    # ============================================     add new position
    # ============================================
    with st.expander('Add New Put Sell Position'):
        col11, col12, col13, col14 = st.columns(4)
        with col11:
            start_date_o_p = st.date_input('Start date', datetime.datetime.now())
            end_date_stat = st.date_input('EXP date')
        with col12:
            strike_o_p = st.number_input('STRIKE', step=0.5, format="%.1f", min_value=1., max_value=5000., value=100.)
        with col13:
            prime_o_p = st.number_input('Prime', step=0.01, format="%.1f", min_value=0., max_value=5000.)
        with col14:
            num_pos_o_p = st.number_input('Number of positions', min_value=1, max_value=365, value=1)


    # ============================================

        if st.button("Open", type="primary"):
            create_new_postion('Put Sell', start_date_o_p, end_date_stat, strike_o_p, num_pos_o_p, prime_o_p)
            st.success('Position is OPEN waiting for $$$')

    # download all open position
    path = 'Side_bar/side_bar_data/put_sell/'
    filenames = glob.glob(path + "*.csv")

    # show all open position
    for csv_position_df in filenames:
        tick = get_tick_from_csv_name(csv_position_df)
        postion_df = pd.read_csv(csv_position_df)
        with st.expander(tick + (' .'*20) + 'PnL: ' + postion_df['DKS'].iloc[0]):
            st.text(tick)
            st.dataframe(postion_df, hide_index=True)


