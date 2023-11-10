import streamlit as st
import pandas as pd
import glob
from .Support import *

def put_sell():
    # add new position
    with st.expander('Add New Put Sell Position'):
        st.write('Coming soon...')

    # download all open position
    path = 'Side_bar/side_bar_data/put_sell/'
    filenames = glob.glob(path + "*.csv")

    # show all open position
    for csv_position_df in filenames:
        tick = get_tick_from_csv_name(csv_position_df)
        st.text(tick)
        st.dataframe(pd.read_csv(csv_position_df))
