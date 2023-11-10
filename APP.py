import streamlit as st
import pandas as pd
from Road_map import *


# ---- HIDE STREAMLIT STYLE ----
# # MainMenu {visibility: hidden;}
# header {visibility: hidden;}
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

infoType = st.sidebar.radio(
    "Choose an info type",
    ('Portfolio', 'Market View', 'Put Sell', 'Call Sell', 'Strangle', 'OTM Calendar', 'ITM Calendar') # 'Call Monitoring',
)


# =====================================   Portfolio
if infoType == 'Portfolio':
    portfolio()

# =====================================   Portfolio
if infoType == 'Market View':
    market_view()

# =====================================   Put Sell
if infoType == 'Put Sell':
    put_sell()

# =====================================   Put Sell
if infoType == 'Call Sell':
    call_sell()

# =====================================   Put Sell
if infoType == 'Strangle':
    strangle()

# =====================================   Put Sell
if infoType == 'OTM Calendar':
    otm_calendar()

# =====================================   Put Sell
if infoType == 'ITM Calendar':
    itm_calendar()








