import streamlit as st
import pandas as pd
from Road_map import *

st.set_page_config(page_icon='💵', page_title="Monitoring" )
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
    ('Portfolio', 'Market View', 'Put Sell', 'Call Sell', 'Strangle', 'OTM Calendar', 'ITM Calendar', 'git test') # 'Call Monitoring',
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

# =====================================   Put Sell
if infoType == 'git test':
    from git import Repo
    PATH_OF_GIT_REPO = "https://github.com/Anton-Filimoncev/MONITORING_SYSTEM.git"
    COMMIT_MESSAGE = 'comment from python script'

    #
    def git_push():
        # try:
        repo = Repo(search_parent_directories=True)
        repo.git.add(update=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name='origin')
        origin.push()
        # except:
        #     print('Some error occured while pushing the code')


    git_push()








