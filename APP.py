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


infoType = 'PORTFOLIO'

infoType = st.sidebar.checkbox("PORTFOLIO")


with st.sidebar.expander('FUTURES'):
    infoType_F_P = st.checkbox("F. Put")
    infoType_F_C = st.checkbox("F. Call")
    infoType_F_ST = st.checkbox("F. Strangle")
    # infoType = st.radio(
    #     "Choose an info type",
    #     ('F. Put', 'F. Call'), index=None
    #     # 'Call Monitoring',
    # )

with st.sidebar.expander('OPTIONS'):
    infoType_O_P_S = st.checkbox("Put Sell")
    infoType_O_C_S = st.checkbox("Call Sell")

    # infoType = st.radio(
    #     "Choose an info type",
    #     ('Put Sell', 'Call Sell', 'Strangle', 'OTM Calendar', 'ITM Calendar', 'git test'), index=None # 'Call Monitoring',
    # )
#
# # with st.sidebar.expander('OPTIONS'):
# #     infoType_o = st.radio(
# #         "Choose an info type",
# #         ('Put Sell', 'Call Sell', 'Strangle', 'OTM Calendar', 'ITM Calendar', 'git test'), index=None# 'Call Monitoring',
# #     , key=3)
#
#
# # ************************************* PORTFOLIO ***************************************
# # =====================================   Portfolio
if infoType:
    st.text('MAIN')

# # ************************************* FUTURES ***************************************
# =====================================   Put
if infoType_F_P:
    f_put()


if infoType_F_C:
    f_call()

if infoType_F_ST:
    f_strangle()

# # ************************************* OPTIONS ***************************************
#
# =====================================   Put Sell
if infoType_O_P_S:
    put_sell()

# =====================================   Call Sell
if infoType_O_C_S:
    call_sell()
#
# # =====================================   Put Sell
# if infoType_o == 'Strangle':
#     strangle()
#
# # =====================================   Put Sell
# if infoType_o == 'OTM Calendar':
#     otm_calendar()
#
# # =====================================   Put Sell
# if infoType_o == 'ITM Calendar':
#     itm_calendar()

# =====================================   Put Sell
# if infoType == 'git test':
#     from git import Repo
#     PATH_OF_GIT_REPO = "https://github.com/Anton-Filimoncev/MONITORING_SYSTEM.git"
#     COMMIT_MESSAGE = 'comment from python script'
#
#
#
#     #
#     def git_push():
#         # try:
#         repo = Repo(search_parent_directories=True)
#         repo.git.add(update=True)
#         repo.index.commit(COMMIT_MESSAGE)
#         origin = repo.remote(name='origin')
#         origin.push()
#         # except:
#         #     print('Some error occured while pushing the code')
#
#
#     git_push()








