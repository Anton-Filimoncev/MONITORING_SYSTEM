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

f_put()


# #
# # if st.sidebar.button("PORTFOLIO", type="primary", key=0):
# #     infoType = "PORTFOLIO"
# #     infoType = None
# #     infoType = None
# #
# #
# # with st.sidebar.expander('FUTURES'):
# #     if st.button("F. Put", type="primary", key=1):
# #         infoType = 'F. Put'
# #
# #     # infoType_f = st.radio(
# #     #     "Choose an info type",
# #     #     ('F. Put', 'F. Call Sell', 'F. Strangle', 'F. OTM Calendar', 'F. ITM Calendar', 'F. git test'), index=None # 'Call Monitoring',
# #     #     , key=2
# #     # )
# #
# # with st.sidebar.expander('OPTIONS'):
# #     infoType_o = st.radio(
# #         "Choose an info type",
# #         ('Put Sell', 'Call Sell', 'Strangle', 'OTM Calendar', 'ITM Calendar', 'git test'), index=None# 'Call Monitoring',
# #     , key=3)
#
#
# # ************************************* PORTFOLIO ***************************************
# # =====================================   Portfolio
#
#
# # ************************************* FUTURES ***************************************
# # =====================================   Put
# if infoType == 'F. Put':
#     f_put()
#
# # ************************************* FUTURES ***************************************
#
# # =====================================   Portfolio
# if infoType_o == 'PORTFOLIO':
#     portfolio()
#
# # =====================================   Put Sell
# if infoType_o == 'Put Sell':
#     put_sell()
#
# # =====================================   Put Sell
# if infoType_o == 'Call Sell':
#     call_sell()
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








