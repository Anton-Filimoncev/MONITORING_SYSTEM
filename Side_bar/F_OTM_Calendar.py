import numpy as np
import streamlit as st
import pandas as pd
import glob
from .Support import *
import datetime
from .matrix  import price_vol_matrix, price_vol_matrix_covered
from .SELECTION.Support_Selection import *
from .databentos.get_databento import get_bento_data
from .SELECTION.MARKET_DATA import *
from .barchart.otm_calendar import barchart_selection
import os

def f_otm_cal():
    col111, col121, col131, col141 = st.columns(4)
    with col111:
        tick = st.text_input('Ticker', 'ZC=F')
        side = st.selectbox(
            "TYPE",
            ("Put", "Call"),
            index=None,
            placeholder="Select TYPE...",
        )
    with col121:
        rate = st.number_input('Risk Rate', step=0.5, format="%.1f", min_value=1., max_value=5000., value=4.8)

    with col131:
        percentage_array = st.number_input('Percentage', step=1, min_value=1, max_value=5000, value=30)

    with col141:
        multiplier = st.number_input('Multiplicator', min_value=1, max_value=1000000, value=100)

    folder_path = 'Side_bar/BARCHART_DATA/diagonal'  # Замените на путь к вашей папке
    file_names = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    dte_list = []

    for file in file_names:
        dte_list.append(file.split('_')[0][-2:] + '_' + file.split('_')[1] + '_' + file.split('_')[2][:2])
    date_format = "%m_%d_%y"  # Формат для преобразования
    date1 = datetime.datetime.strptime(dte_list[0], date_format)
    date2 = datetime.datetime.strptime(dte_list[1], date_format)

    # Сравниваем даты
    if date1 < date2:
        short_df = pd.read_csv(f'{folder_path}/{file_names[0]}')
        long_df = pd.read_csv(f'{folder_path}/{file_names[1]}')
        print('short_df', file_names[0])
        # Вычисляем разницу в днях
        short_dte = (date1 - datetime.datetime.now()).days
        long_dte = (date2 - datetime.datetime.now()).days
    elif date1 > date2:
        short_df = pd.read_csv(f'{folder_path}/{file_names[1]}')
        print('short_df', file_names[1])
        long_df = pd.read_csv(f'{folder_path}/{file_names[0]}')
        # Вычисляем разницу в днях
        short_dte = (date2 - datetime.datetime.now()).days
        long_dte = (date1 - datetime.datetime.now()).days

    barchart_button = st.button('Run')
    if barchart_button:
        return_df, best_df = barchart_selection(short_df, long_df, side, short_dte, long_dte, tick, rate,
                                                percentage_array, multiplier)
        st.dataframe(best_df, hide_index=True, column_config=None)
        st.dataframe(return_df, hide_index=True, column_config=None)