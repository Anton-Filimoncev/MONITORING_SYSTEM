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


    path = 'Side_bar/side_bar_data/futures/otm_calendar/'
    path_bento = 'Side_bar/databentos/req/'
    filenames = glob.glob(path + "*.csv")

    with st.form(key='main_form'):


        col1, col2, col3 = st.columns([6,1,1])
        with col1:
            st.header('FUTURES OTM CALENDAR')
        with col2:
            risk_rate = st.number_input('Risk Rate', step=0.5, format="%.1f", min_value=1., max_value=5000., value=4.8)

        with col3:
            # refresh_btn = st.button("Refresh ALL", type="primary")
            # refresh_btn = True
            pass

        # ============================================
        # ============================================     add new position
        # ============================================
        with st.expander('Add New F. OTM Calendar Position'):
            col11, col12, col13, col14 = st.columns(4)
            with col11:
                dia_type = st.selectbox(
                    "TYPE",
                    ("PUT", "CALL"),
                    index=None,
                    placeholder="Select TYPE...",
                )
                end_date_stat_short = st.date_input('EXP date SHORT')
                end_date_stat_long = st.date_input('EXP date LONG')
                iv_short = st.number_input('IV SHORT', step=0.01, format="%.2f", min_value=0., max_value=5000., value=0.0)
                iv_long = st.number_input('IV LONG', step=0.01, format="%.2f", min_value=0., max_value=5000., value=0.0)
            with col12:
                ticker = st.text_input('Ticker', '')

                start_date = st.date_input('Start date', datetime.datetime.now())
                underlying_short = st.number_input('Start BA Price SHORT', step=0.1, format="%.2f", min_value=0., max_value=50000., value=0.)
                underlying_long = st.number_input('Start BA Price LONG', step=0.1, format="%.2f", min_value=0.,
                                              max_value=50000., value=0.)
                margin = st.number_input('Margin', step=0.5, format="%.1f", min_value=0., max_value=55000., value=6000.)

            with col13:
                prime_short = st.number_input('Start Prime SHORT', step=0.01, format="%.2f", min_value=0., max_value=5000.)
                prime_long = st.number_input('Start Prime LONG', step=0.01, format="%.2f", min_value=0., max_value=5000.)
                strike_short = st.number_input('Strike SHORT', step=0.5, format="%.1f", min_value=0., max_value=500000., value=1.)
                strike_long = st.number_input('Strike LONG', step=0.5, format="%.1f", min_value=0., max_value=500000.,
                                                   value=0.)
                percentage_array = st.number_input('Percentage', step=1, min_value=1, max_value=5000, value=30)

            with col14:
                count_short = st.number_input('Positions SHORT', min_value=-100, max_value=365, value=-1)
                count_long = st.number_input('Positions LONG', min_value=-100, max_value=365, value=1)
                multiplier = st.number_input('Multiplicator', min_value=1, max_value=1000000, value=100)
                size = st.number_input('Size', min_value=1, max_value=100000, value=100)
                commission = st.number_input('Commission', step=0.1, format="%.1f", min_value=0., max_value=5000., value=3.4)

        submit_button = st.form_submit_button('Commit')
        col31, col32 = st.columns(2)

        # with col31:
        #     # ============================================
        #     if st.button("Commit", type="primary"):

        try:
            del st.session_state[f'position_data']
        except:
            pass

        days_to_expiration_short = (end_date_stat_short - datetime.datetime.now().date()).days
        days_to_expiration_long = (end_date_stat_long - datetime.datetime.now().date()).days
        print('days_to_expiration_short', days_to_expiration_short)
        print('days_to_expiration_long', days_to_expiration_long)
        # ---- OPTION ----
        df_short = pd.DataFrame({
            'position_type': ['F. OTM Calendar'],
            'symbol': [ticker],
            'symbol_bento': [ticker],
            'side': [dia_type],
            'strike': [strike_short],
            'days_to_exp': [days_to_expiration_short],
            'exp_date': [end_date_stat_short],
            'count': [count_short],
            'underlying': [underlying_short],
            'rate': [risk_rate],
            # 'closing_days_array': [percentage_array],
            'prime': [prime_short],
            'iv': [iv_short],
            'percentage_array': [percentage_array],
            'multiplier': [multiplier],
            'commission': [commission],
            'start_date': [start_date],
            'margin': [margin],
        })

        # ---- FUTURES ---
        df_long = pd.DataFrame({
            'position_type': ['F. OTM Calendar'],
            'symbol': [ticker],
            'symbol_bento': [ticker],
            'side': [dia_type],
            'strike': [strike_long],
            'days_to_exp': [days_to_expiration_long],
            'exp_date': [end_date_stat_long],
            'count': [count_long],
            'underlying': [underlying_long],
            'rate': [risk_rate],
            # 'closing_days_array': [percentage_array],
            'prime': [prime_long],
            'iv': [iv_long],
            'percentage_array': [percentage_array],
            'multiplier': [multiplier],
            'commission': [commission],
            'start_date': [start_date],
            'margin': [margin],
        })

        input_new_df = pd.concat([df_short, df_long]).reset_index(drop=True)

        if f'position_data' not in st.session_state:
            st.session_state[f'position_data'] = input_new_df

        print('st.session_state')
        print(st.session_state[f'position_data'])
        if submit_button:
            st.success('Success commit!')

    # st.button("Open", type="primary")

    if st.button("Open", type="primary"):
        create_new_postion(st.session_state[f'position_data'], path, path_bento, risk_rate)
        st.success('Position is OPEN waiting for $$$')


    infoType_plot_matrix = st.checkbox("POP and MATRIX")
    # try:
    if infoType_plot_matrix:
        emulate_df = st.session_state[f'position_data'].copy()

        position_emulate = emulate_position(emulate_df, path, path_bento, risk_rate)
        # if f'position_emulate' not in st.session_state:
        #     st.session_state[f'position_emulate'] = position_emulate

        dte = st.slider("Select DTE", 1, days_to_expiration_short, value=days_to_expiration_short)
        print('dteeeeeeeeeeeeee', dte)
        fig_map, weighted_profit_mtrx, weighted_loss_mtrx, weighted_rr_mtrx = price_vol_matrix_covered(emulate_df, dte)

        st.dataframe(position_emulate, hide_index=True, column_config=None)
        st.text(f'Weighted Profit: {weighted_profit_mtrx}')
        st.text(f'Weighted Loss: {weighted_loss_mtrx}')
        st.text(f'Weighted R/R: {weighted_rr_mtrx}')
        # st.dataframe(fig_map.style.apply(highlight_mtx, axis=None))
        st.plotly_chart(fig_map)
    # except Exception as err:
    #     print(f'Exception {err}')
    #     st.header(f'Pleas Commit Data')
    #     pass



        # show all open position
        print(filenames)

    with st.expander('F. Dia Position '):
        col21, col22, col23, col24 = st.columns(4)
        with col21:
            dia_type = st.selectbox(
                "Refresh Position",
                # ([get_tick_from_csv_name(csv_position_df) for csv_position_df in filenames]),
                (filenames),
                index=None,
                placeholder="Select TYPE...",
            )

        with col22:
            prime_short_cur = st.number_input('Cur Prime SHORT', step=0.01, format="%.2f", min_value=0.,
                                              max_value=5000.)
            prime_long_cur = st.number_input('Cur Prime LONG', step=0.01, format="%.2f", min_value=0.,
                                             max_value=5000.)
        with col23:
            iv_short_cur = st.number_input('Cur IV SHORT', step=0.01, format="%.2f", min_value=0., max_value=5000., value=0.0)
            iv_long_cur = st.number_input('Cur IV LONG', step=0.01, format="%.2f", min_value=0., max_value=5000., value=0.0)
        with col24:
            b_a_price_short_cur = st.number_input('Cur BA Price SHORT', step=0.1, format="%.2f", min_value=0., max_value=50000., value=0.)
            b_a_price_long_cur = st.number_input('Cur BA Price LONG', step=0.1, format="%.2f", min_value=0.,
                                          max_value=50000., value=0.)


        update_btn = st.button("Update Position", type="primary")

        pos_type = 'F. Diagonal'


    if update_btn:
        print('update_btn')
        df_short_update = pd.DataFrame({
            'position_type': ['F. Diagonal'],
            'iv_current': [iv_short_cur],
            'prime_current': [prime_short_cur],
            'underlying_current': [b_a_price_short_cur],
        })

        # ---- FUTURES ---
        df_long_update = pd.DataFrame({
            'position_type': ['F. Diagonal'],
            'iv_current': [iv_long_cur],
            'prime_current': [prime_long_cur],
            'underlying_current': [b_a_price_long_cur],
        })

        input_update_df = pd.concat([df_short_update, df_long_update]).reset_index(drop=True)

        update_postion_cover(dia_type, pos_type, risk_rate, path_bento, input_update_df)

        st.success('Position Updated!')

    st.header("OPENED POSITIONS")
    # with st.form(key='show_position'):
    # show_button = st.form_submit_button('SHOW POSITION')
    # if show_button:
    with st.container():
        # if show_btn:
        progress_text = "Loading..."
        my_bar = st.progress(0, text=progress_text)
        for num, csv_position_df in enumerate(filenames):
            # with st.form(key=csv_position_df):
            print('num', num)
            print('csv_position_df', csv_position_df)
            tick = get_tick_from_csv_name(csv_position_df)
            print('tick', tick)
            pos_type = 'F. Diagonal'

                # st.success('All data is updated!')
            # else:
            #     print('elseeeeeeeeeee')
            full_postion_df = pd.read_csv(csv_position_df)

            position_df, greeks_df, pl, marg = return_postion(csv_position_df, pos_type, risk_rate)  # greeks_df,
            # with st.expander(tick + (' .' * 5) + 'PL: ' + str(pl) + (' .' * 5) + 'Margin: ' + str(marg) + (
            #         ' .' * 5) + 'POP lognormal: ' + str(pop_log)):

            infoType_plot_matrix = st.checkbox(tick + (' .' * 5) + 'PL: ' + str(pl) + (' .' * 5) + 'Margin: ' + str(marg) + (
                    ' .' * 5) )
            if infoType_plot_matrix:
                with st.container():
                    print('position_df', position_df)
                    st.text(tick)
                    st.dataframe(position_df, hide_index=True, column_config=None)
                    st.dataframe(greeks_df, hide_index=True, column_config=None)

                    dte = st.slider("Select DTE", 1, full_postion_df['days_to_exp'].values[0], value=full_postion_df['days_to_exp'].values[0])
                    fig_map, weighted_profit_mtrx, weighted_loss_mtrx, weighted_rr_mtrx = price_vol_matrix_covered(csv_position_df, dte)

                    st.text(f'Weighted Profit: {weighted_profit_mtrx}')
                    st.text(f'Weighted Loss: {weighted_loss_mtrx}')
                    st.text(f'Weighted R/R: {weighted_rr_mtrx}')
                    # st.dataframe(fig_map.style.apply(highlight_mtx, axis=None))
                    st.plotly_chart(fig_map)

            my_bar.progress(int((100 / len(filenames)) * (num + 1)), text=progress_text)
            my_bar.empty()
                # submit_button = st.form_submit_button(f'Commit_{num}' )


    # =================
    # ================= BARCHART
    # =================
    infoType_barchart = st.checkbox("~~ BARCHART ~~")

    if infoType_barchart:
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