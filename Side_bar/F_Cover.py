import numpy as np
import streamlit as st
import pandas as pd
import glob
from .Support import *
import datetime
from .matrix  import price_vol_matrix, price_vol_matrix_covered

def f_cover():

    path = 'Side_bar/side_bar_data/futures/covered/'
    path_bento = 'Side_bar/databentos/req/'
    filenames = glob.glob(path + "*.csv")

    with st.form(key='main_form'):


        col1, col2, col3 = st.columns([6,1,1])
        with col1:
            st.header('COVERED FUTURES')
        with col2:
            risk_rate = st.number_input('Risk Rate', step=0.5, format="%.1f", min_value=1., max_value=5000., value=4.8)

        with col3:
            # refresh_btn = st.button("Refresh ALL", type="primary")
            # refresh_btn = True
            pass

        # ============================================
        # ============================================     add new position
        # ============================================
        with st.expander('Add New F. Covered Position'):
            col11, col12, col13, col14 = st.columns(4)
            with col11:
                option_type = st.selectbox(
                    "TYPE",
                    ("PUT", "CALL"),
                    index=None,
                    placeholder="Select TYPE...",
                )

                exp_date = st.date_input('EXP')
                iv = st.number_input('IV', step=0.01, format="%.2f", min_value=0., max_value=5000., value=0.0)
                margin = st.number_input('Margin', step=0.5, format="%.1f", min_value=0., max_value=55000., value=6000.)
            with col12:
                ticker = st.text_input('Ticker', '')
                start_date = st.date_input('Start date', datetime.datetime.now())
                underlying_option = st.number_input('Option Underlying', step=0.1, format="%.2f", min_value=0., max_value=50000., value=0.)
                underlying_futures = st.number_input('Futures Price', step=0.1, format="%.2f", min_value=0.,
                                              max_value=50000., value=0.)

            with col13:
                prime = st.number_input('Start Prime', step=0.01, format="%.2f", min_value=0., max_value=5000.)
                strike = st.number_input('Strike', step=0.5, format="%.1f", min_value=0., max_value=500000., value=0.)
                percentage_array = st.number_input('Percentage', step=1, min_value=1, max_value=5000, value=30)
                commission = st.number_input('Commission', step=0.1, format="%.1f", min_value=0., max_value=5000., value=3.4)
            with col14:
                count = st.number_input('Count Option', min_value=-100, max_value=365, value=-1)
                count_futures = st.number_input('Count Futures', min_value=-100, max_value=365, value=1)
                multiplier = st.number_input('Multiplicator', min_value=1, max_value=1000000, value=100)



        submit_button = st.form_submit_button('Commit')
        # col31, col32 = st.columns(2)

        # with col31:
        #     # ============================================
        #     if st.button("Commit", type="primary"):
        try:
            del st.session_state[f'position_data']
        except:
            pass

        days_to_expiration = (exp_date - datetime.datetime.now().date()).days
        print('days_to_expiration', days_to_expiration)
        # ---- OPTION ---
        df_opt = pd.DataFrame({
            'position_type': ['F. Covered'],
            'symbol': [ticker],
            'symbol_bento': [ticker],
            'side': [option_type],
            'strike': [strike],
            'days_to_exp': [days_to_expiration],
            'exp_date': [exp_date],
            'count': [count],
            'underlying': [underlying_option],
            'rate': [risk_rate],
            # 'closing_days_array': [percentage_array],
            'prime': [prime],
            'iv': [iv],
            'percentage_array': [percentage_array],
            'multiplier': [multiplier],
            'commission': [commission],
            'start_date': [start_date],
            'margin': [margin],
        })

        # ---- FUTURES ---
        df_fut = pd.DataFrame({
            'position_type': ['F. Covered'],
            'symbol': [ticker],
            'symbol_bento': [ticker],
            'side': ['STOCK'],
            'strike': [strike],
            'days_to_exp': [days_to_expiration],
            'exp_date': [exp_date],
            'count': [count_futures],
            'underlying': [underlying_futures],
            'rate': [risk_rate],
            # 'closing_days_array': [percentage_array],
            'prime': [prime],
            'iv': [iv],
            'percentage_array': [percentage_array],
            'multiplier': [multiplier],
            'commission': [commission],
            'start_date': [start_date],
            'margin': [margin],
        })

        input_new_df = pd.concat([df_opt, df_fut]).reset_index(drop=True)

        if f'position_data' not in st.session_state:
            st.session_state[f'position_data'] = input_new_df

        print('st.session_state')
        print(st.session_state[f'position_data'])
        if submit_button:
            st.success('Success commit!')
            print(st.session_state[f'position_data'])

    # st.button("Open", type="primary")

    if st.button("Open", type="primary"):
        create_new_postion(st.session_state[f'position_data'], path, path_bento, risk_rate)
        st.success('Position is OPEN waiting for $$$')


    infoType_plot_matrix = st.checkbox("POP and MATRIX")
    # try:
    if infoType_plot_matrix:
        emulate_df = st.session_state[f'position_data'].copy()


        position_emulate = emulate_position(emulate_df, path, path_bento, risk_rate)

        dte = st.slider("Select DTE", 1, emulate_df['days_to_exp'].values[0], value=emulate_df['days_to_exp'].values[0])

        fig_map, weighted_profit_mtrx, weighted_loss_mtrx, weighted_rr_mtrx = price_vol_matrix_covered(emulate_df, dte)

        st.dataframe(position_emulate[0:1], hide_index=True, column_config=None)
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

    with st.expander('F. Covered Position '):
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
            prime_cur = st.number_input('Cur Prime', step=0.01, format="%.2f", min_value=0.,
                                              max_value=5000.)
        with col23:
            iv_cur = st.number_input('Cur IV', step=0.01, format="%.2f", min_value=0., max_value=5000., value=0.0)
        with col24:
            underlying_option_cur = st.number_input('Option Underlying', step=0.1, format="%.2f", min_value=0., max_value=50000., value=0.)
            underlying_futures_cur = st.number_input('Futures Price', step=0.1, format="%.2f", min_value=0.,
                                          max_value=50000., value=0.)


        update_btn = st.button("Update Position", type="primary")

        pos_type = 'F. Covered'

    if update_btn:
        print('update_btn')
        df_opt_update = pd.DataFrame({
            'position_type': ['F. Covered'],
            'iv_current': [iv_cur],
            'prime_current': [prime_cur],
            'underlying_current': [underlying_option_cur],
        })

        # ---- FUTURES ---
        df_fut_update = pd.DataFrame({
            'position_type': ['F. Covered'],
            'iv_current': [iv_cur],
            'prime_current': [prime_cur],
            'underlying_current': [underlying_futures_cur],
        })


        input_update_df = pd.concat([df_opt_update, df_fut_update]).reset_index(drop=True)

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
            pos_type = 'F. Covered'

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

                    dte = st.slider("DTE", 1, full_postion_df['days_to_exp'].values[0], value=full_postion_df['days_to_exp'].values[0])
                    fig_map, weighted_profit_mtrx, weighted_loss_mtrx, weighted_rr_mtrx = price_vol_matrix_covered(csv_position_df, dte)

                    st.text(f'Weighted Profit: {weighted_profit_mtrx}')
                    st.text(f'Weighted Loss: {weighted_loss_mtrx}')
                    st.text(f'Weighted R/R: {weighted_rr_mtrx}')
                    # st.dataframe(fig_map.style.apply(highlight_mtx, axis=None))
                    st.plotly_chart(fig_map)

            my_bar.progress(int((100 / len(filenames)) * (num + 1)), text=progress_text)
            my_bar.empty()
                # submit_button = st.form_submit_button(f'Commit_{num}' )


