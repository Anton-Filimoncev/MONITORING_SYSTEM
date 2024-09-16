import numpy as np
import streamlit as st
import pandas as pd
import glob
from .Support import *
import datetime
from .matrix import price_vol_matrix
from .STRANGLE_selection.MARKET_DATA import *
from .STRANGLE_selection.strang_select import *
from .databentos.get_databento import get_bento_data


def f_strangle():

    path = 'Side_bar/side_bar_data/futures/strangle/'
    path_bento = 'Side_bar/databentos/req/'
    filenames = glob.glob(path + "*.csv")

    with st.form(key='main_form'):

        col1, col2, col3 = st.columns([6, 1, 1])
        with col1:
            st.header('FUTURES STRANGLE')
        with col2:
            risk_rate = st.number_input('Risk Rate', step=0.5, format="%.1f", min_value=1., max_value=5000., value=4.8)

        with col3:
            # refresh_btn = st.button("Refresh ALL", type="primary")
            # refresh_btn = True
            pass

        # ============================================
        # ============================================     add new position
        # ============================================
        with st.expander('Add New F. Strangle Position'):
            col11, col12, col13, col14 = st.columns(4)
            with col11:
                percentage_array = st.number_input('Percentage', step=1, min_value=1, max_value=5000, value=30)
                end_date_stat = st.date_input('EXP date')
                iv_call = st.number_input('IV CALL', step=0.01, format="%.2f", min_value=0., max_value=5000.,
                                           value=0.0)
                iv_put = st.number_input('IV PUT', step=0.01, format="%.2f", min_value=0., max_value=5000., value=0.0)

            with col12:
                ticker = st.text_input('Ticker', '')
                try:
                    start_b_a_price_yahoo = yf.download(ticker)['Close'].iloc[-1]
                except:
                    start_b_a_price_yahoo = 0.
                start_date_o_p = st.date_input('Start date', datetime.datetime.now())
                start_b_a_price = st.number_input('Start BA Price', step=0.1, format="%.2f", min_value=0.,
                                                        max_value=50000., value=0.)
                margin_o_p = st.number_input('Margin', step=0.5, format="%.1f", min_value=0., max_value=55000.,
                                             value=6000.)

            with col13:
                prime_call_o_p = st.number_input('Start Prime CALL', step=0.01, format="%.2f", min_value=0.,
                                                  max_value=5000.)
                prime_put_o_p = st.number_input('Start Prime PUT', step=0.01, format="%.2f", min_value=0.,
                                                 max_value=5000.)
                strike_call_o_p = st.number_input('Strike CALL', step=0.5, format="%.1f", min_value=1.,
                                                   max_value=5000., value=100.)
                strike_put_o_p = st.number_input('Strike PUT', step=0.5, format="%.1f", min_value=1., max_value=5000.,
                                                  value=100.)


            with col14:
                num_pos_o_p = st.number_input('Positions', min_value=-100, max_value=365, value=1)
                multiplicator_o_p = st.number_input('Multiplicator', min_value=1, max_value=1000000, value=100)
                commission_o_p = st.number_input('Commission', step=0.1, format="%.1f", min_value=0., max_value=5000.,
                                                 value=3.4)

        submit_button = st.form_submit_button('Commit')
        col31, col32 = st.columns(2)

        # with col31:
        #     # ============================================
        #     if st.button("Commit", type="primary"):
        try:
            del st.session_state[f'position_data']
        except:
            pass
        input_new_df = pd.DataFrame({
            'Position_type': ['F. Strangle'],
            'Symbol': [ticker],
            'Symbol Bento': [ticker],
            'Start_date': [start_date_o_p],
            'Exp_date': [end_date_stat],
            'Rate': [risk_rate],
            'IV_Call': [iv_call],
            'IV_Put': [iv_put],
            'Percentage_Array': [percentage_array],
            'Strike_Call': [strike_call_o_p],
            'Strike_Put': [strike_put_o_p],
            'Number_pos': [num_pos_o_p],
            'Prime_Call': [prime_call_o_p],
            'Prime_Put': [prime_put_o_p],
            'Commission': [commission_o_p],
            'Margin': [margin_o_p],
            'Start_underlying': [start_b_a_price],
            'Multiplicator': [multiplicator_o_p],

        })

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
        emulate_df['DTE'] = (emulate_df['Exp_date'] - emulate_df['Start_date']).values[0].days

        position_emulate = emulate_position(emulate_df, path, path_bento, risk_rate)
        # if f'position_emulate' not in st.session_state:
        #     st.session_state[f'position_emulate'] = position_emulate

        dte = st.slider("Select DTE", 1, emulate_df['DTE'].values[0], value=emulate_df['DTE'].values[0])
        print('dteeeeeeeeeeeeee', dte)

        emulate_df['underlying_short'] = emulate_df['Start_underlying']
        emulate_df['underlying_long'] = emulate_df['Start_underlying']
        fig_map, weighted_profit_mtrx, weighted_loss_mtrx, weighted_rr_mtrx = price_vol_matrix(emulate_df, dte)

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

    with st.expander('F. Strangle Position '):
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
            prime_put_cur = st.number_input('Cur Prime Put', step=0.01, format="%.2f", min_value=0.,
                                              max_value=5000.)
            prime_call_cur = st.number_input('Cur Prime Call', step=0.01, format="%.2f", min_value=0.,
                                             max_value=5000.)
        with col23:
            iv_put_cur = st.number_input('Cur IV Put', step=0.01, format="%.2f", min_value=0., max_value=5000.,
                                           value=0.0)
            iv_call_cur = st.number_input('Cur IV Call', step=0.01, format="%.2f", min_value=0., max_value=5000.,
                                          value=0.0)
        with col24:
            b_a_price_cur = st.number_input('Cur BA Price', step=0.1, format="%.2f", min_value=0.,
                                                  max_value=50000., value=start_b_a_price_yahoo)

        update_btn = st.button("Update Position", type="primary")

        pos_type = 'F. Strangle'

    if update_btn:
        print('update_btn')
        input_update_df = pd.DataFrame({
            'Position_type': ['F. Strangle'],
            'IV_Put_Current': [iv_put_cur],
            'IV_Call_Current': [iv_call_cur],
            'Prime_put_Current': [prime_put_cur],
            'Prime_call_Current': [prime_call_cur],
            'underlying_Current': [b_a_price_cur],
        })

        update_postion_strangle(dia_type, pos_type, risk_rate, path_bento, input_update_df)

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
            infoType_plot_matrix = st.checkbox(
                tick + (' .' * 5) + 'PL: ' + str(pl) + (' .' * 5) + 'Margin: ' + str(marg) + (
                        ' .' * 5))
            if infoType_plot_matrix:
                with st.container():
                    print('position_df', position_df)

                    dte = st.slider("DTE", 1, full_postion_df['DTE'].values[0], value=full_postion_df['DTE'].values[0])
                    fig_map, weighted_profit_mtrx, weighted_loss_mtrx, weighted_rr_mtrx = price_vol_matrix(
                        csv_position_df, dte)

                    st.text(tick)
                    st.dataframe(position_df, hide_index=True, column_config=None)
                    st.dataframe(greeks_df, hide_index=True, column_config=None)
                    st.text(f'Weighted Profit: {weighted_profit_mtrx}')
                    st.text(f'Weighted Loss: {weighted_loss_mtrx}')
                    st.text(f'Weighted R/R: {weighted_rr_mtrx}')
                    # st.dataframe(fig_map.style.apply(highlight_mtx, axis=None))
                    st.plotly_chart(fig_map)

            my_bar.progress(int((100 / len(filenames)) * (num + 1)), text=progress_text)
            my_bar.empty()
            # submit_button = st.form_submit_button(f'Commit_{num}' )


    st.header('-'*10 + 'SELECTION' + '-'*10)
    with st.container():

        col1, col2, col3 = st.columns(3)

        with col1:
            ticker = st.text_input('Ticker', 'CL=F')
            ticker_b = st.text_input('Ticker Bento', 'LO')
        with col2:
            nearest_dte = st.number_input('Nearest DTE', step=1, min_value=1, max_value=50000, value=90)
        with col3:
            data_type = st.radio('Instrument Type', ["OPTIONS", "FUTURES"])

        if 'quotes' not in st.session_state:
            st.session_state['quotes'] = np.nan

        if 'needed_exp_date' not in st.session_state:
            st.session_state['needed_exp_date'] = np.nan

        path_bento = 'Side_bar/databentos/req/'


        if data_type == "OPTIONS":
            instr_type = 'OPT'
            if st.button("GET MARKET DATA", type="primary"):
                needed_exp_date, dte = hedginglab_get_exp_date(ticker, nearest_dte)
                quotes = hedginglab_get_quotes(ticker, nearest_dte)
                quotes['iv'] = quotes['iv'] * 100
                # st.text(dte)
                st.text('Exp Date: ' + str(needed_exp_date.date()))
                # st.dataframe(quotes)
                st.session_state['quotes'] = quotes
                st.session_state['needed_exp_date'] = needed_exp_date
                st.success('Market Data Downloaded!')

        if data_type == "FUTURES":
            instr_type = 'FUT'
            if st.button("GET BENTO DATA", type="primary"):
                needed_exp_date, quotes = get_bento_data(ticker, ticker_b, nearest_dte, 'Strangle', path_bento)
                quotes['iv'] = quotes['iv'] * 100
                # st.text(dte)
                st.text('Exp Date: ' + str(needed_exp_date.date()))
                # st.dataframe(quotes)
                st.session_state['quotes'] = quotes
                st.session_state['needed_exp_date'] = needed_exp_date
                st.success('Market Data Downloaded!')

        quotes = st.session_state['quotes']
        needed_exp_date = st.session_state['needed_exp_date']

        if '-' in ticker or '=' in ticker:
            try:
                days_to_expiration = (needed_exp_date - datetime.datetime.now()).days
            except:
                days_to_expiration = np.nan  # st.date_input('EXP date')
        else:
            try:
                days_to_expiration = (needed_exp_date - datetime.datetime.now()).days
            except:
                days_to_expiration = np.nan  # st.date_input('EXP date')

        col11, col12, col13 = st.columns(3)
        with col11:
            rate = st.number_input('Risk Rate', step=0.01, format="%.2f", min_value=0., max_value=50000., value=4.)
        with col12:
            percentage_array = st.number_input('Percentage', step=1, min_value=1, max_value=50000, value=50)

        with col13:
            try:
                closing_days_array = st.number_input('Closing Days Proba', step=1, min_value=0, max_value=50000,
                                                     value=int(days_to_expiration))
            except:
                closing_days_array = st.number_input('Closing Days Proba')
        #
        # if submit_button_select:
        #     st.success('Success commit!')

        if st.button("Calculate", type="primary"):
            print('quotes')
            print(quotes)

            strengle_data = get_strangle(ticker, rate, percentage_array, days_to_expiration, closing_days_array, quotes)

            st.text('Best Parameters:')
            st.dataframe(strengle_data, hide_index=True, column_config=None)