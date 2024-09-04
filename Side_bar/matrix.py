import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from .popoption.poptions import *
import plotly.express as px
import plotly.graph_objects as go


# from Road_map import *
def gat_estimated_vol(df_data, dte):
    realized_estimate = {}
    # calculate realized volatility
    intraday_vol = np.log(df_data['High'].shift(-1) / df_data['Open'].shift(-1)) * np.log(
        df_data['High'].shift(-1) / df_data['Close'].shift(-1)) + \
                   np.log(df_data['Open'].shift(-1) / df_data['Open'].shift(-1)) * \
                   np.log(df_data['Low'].shift(-1) / df_data['Close'].shift(-1))
    intraday_vol = (np.sqrt(intraday_vol) * 100 * 16) ** 2
    overnight_vol = (np.abs(np.log(df_data['Open'] / df_data['Close'])) * 100 * 16) ** 2
    realized_vol = np.sqrt(intraday_vol + overnight_vol)
    # calculate estimates for short medium and long term realized vol
    short_term_vol = np.sqrt(np.mean(realized_vol[-5:] ** 2))
    med_term_vol = np.sqrt(np.mean(realized_vol[-20:] ** 2))
    long_term_vol = np.sqrt(np.mean(realized_vol ** 2))
    # aggregate it into a single estimate
    estimate_vol = 0.6 * short_term_vol + 0.3 * med_term_vol + 0.1 * long_term_vol
    return estimate_vol

def volatility_calc(stock_yahoo, period):
    # ======= HIST volatility calculated ===========
    # TRADING_DAYS = 252
    returns = np.log(stock_yahoo / stock_yahoo.shift(1))
    returns.fillna(0, inplace=True)
    volatility = returns.rolling(window=period).std() * np.sqrt(365)

    return volatility

def three_colorscale(z, colors):
    if len(colors) !=3:
        raise ValueError("")
    neg=z[np.where(z<0)]
    pos=z[np.where(z>0)]
    a, c = neg.min(), neg.max()
    try:
        d, b = pos.min(), pos.max()
    except:
        d, b = 0, 0
    bvals= [a, c/2, d/2, b]
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]
    dcolorscale = []
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale


def foo(start, stop, count):
    step = (stop - start) / float(count)
    return [start + i * step for i in range(count)]

def price_vol_matrix(df, dte):
    try:
        input_df = pd.read_csv(df).iloc[0]
    except:
        input_df = df.iloc[0]
    print(input_df)
    print(input_df['underlying_short'])
    full_matrix = pd.DataFrame()

    # если позиция уже открыта, переопределяем переменные, если нет, то по открытию считаем

    try:
        input_df['IV_short'] = input_df['IV_SHORT_Current']
        input_df['IV_long'] = input_df['IV_LONG']
        input_df['underlying_short'] = input_df['underlying_short_Current']
        input_df['underlying_long'] = input_df['underlying_long_Current']

    except:
        print('AAAAAAA except')
        pass

    mean_sigma = (input_df['IV_short'] + input_df['IV_long'])/2
    mean_underlying = (input_df['underlying_short'] + input_df['underlying_long'])/2

    # min_dte = np.min([input_df['DTE_short'], input_df['DTE_long']])
    # max_dte = np.max([input_df['DTE_short'], input_df['DTE_long']])
    # spread = np.max([input_df['underlying_short'], input_df['underlying_long']]) - np.min([input_df['underlying_short'], input_df['underlying_long']])

    df = pd.DataFrame()

    df['Days to EXP'] = [input_df['DTE_short'], input_df['DTE_long']]
    df['rate'] = input_df['Rate']
    df['Type'] = input_df['Position_side']
    df['Symb'] = input_df['Symbol']
    df['IV'] = [input_df['IV_SHORT'], input_df['IV_LONG']]
    # df['IV'] = df['IV'] / 100
    df['strike'] = [input_df['Strike_short'], input_df['Strike_long']]
    df['underlying'] = [input_df['underlying_short'], input_df['underlying_long']]
    df['Count'] = [input_df['Number_pos_short'], input_df['Number_pos_long']]
    df['prime'] = [input_df['Prime_short'], input_df['Prime_long']]
    df['multiplier'] = input_df['Multiplicator']

    min_dte = df['Days to EXP'].min()
    max_dte = df['Days to EXP'].max()
    spread = df['underlying'].max() - df['underlying'].min()

    print('df')
    print(df)

    #
    #
    # Позиции с одной стоимость БА
    #
    #
    print(input_df['underlying_short'])
    print(input_df['underlying_long'])
    if input_df['underlying_short'] == input_df['underlying_long']:
        for num, row in df.iterrows():
            position_dte = row['Days to EXP']
            row['rate'] = row['rate'] / 100
            side = row['Type']
            yahoo_stock = yf.download(row['Symb'])
            hist_vol = volatility_calc(yahoo_stock['Close'], 245)
            # print('hist_vol', hist_vol)
            std_hv_step = (hist_vol[-245:].std() * 2) / 10
            max_under_std = yahoo_stock['Close'][-245:].std() * 2
            underlying_step = max_under_std / 10
            sigma_list = []
            sigma_name = []
            underlying_list = []
            underlying_name = []
            start_up = mean_sigma / 100
            start_down = mean_sigma / 100
            underlying_up = mean_underlying
            underlying_down = mean_underlying

            for i in reversed(range(10)):
                start_up = start_up + std_hv_step
                underlying_up = underlying_up + underlying_step
                sigma_list.append(start_up)
                underlying_list.append(underlying_up)
                sigma_name.append(round((underlying_step/max_under_std * i+1 - 1)*2 + 0.2, 2))
                underlying_name.append(round((underlying_step/max_under_std * i+1 - 1)*2 + 0.2, 2))

            sigma_list.append(mean_sigma / 100)
            sigma_list = sorted(sigma_list, reverse=True)
            sigma_name.append(0)
            underlying_list.append(mean_underlying)
            underlying_name.append(0)

            print('underlying_list')
            print(underlying_list)

            for i in range(10):
                start_down = start_down - std_hv_step
                underlying_down = underlying_down - underlying_step
                sigma_list.append(start_down)
                underlying_list.append(underlying_down)
                sigma_name.append(-round((underlying_step/max_under_std * i+1 - 1)*2 + 0.2, 2))
                underlying_name.append(-round((underlying_step/max_under_std * i+1 - 1)*2 + 0.2, 2))

            loop_df = pd.DataFrame({
                'IV': sigma_list,
                'underlying': underlying_list,
                'sigma_name': sigma_name,
                'underlying_name': underlying_name,
            })

            matrix = pd.DataFrame()

            for stock_price in sorted(loop_df['underlying'].values.tolist()):
                black_df = pd.DataFrame()
                black_df['sigma'] = loop_df['IV']
                black_df['strike'] = row['strike']
                black_df['dte'] = (dte + position_dte - min_dte) * 1/365
                black_df['rate'] = row['rate'] / 100
                black_df['stock_price'] = stock_price

                if side == 'PUT':
                    option_prices = black76Put(black_df)
                elif side == 'CALL':
                    option_prices = black76Call(black_df)
                elif side == 'STOCK':
                    option_prices = black_df['stock_price']

                if side != 'STOCK':
                    if row['Count'] < 0:
                        return_matrix = row['prime'] - option_prices
                    if row['Count'] > 0:
                        return_matrix = option_prices - row['prime']
                else:
                    if row['Count'] < 0:
                        return_matrix = row['underlying'] - option_prices
                    if row['Count'] > 0:
                        return_matrix = option_prices - row['underlying']
                return_matrix = return_matrix.to_frame(f'{round(stock_price, 4)}')

                matrix = pd.concat([matrix, return_matrix], axis=1)

            matrix['IV'] = loop_df['IV']
            # matrix = matrix.sort_values('IV')
            matrix = matrix.set_index('IV') * row['multiplier'] * np.abs(row['Count'])

            full_matrix = pd.concat([matrix, full_matrix], axis=0)

    #
    #
    # Позиции со спредом
    #
    #

    else:
        prices_list = []
        for num, row in df.iterrows():
            current_price = row['underlying']
            position_dte = row['Days to EXP']
            row['rate'] = row['rate'] / 100
            side = row['Type']
            yahoo_stock = yf.download(row['Symb'])
            hist_vol = volatility_calc(yahoo_stock['Close'], 245)
            # print('hist_vol', hist_vol)

            # underlying_list = np.arange(current_price-(spread/2), current_price+spread, spread/0)
            underlying_list = foo(current_price-(spread*2), current_price+(spread*2), 10)
            # underlying_name = np.arange(-(spread/2), spread, spread/10)
            underlying_name = foo(-(spread*2), (spread*2), 10)
            prices_list.append(underlying_list)
            print('underlying_list', underlying_list)
            print('underlying_name', underlying_name)

            std_hv_step = (hist_vol[-245:].std() * 2) / 10
            max_under_std = yahoo_stock['Close'][-245:].std() * 2
            underlying_step = max_under_std / 10
            sigma_list = []
            sigma_name = []
            # underlying_list = []
            # underlying_name = []
            start_up = mean_sigma / 100
            start_down = mean_sigma / 100
            underlying_up = mean_underlying
            underlying_down = mean_underlying

            for i in reversed(range(10)):
                start_up = start_up + std_hv_step
                underlying_up = underlying_up + underlying_step
                sigma_list.append(start_up)
                # underlying_list.append(underlying_up)
                sigma_name.append(round((underlying_step / max_under_std * i + 1 - 1) * 2 + 0.2, 2))
                # underlying_name.append(round((underlying_step / max_under_std * i + 1 - 1) * 2 + 0.2, 2))

            sigma_list.append(mean_sigma / 100)
            sigma_list = sorted(sigma_list, reverse=True)
            sigma_name.append(0)
            # underlying_list.append(mean_underlying)
            # underlying_name.append(0)

            for i in range(10):
                start_down = start_down - std_hv_step
                underlying_down = underlying_down - underlying_step
                sigma_list.append(start_down)
                # underlying_list.append(underlying_down)
                sigma_name.append(-round((underlying_step / max_under_std * i + 1 - 1) * 2 + 0.2, 2))
                # underlying_name.append(-round((underlying_step / max_under_std * i + 1 - 1) * 2 + 0.2, 2))

            loop_df = pd.DataFrame({
                'IV': sigma_list,
                # 'underlying': underlying_list,
                'sigma_name': sigma_name,
                # 'underlying_name': underlying_name,
            })

            matrix = pd.DataFrame()

            for stock_price in sorted(underlying_list):
                black_df = pd.DataFrame()
                black_df['sigma'] = loop_df['IV']
                black_df['strike'] = row['strike']
                black_df['dte'] = (dte + position_dte - min_dte) * 1 / 365
                black_df['rate'] = row['rate'] / 100
                black_df['stock_price'] = stock_price

                if side == 'PUT':
                    option_prices = black76Put(black_df)
                elif side == 'CALL':
                    option_prices = black76Call(black_df)
                elif side == 'STOCK':
                    option_prices = black_df['stock_price']

                if side != 'STOCK':
                    if row['Count'] < 0:
                        return_matrix = row['prime'] - option_prices
                    if row['Count'] > 0:
                        return_matrix = option_prices - row['prime']
                else:
                    if row['Count'] < 0:
                        return_matrix = row['underlying'] - option_prices
                    if row['Count'] > 0:
                        return_matrix = option_prices - row['underlying']

                return_matrix = return_matrix.to_frame()
                matrix = pd.concat([matrix, return_matrix], axis=1)

            print(matrix)
            matrix.columns = underlying_name
            matrix['IV'] = loop_df['IV']

            # matrix = matrix.sort_values('IV')
            matrix = matrix.set_index('IV') * row['multiplier'] * np.abs(row['Count'])

            full_matrix = pd.concat([matrix, full_matrix], axis=0)



    full_matrix = full_matrix.groupby('IV').sum()

    full_matrix = full_matrix.set_index(loop_df.sort_values('sigma_name')['sigma_name'])
    full_matrix.columns = sorted(underlying_name)

    if full_matrix.min().min() > 0:
        pl_colorscale = 'greens'
    elif full_matrix.max().max() > 0:
        pl_colorscale = three_colorscale(full_matrix.to_numpy(), ["#DC3714", "#C0C0C0","#19BD1B"])
    else:
        pl_colorscale = 'Reds'
    # pl_colorscale = 'Reds'



    temp_df = pd.DataFrame({
        'Fist': prices_list[0],
        'Sec': prices_list[1]
    })

    temp_df = temp_df.astype('str')
    temp_df['X'] = temp_df['Fist'] + '_' + temp_df['Sec']

    fig_map = go.Figure(data=go.Heatmap(
        z=full_matrix.round(2).values.tolist(),
        x=temp_df['X'] .values.tolist(),
        y=full_matrix.index.values.tolist(),
        hoverongaps=False,
        colorscale = pl_colorscale,
        colorbar_thickness=24,
        xgap=0.5,
        ygap=0.5,
    )
    )

    fig_map.update_traces(

        hovertemplate='<br>'.join([
            'Price: %{x}',
            'IV: %{y}',
            'Values: %{z}',
        ])
    )

    fig_map.update_layout(title="Heatmap",
                          yaxis={"title": 'IV STD'},
                          xaxis={"title": 'Prices STD'},)

    # fig_map.show()

    weighted_profit = round(full_matrix[full_matrix > 0].mean(), 2).mean()
    weighted_loss = round(full_matrix[full_matrix < 0].mean(), 2).mean()
    weighted_rr = round(weighted_profit / np.abs(weighted_loss), 2).mean()
    if len(full_matrix[full_matrix < 0]) <= 0:
        weighted_loss = 0
        weighted_rr = round(weighted_profit, 2)

    print('weighted_rr', weighted_rr)
    print(full_matrix)
    return fig_map, weighted_profit, weighted_loss, weighted_rr


def price_vol_matrix_covered(df, dte):
    try:
        input_df = pd.read_csv(df).iloc[0]
    except:
        input_df = df.iloc[0]
    print(input_df)
    full_matrix = pd.DataFrame()

    # если позиция уже открыта, переопределяем переменные, если нет, то по открытию считаем

    try:
        input_df['IV'] = input_df['IV_Current']
        input_df['Underlying'] = input_df['Underlying_Current']
        input_df['Underlying_stock'] = input_df['Underlying_stock_Current']

    except:
        print('AAAAAAA except')
        pass

    mean_sigma = input_df['IV']
    mean_underlying = (input_df['Underlying_stock'] + input_df['Underlying'])/2

    # min_dte = np.min([input_df['DTE_short'], input_df['DTE_long']])
    # max_dte = np.max([input_df['DTE_short'], input_df['DTE_long']])
    # spread = np.max([input_df['underlying_short'], input_df['underlying_long']]) - np.min([input_df['underlying_short'], input_df['underlying_long']])

    df = pd.DataFrame()
    df['Days to EXP'] = [input_df['DTE'], input_df['DTE']]
    df['rate'] = input_df['Rate']
    df['Type'] = input_df['Position_side']
    df['Symb'] = input_df['Symbol']
    df['IV'] = [input_df['IV'], input_df['IV']]
    # df['IV'] = df['IV'] / 100
    df['strike'] = [input_df['Strike'], input_df['Strike']]
    df['underlying'] = [input_df['Underlying'], input_df['Underlying_stock']]
    df['Count'] = [input_df['Number_pos'], input_df['Number_pos']]
    df['prime'] = [input_df['Prime'], input_df['Prime']]
    df['multiplier'] = input_df['Multiplicator']

    min_dte = df['Days to EXP'].min()
    max_dte = df['Days to EXP'].max()
    spread = df['underlying'].max() - df['underlying'].min()

    print('df')
    print(df)

    #
    #
    # Позиции с одной стоимость БА
    #
    #

    if input_df['Underlying_stock'] == input_df['Underlying']:
        for num, row in df.iterrows():
            position_dte = row['Days to EXP']
            row['rate'] = row['rate'] / 100
            side = row['Type']
            yahoo_stock = yf.download(row['Symb'])
            hist_vol = volatility_calc(yahoo_stock['Close'], 245)
            # print('hist_vol', hist_vol)
            std_hv_step = (hist_vol[-245:].std() * 2) / 10
            max_under_std = yahoo_stock['Close'][-245:].std() * 2
            underlying_step = max_under_std / 10
            sigma_list = []
            sigma_name = []
            underlying_list = []
            underlying_name = []
            start_up = mean_sigma / 100
            start_down = mean_sigma / 100
            underlying_up = mean_underlying
            underlying_down = mean_underlying

            for i in reversed(range(10)):
                start_up = start_up + std_hv_step
                underlying_up = underlying_up + underlying_step
                sigma_list.append(start_up)
                underlying_list.append(underlying_up)
                sigma_name.append(round((underlying_step/max_under_std * i+1 - 1)*2 + 0.2, 2))
                underlying_name.append(round((underlying_step/max_under_std * i+1 - 1)*2 + 0.2, 2))

            sigma_list.append(mean_sigma / 100)
            sigma_list = sorted(sigma_list, reverse=True)
            sigma_name.append(0)
            underlying_list.append(mean_underlying)
            underlying_name.append(0)

            print('underlying_list')
            print(underlying_list)

            for i in range(10):
                start_down = start_down - std_hv_step
                underlying_down = underlying_down - underlying_step
                sigma_list.append(start_down)
                underlying_list.append(underlying_down)
                sigma_name.append(-round((underlying_step/max_under_std * i+1 - 1)*2 + 0.2, 2))
                underlying_name.append(-round((underlying_step/max_under_std * i+1 - 1)*2 + 0.2, 2))

            loop_df = pd.DataFrame({
                'IV': sigma_list,
                'underlying': underlying_list,
                'sigma_name': sigma_name,
                'underlying_name': underlying_name,
            })

            matrix = pd.DataFrame()

            for stock_price in sorted(loop_df['underlying'].values.tolist()):
                black_df = pd.DataFrame()
                black_df['sigma'] = loop_df['IV']
                black_df['strike'] = row['strike']
                black_df['dte'] = (dte + position_dte - min_dte) * 1/365
                black_df['rate'] = row['rate'] / 100
                black_df['stock_price'] = stock_price

                if side == 'PUT':
                    option_prices = black76Put(black_df)
                elif side == 'CALL':
                    option_prices = black76Call(black_df)
                elif side == 'STOCK':
                    option_prices = black_df['stock_price']

                if side != 'STOCK':
                    if row['Count'] < 0:
                        return_matrix = row['prime'] - option_prices
                    if row['Count'] > 0:
                        return_matrix = option_prices - row['prime']
                else:
                    if row['Count'] < 0:
                        return_matrix = row['underlying'] - option_prices
                    if row['Count'] > 0:
                        return_matrix = option_prices - row['underlying']
                return_matrix = return_matrix.to_frame(f'{round(stock_price, 4)}')

                matrix = pd.concat([matrix, return_matrix], axis=1)

            matrix['IV'] = loop_df['IV']
            # matrix = matrix.sort_values('IV')
            matrix = matrix.set_index('IV') * row['multiplier'] * np.abs(row['Count'])

            full_matrix = pd.concat([matrix, full_matrix], axis=0)

    #
    #
    # Позиции со спредом
    #
    #

    else:
        prices_list = []
        for num, row in df.iterrows():
            current_price = row['underlying']
            position_dte = row['Days to EXP']
            row['rate'] = row['rate'] / 100
            side = row['Type']
            yahoo_stock = yf.download(row['Symb'])
            hist_vol = volatility_calc(yahoo_stock['Close'], 245)
            # print('hist_vol', hist_vol)

            # underlying_list = np.arange(current_price-(spread/2), current_price+spread, spread/0)
            underlying_list = foo(current_price-(spread*2), current_price+(spread*2), 10)
            # underlying_name = np.arange(-(spread/2), spread, spread/10)
            underlying_name = foo(-(spread*2), (spread*2), 10)
            prices_list.append(underlying_list)
            print('underlying_list', underlying_list)
            print('underlying_name', underlying_name)

            std_hv_step = (hist_vol[-245:].std() * 2) / 10
            max_under_std = yahoo_stock['Close'][-245:].std() * 2
            underlying_step = max_under_std / 10
            sigma_list = []
            sigma_name = []
            # underlying_list = []
            # underlying_name = []
            start_up = mean_sigma / 100
            start_down = mean_sigma / 100
            underlying_up = mean_underlying
            underlying_down = mean_underlying

            for i in reversed(range(10)):
                start_up = start_up + std_hv_step
                underlying_up = underlying_up + underlying_step
                sigma_list.append(start_up)
                # underlying_list.append(underlying_up)
                sigma_name.append(round((underlying_step / max_under_std * i + 1 - 1) * 2 + 0.2, 2))
                # underlying_name.append(round((underlying_step / max_under_std * i + 1 - 1) * 2 + 0.2, 2))

            sigma_list.append(mean_sigma / 100)
            sigma_list = sorted(sigma_list, reverse=True)
            sigma_name.append(0)
            # underlying_list.append(mean_underlying)
            # underlying_name.append(0)

            for i in range(10):
                start_down = start_down - std_hv_step
                underlying_down = underlying_down - underlying_step
                sigma_list.append(start_down)
                # underlying_list.append(underlying_down)
                sigma_name.append(-round((underlying_step / max_under_std * i + 1 - 1) * 2 + 0.2, 2))
                # underlying_name.append(-round((underlying_step / max_under_std * i + 1 - 1) * 2 + 0.2, 2))

            loop_df = pd.DataFrame({
                'IV': sigma_list,
                # 'underlying': underlying_list,
                'sigma_name': sigma_name,
                # 'underlying_name': underlying_name,
            })

            matrix = pd.DataFrame()

            for stock_price in sorted(underlying_list):
                black_df = pd.DataFrame()
                black_df['sigma'] = loop_df['IV']
                black_df['strike'] = row['strike']
                black_df['dte'] = (dte + position_dte - min_dte) * 1 / 365
                black_df['rate'] = row['rate'] / 100
                black_df['stock_price'] = stock_price

                if side == 'PUT':
                    option_prices = black76Put(black_df)
                elif side == 'CALL':
                    option_prices = black76Call(black_df)
                elif side == 'STOCK':
                    option_prices = black_df['stock_price']

                if side != 'STOCK':
                    if row['Count'] < 0:
                        return_matrix = row['prime'] - option_prices
                    if row['Count'] > 0:
                        return_matrix = option_prices - row['prime']
                else:
                    if row['Count'] < 0:
                        return_matrix = row['underlying'] - option_prices
                    if row['Count'] > 0:
                        return_matrix = option_prices - row['underlying']

                return_matrix = return_matrix.to_frame()
                matrix = pd.concat([matrix, return_matrix], axis=1)

            print(matrix)
            matrix.columns = underlying_name
            matrix['IV'] = loop_df['IV']

            # matrix = matrix.sort_values('IV')
            matrix = matrix.set_index('IV') * row['multiplier'] * np.abs(row['Count'])

            full_matrix = pd.concat([matrix, full_matrix], axis=0)



    full_matrix = full_matrix.groupby('IV').sum()

    full_matrix = full_matrix.set_index(loop_df.sort_values('sigma_name')['sigma_name'])
    full_matrix.columns = sorted(underlying_name)

    if full_matrix.min().min() > 0:
        pl_colorscale = 'greens'
    elif full_matrix.max().max() > 0:
        pl_colorscale = three_colorscale(full_matrix.to_numpy(), ["#DC3714", "#C0C0C0","#19BD1B"])
    else:
        pl_colorscale = 'Reds'
    # pl_colorscale = 'Reds'



    temp_df = pd.DataFrame({
        'Fist': prices_list[0],
        'Sec': prices_list[1]
    })

    temp_df = temp_df.astype('str')
    temp_df['X'] = temp_df['Fist'] + '_' + temp_df['Sec']

    fig_map = go.Figure(data=go.Heatmap(
        z=full_matrix.round(2).values.tolist(),
        x=temp_df['X'] .values.tolist(),
        y=full_matrix.index.values.tolist(),
        hoverongaps=False,
        colorscale = pl_colorscale,
        colorbar_thickness=24,
        xgap=0.5,
        ygap=0.5,
    )
    )

    fig_map.update_traces(

        hovertemplate='<br>'.join([
            'Price: %{x}',
            'IV: %{y}',
            'Values: %{z}',
        ])
    )

    fig_map.update_layout(title="Heatmap",
                          yaxis={"title": 'IV STD'},
                          xaxis={"title": 'Prices STD'},)

    # fig_map.show()

    weighted_profit = round(full_matrix[full_matrix > 0].mean(), 2).mean()
    weighted_loss = round(full_matrix[full_matrix < 0].mean(), 2).mean()
    weighted_rr = round(weighted_profit / np.abs(weighted_loss), 2).mean()
    if len(full_matrix[full_matrix < 0]) <= 0:
        weighted_loss = 0
        weighted_rr = round(weighted_profit, 2)

    print('weighted_rr', weighted_rr)
    print(full_matrix)
    return fig_map, weighted_profit, weighted_loss, weighted_rr



