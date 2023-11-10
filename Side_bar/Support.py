import pandas as pd


def get_tick_from_csv_name(csv_position_df):
    ticker_name = csv_position_df.split("\\")[-1].split(".")[0]
    return ticker_name