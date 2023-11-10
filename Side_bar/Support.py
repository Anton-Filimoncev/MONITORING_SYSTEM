import pandas as pd
import time

def get_tick_from_csv_name(csv_position_df):
    ticker_name = csv_position_df.split("/")[-1].split(".")[0]
    return ticker_name

def create_new_postion(pos_type):
    if pos_type == 'Put Sell':
        time.sleep(2)
        pass

    return 'aaaaaaaaaaaaaaaaaa'

