import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from concurrent.futures.thread import ThreadPoolExecutor
import os
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
import pickle

tick = 'NG'
exp_date = '9/25/2025'

with open('data.txt', 'r') as file:
    full_txt = file.read()

full_txt = full_txt[full_txt.index(tick):]
print(full_txt)
full_txt = full_txt[full_txt.index(exp_date):full_txt.index(exp_date)+60000]
# print(full_txt)
print(full_txt)
data_list = list(filter(len, full_txt.split('\n')))
print(data_list)

row_list = []
flag = 0
for value_num in range(len(data_list)):
    # try:
    print(data_list[value_num])
    print('00')
    if data_list[value_num] == '   IMPLIEDS-----------------------':
        flag += 1
    if flag <= 1:
        row_list.append(data_list[value_num])
    else:
        break
    # except:
    #     pass

print(row_list)

df = pd.DataFrame()
df[['STRIKE', 'CALL', 'VTY', 'DELTA', 'GAMMA', 'THETA', 'VEGA', 'PUT', 'PUTDEL', 'CVOL', 'PVOL']] = np.nan
print(df)
df.loc[0, ['STRIKE', 'CALL', 'VTY', 'DELTA', 'GAMMA', 'THETA', 'VEGA', 'PUT', 'PUTDEL', 'CVOL', 'PVOL']] = [2750,0.20,13.9,0.00,0.01,0.01,0.1716,36.30,-1.00,1,1]
print(df)
for num, val_row in enumerate(row_list):
    # print(num, val_row)
    if num == 1:
        IV = val_row[val_row.index('IV:'):].split(' ')[2]
    if num >= 5:
        try:
            df.loc[num, ['STRIKE', 'CALL', 'VTY', 'DELTA', 'GAMMA', 'THETA', 'VEGA', 'PUT', 'PUTDEL', 'CVOL', 'PVOL']] = val_row.strip().replace(' ', ',').replace(',,', ',').replace(',,', ',').replace(',,', ',').split(',')
        except:
            pass

df = df.dropna()
df = df.astype('float')
print(df[-22:])
print('IV:', IV)



