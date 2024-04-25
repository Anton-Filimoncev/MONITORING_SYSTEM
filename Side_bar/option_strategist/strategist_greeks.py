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


start_df_name = 'Regime_ALL.xlsx'

def greeks():
    barcode = 'dranatom'
    password = 'MSIGX660'
    # ____________________ Работа с Selenium ____________________________
    # path = os.path.join(os.getcwd(), 'chromedriver.exe')
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--start-maximized")
    # chrome_options.add_argument('headless')

    checker = webdriver.Chrome(options=chrome_options)

    checker.get(f'https://www.optionstrategist.com/subscriber-content/greeks#futures')
    sleep(2)

    sign_in_userName = checker.find_element(by=By.ID, value="edit-name")
    sign_in_userName.send_keys(barcode)
    sign_in_password = checker.find_element(by=By.ID, value="edit-pass")
    sign_in_password.send_keys(password)
    sign_in = checker.find_element(by=By.ID,
                                   value='''edit-submit''')
    sign_in.click()
    sleep(5)
    try:
        close_popup = checker.find_element(By.XPATH, '//*[@id="PopupSignupForm_0"]/div[2]/div[1]')
        close_popup.click()
    except:
        pass

    try:
        close_popup = checker.find_element(By.XPATH, '//*[@id="PopupSignupForm_0"]/div[2]/button')
        close_popup.click()
    except:
        pass
    #

    select_fresh_txt = checker.find_element(By.XPATH,
                                            '''//*[@id="node-38"]/div/div/div/div/table[2]/tbody/tr[1]/td[1]''')

    select_fresh_txt.click()


    html_txt = checker.page_source
    # print("The current url is:" + str(checker.page_source))

    full_txt = (html_txt[:html_txt.rindex('</pre>')]).replace('* ', '').replace(
        '^ ', '')

    # full_txt = (html_txt[html_txt.index('800-724-1817'):html_txt.rindex('</pre>')]).replace('* ', '').replace(
    #     '^ ', '')

    # full_txt = full_txt

    print(full_txt)


    # with open('data.txt', 'r') as file:
    #     full_txt = file.read()

    # col_name_replace = 'Symbol (option symbols)           hv20  hv50 hv100    DATE   curiv Days/Percentile Close'
    # full_txt = full_txt.replace(col_name_replace, '').replace('\n', ' ')

    # with open('data.txt', 'w') as f:
    #     f.write(full_txt)

    return full_txt


def hist_vol_analysis(full_txt, ticker, exp_date):
    full_txt = full_txt[full_txt.index(ticker):]

    full_txt = full_txt[full_txt.index(exp_date):full_txt.index(exp_date) + 60000]
    # print(full_txt)

    data_list = list(filter(len, full_txt.split('\n')))

    row_list = []
    flag = 0
    for value_num in range(len(data_list)):
        # try:
        if data_list[value_num] == '   IMPLIEDS-----------------------':
            flag += 1
        if flag <= 1:
            row_list.append(data_list[value_num])
        else:
            break
        # except:
        #     pass

    df = pd.DataFrame()
    df[['STRIKE', 'CALL', 'VTY', 'DELTA', 'GAMMA', 'THETA', 'VEGA', 'PUT', 'PUTDEL', 'CVOL', 'PVOL']] = np.nan

    for num, val_row in enumerate(row_list):
        # print(num, val_row)
        if num == 1:
            IV = val_row[val_row.index('IV:'):].split(' ')[2]
        if num >= 5:
            try:
                df.loc[num, ['STRIKE', 'CALL', 'VTY', 'DELTA', 'GAMMA', 'THETA', 'VEGA', 'PUT', 'PUTDEL', 'CVOL',
                             'PVOL']] = val_row.strip().replace(' ', ',').replace(',,', ',').replace(',,', ',').replace(
                    ',,', ',').split(',')
            except:
                pass

    df = df.dropna()
    df = df.astype('float')
    print(df)
    print('IV:', IV)

    return df, IV

def greeks_start(ticker, exp_date):
    full_txt = greeks()
    history_vol = hist_vol_analysis(full_txt, ticker, exp_date)
    return history_vol


hist_vol_df = greeks_start()



