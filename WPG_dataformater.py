import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from pandas import ExcelWriter
import openpyxl


#-----------------------------------Data pre-process-------------------------------------------------------
# get  training data
df = pd.read_excel('./data/WPG_data_test.xlsx')

# -----------------------------------input data random pre-process------------------------------------
df_ran = df.sample(frac=1)

# define which title to be noimal
df_nominal = df_ran.ix[:, ['Report Date', 'Customer',
                           'Type', 'Item Short Name', 'Brand', 'Sales']]
df_numerical_tmp = df_ran.ix[:, ['Actual AWU', 'Avail.',
                                 'BL <= 9WKs', 'Backlog', 'DC OH', 'Hub OH', 'TTL OH', 'FCST M']]

df_numerical = df_numerical_tmp.apply(
    pd.to_numeric, errors='coerce').fillna(-1)

result = pd.concat([df_nominal, df_numerical], axis=1)

result.to_csv('./data/WPG.csv', index=False)
