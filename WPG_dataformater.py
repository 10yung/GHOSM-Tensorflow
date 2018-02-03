import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from pandas import ExcelWriter
from sklearn.preprocessing import MinMaxScaler
import openpyxl


#-----------------------------------Data pre-process-------------------------------------------------------
# get  training data
df = pd.read_excel('./data/WPG_data_test.xlsx')

print('finish reading input file')

# define which title to be noimal
df_nominal = df.ix[:, ['Report Date', 'Customer',
                           'Type', 'Item Short Name', 'Brand', 'Sales']]
df_numerical_tmp = df.ix[:, ['Actual AWU', 'Avail.',
                                 'BL <= 9WKs', 'Backlog', 'DC OH', 'Hub OH', 'TTL OH', 'FCST M']]

df_selected_label = df.ix[:, ['Actual AWU', 'Backlog']]
df_selected_label = df_selected_label.div(df['Avail.'], axis='index')
df_selected_label.columns = ['AWU_Avial_ratio', 'Backlog_Avail_ratio']


df_numerical = df_numerical_tmp.apply(
    pd.to_numeric, errors='coerce').fillna(-1)

scaler = MinMaxScaler()

df_numerical = pd.DataFrame(scaler.fit_transform(df_numerical), columns =['Actual AWU', 'Avail.',
                                 'BL <= 9WKs', 'Backlog', 'DC OH', 'Hub OH', 'TTL OH', 'FCST M'])


result = pd.concat([df_nominal, df_selected_label, df_numerical], axis=1)
result.to_excel('./data/WPG_test.xlsx', index=False)
