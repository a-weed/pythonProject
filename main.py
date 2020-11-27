import pandas as pd
import quandl
import math

quandl.ApiConfig.api_key = 'U8bAyuBxWfoTb1LeVLGR'

df = quandl.get('EOD/MSFT')

df = df[['Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume',]]
df["HL_PCT"] = (df['Adj_High'] - df['Adj_Low']) / df['Adj_Low'] * 100.0
df["PCT_change"] = (df['Adj_Low'] - df['Adj_High']) / df['Adj_High'] * 100.0

df = df[['Adj_Close','HL_PCT','PCT_change','Adj_Volume']]

forecast_col = 'Adj_Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))


