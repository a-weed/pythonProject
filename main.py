import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split

quandl.ApiConfig.api_key = 'U8bAyuBxWfoTb1LeVLGR'

df = quandl.get('EOD/MSFT')

df = df[['Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume',]]
df["HL_PCT"] = (df['Adj_High'] - df['Adj_Low']) / df['Adj_Low'] * 100.0
df["PCT_change"] = (df['Adj_Low'] - df['Adj_High']) / df['Adj_High'] * 100.0

df = df[['Adj_Close','HL_PCT','PCT_change','Adj_Volume']]

forecast_col = 'Adj_Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

print(accuracy)


