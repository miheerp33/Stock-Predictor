
import numpy as np

from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

import yfinance as yf
model = KNeighborsClassifier()

def modeltest(testX, testY):
    predictions = model.predict(testX)

    for i in range(len(predictions)):
        print('Predicted: ', predictions[i], '||| Actual: ',testY['Result'].tolist()[i])
    print('Accuracy:', accuracy_score(testY.values.ravel(), predictions), 'in ', len(predictions), 'rows.')
aapl = yf.Ticker("VOO")
hist = aapl.history(period='ytd',interval='1d')
testResult = []
datesDict = {}
for ind in hist.index:


        if hist['Close'][ind] > hist['Open'][ind]:
            testResult.append('Higher')
        else:
            testResult.append('Lower')

hist['Result'] = testResult

openList = hist['Open'].tolist()
openSeries = pd.Series(openList)
tenday_windows = openSeries.rolling(10)
fiveday_windows = openSeries.rolling(5)
tendayMovingAverage = tenday_windows.mean()
fivedayMovingAverage = fiveday_windows.mean()

numlist = []
for i in range(len(hist)):
    numlist.append(i)
b = 0
for ind in hist.index:
    datesDict[b] = ind
    b += 1


d = {'Open': hist['Open'].tolist(),'Close': hist['Close'].tolist(), 'Result': hist['Result'].tolist()}
df2 = pd.DataFrame(data=d)
df2['5MA'] = fivedayMovingAverage
df2['10MA'] = tendayMovingAverage
df2 = df2.drop(range(0,9))


test_maDiff = []
prevtest_maDiff = []
prev2test_maDiff = []
for ind in df2.index:

    test_maDiff.append(df2['5MA'][ind]-df2['10MA'][ind])
    if ind >= 10:
        prevtest_maDiff.append(df2['5MA'][ind-1]-df2['10MA'][ind-1])
    else:
        prevtest_maDiff.append(np.nan)
    if ind >= 11:
        prev2test_maDiff.append(df2['5MA'][ind-2]-df2['10MA'][ind-2])
    else:
        prev2test_maDiff.append(np.nan)
df2['maDifference'] = test_maDiff
df2['maDifference-1day'] = prevtest_maDiff
df2['maDifference-2day'] = prev2test_maDiff
df2 = df2[['maDifference', 'maDifference-1day', 'maDifference-2day','Result']]
df2 = df2.drop(range(9,11))

AAPL_testX = df2[['maDifference', 'maDifference-1day', 'maDifference-2day']]
AAPL_testY = df2[['Result']]

AAPL_trainX, AAPL_validateX, AAPL_trainY, AAPL_validateY = train_test_split(AAPL_testX, AAPL_testY, test_size=0.20, random_state=1)
model.fit(AAPL_trainX,AAPL_trainY.values.ravel())

modeltest(AAPL_validateX, AAPL_validateY)


