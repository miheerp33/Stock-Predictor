
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
stock = yf.Ticker("BAC")
hist = stock.history(period='max', interval='1d')
testResult = []
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
d = {'Open': hist['Open'].tolist(),'Close': hist['Close'].tolist(), 'Result': hist['Result'].tolist()}
df = pd.DataFrame(data=d)
df['5MA'] = fivedayMovingAverage
df['10MA'] = tendayMovingAverage
df = df.drop(range(0, 9))
maDiff = []
prevmaDiff = []
prev2maDiff = []
for ind in df.index:
    maDiff.append(df['5MA'][ind] - df['10MA'][ind])
    if ind >= 10:
        prevmaDiff.append(df['5MA'][ind - 1] - df['10MA'][ind - 1])
    else:
        prevmaDiff.append(np.nan)
    if ind >= 11:
        prev2maDiff.append(df['5MA'][ind - 2] - df['10MA'][ind - 2])
    else:
        prev2maDiff.append(np.nan)
df['maDifference'] = maDiff
df['maDifference-1day'] = prevmaDiff
df['maDifference-2day'] = prev2maDiff
df = df[['maDifference', 'maDifference-1day', 'maDifference-2day', 'Result']]
df = df.drop(range(9, 11))
testX = df[['maDifference', 'maDifference-1day', 'maDifference-2day']]
testY = df[['Result']]
trainX, validateX, trainY, validateY = train_test_split(testX, testY, test_size=0.20, random_state=1)
model.fit(trainX, trainY.values.ravel())
modeltest(validateX, validateY)


