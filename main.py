import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# 設定起始日期和結束日期
start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime.now()

# 下載 AAPL 的股票數據
aapl_data = yf.download('AAPL', start=start_date, end=end_date)

# 繪製股價折線圖
plt.plot(aapl_data['Close'])
plt.title('AAPL Stock Price History')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# 計算 5 日移動平均線並加入到 DataFrame 中
aapl_data['MA5'] = aapl_data['Close'].rolling(window=5).mean()
# 刪除缺失值
aapl_data = aapl_data.dropna()

# 分割資料，將 80% 作為訓練集，20% 作為測試集
X = aapl_data[['MA5']]
y = aapl_data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 建立線性回歸模型，並使用訓練集進行訓練
lr = LinearRegression()
lr.fit(X_train, y_train)

# 評估模型表現
print('R²: %.2f' % lr.score(X_test, y_test))

# 預測未來 10 天的股價
last_date = aapl_data.index[-1]
# 產生未來 10 天的日期
prediction_dates = pd.date_range(last_date, periods=10, freq='D')
# 建立一個新的 DataFrame 來儲存預測的股價
next_data = pd.DataFrame(index=prediction_dates, columns=aapl_data.columns)
# 計算最後 5 個收盤價的移動平均，作為新的 5 日移動平均線
next_data['MA5'] = aapl_data['Close'].rolling(window=5).mean().tail(10).values
# 使用建立好的線性回歸模型進行預測
next_data['Close'] = lr.predict(next_data[['MA5']])
# 輸出預測結果
print(next_data['Close'])
