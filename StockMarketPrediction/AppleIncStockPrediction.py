import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


apple_inc_file = '/Users/eyosiasdesta/AIProjects/StockMarketPrediction/StockMarketPrediction/AAPL.csv'
df = pd.read_csv(apple_inc_file, parse_dates=['Date'], index_col='Date')

features = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

data = df[features]

scalar = MinMaxScaler()
sclaed_data = scalar.fit_transform(data)

X = sclaed_data[:, :-1]
y = sclaed_data[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Set: {mse}')


predictions = model.predict(X_test)

print(df.head())
