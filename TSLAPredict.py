import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import yfinance as yf

# Load data from CSV file
data = pd.read_csv('TSLA_Data.csv')

data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Feature engineer other variables
data['Price_Range'] = data['High'] - data['Low']
data['Daily_Change'] = data['Close'] - data['Open']

data['Next_Day_Close'] = data['Close'].shift(-1)
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data['Log_Return_Volatility'] = data['Log_Return'].rolling(window=2).std() * np.sqrt(252)
data['Pct_Change'] = data['Close'].pct_change() * 100
data['daily_return'] = data['Close'].pct_change()

data['Prev_Close'] = data['Close'].shift(1)

data['True_Range'] = np.maximum(data['High'] - data['Low'],
                                np.maximum(abs(data['High'] - data['Prev_Close']),
                                           abs(data['Low'] - data['Prev_Close'])))


data['ATR'] = data['True_Range'].rolling(window=2).mean()

data = data.drop(columns=['Prev_Close'])

latest_row = data.iloc[-1].copy()
latest_row['Next_Day_Close'] = -1

data.dropna(inplace=True)

latest_row_df = pd.DataFrame([latest_row])
data = pd.concat([data, latest_row_df], ignore_index=True)

# Select features and target
features = ['Open', 'Close', 'High', 'Low', 'Volume', 'Pct_Change', 'Log_Return', 'Log_Return_Volatility', 'ATR']
target = 'Next_Day_Close'

# Normalize features and target
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
data[features] = x_scaler.fit_transform(data[features])
y = y_scaler.fit_transform(data['Next_Day_Close'].values.reshape(-1, 1))

# Create sequences
X = []
y_seq = []
sequence_length = 3

for i in range(len(data) - sequence_length):
    X.append(data[features].iloc[i:i+sequence_length].values)
    y_seq.append(y[i+sequence_length])

# Use the last available data to predict the next closing price
latest_data = data[features].iloc[-sequence_length:].values
latest_data = latest_data.reshape((1, sequence_length, len(features)))

X = np.array(X)
y_seq = np.array(y_seq)
latest_data = np.array(latest_data)

X_train = X
y_train = y_seq

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, len(features))))
model.add(Dropout(0.3))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile Model
optimizer = Adam(learning_rate=0.00025)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train with early stopping
early_stopping = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])

print("Model training complete!")

# Make prediction for next days closing price
next_day_pred = model.predict(latest_data)

# Unscale the prediction to get actual values
next_day_price = y_scaler.inverse_transform(next_day_pred)


# Commented since yfinance has received too many reqs
# Grabs current price of TSLA stock as of running the code
"""
# Fetch the stock data
stock = yf.Ticker('TSLA')

# Get the latest market data (1 minute interval)
stock_data = stock.history(period="1d", interval="1m")

# Get the most recent closing price
current_price = stock_data['Close'][-1]

print(f"The current price of TSLA is: ${current_price}")
"""
print(f"Predicted next day's closing price: {next_day_price[0][0]}")

# Create sequences for the entire dataset with corresponding dates
X_full = []
y_actual = []
dates = []

for i in range(len(data) - sequence_length):
    X_full.append(data[features].iloc[i:i+sequence_length].values)
    y_actual.append(y[i+sequence_length])  # Actual values
    dates.append(data['Date'].iloc[i+sequence_length])  # Store corresponding date

X_full = np.array(X_full)
y_actual = np.array(y_actual)

# Predict on the entire dataset
y_pred_scaled = model.predict(X_full)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_actual = y_scaler.inverse_transform(y_actual.reshape(-1, 1))

# Predict the next day's price
latest_data = data[features].iloc[-sequence_length:].values
latest_data = latest_data.reshape((1, sequence_length, len(features)))

next_day_pred_scaled = model.predict(latest_data)
next_day_pred = y_scaler.inverse_transform(next_day_pred_scaled)[0][0]

# Append next day's prediction to the plot
next_day_date = data['Date'].iloc[-1] + pd.Timedelta(days=1)
dates.append(next_day_date)
y_pred = np.append(y_pred, next_day_pred)
y_actual = np.append(y_actual, np.nan)

# Plot actual vs predicted prices
# note: price day after present is zero since we don't know what it is
plt.figure(figsize=(12, 6))
plt.plot(dates[:-1], y_actual[:-1], label="Actual Closing Price", color='blue', linewidth=1)
plt.plot(dates, y_pred, label="Predicted Closing Price", color='red', linewidth=1)
plt.scatter([next_day_date], [next_day_pred], color='green', label="Next Day Prediction", marker='o', s=100)
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual vs. Predicted Closing Prices (Including Next Day)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

print(f"Predicted closing price for {next_day_date.strftime('%Y-%m-%d')}: ${next_day_pred:.2f}")