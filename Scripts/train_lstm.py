import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv(r"C:\Users\Suyamburajan A\PycharmProjects\traffic\venv\metr_queue.csv", parse_dates=['timestamp'], index_col='timestamp')

# Use only sensor columns
data = df.values  # shape: (time_steps, 4)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_dataset(dataset, lookback=12):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i+lookback])
        y.append(dataset[i+lookback])
    return np.array(X), np.array(y)

lookback = 12  # e.g., use past 1 hour (12 x 5min)
X, y = create_dataset(scaled_data, lookback)

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(lookback, 4)))
model.add(Dense(4))  # one output per sensor
model.compile(loss='mse', optimizer='adam')
model.summary()

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Predict
predicted = model.predict(X_test)
# Only scale back the predictions
# Inverse scale predictions
predicted_inverse = scaler.inverse_transform(predicted)
true_inverse = scaler.inverse_transform(y_test)
# Save model & predictions
model.save("model.h5")
pd.DataFrame(predicted_inverse, columns=["lane_1", "lane_2", "lane_3", "lane_4"]).to_csv("forecast.csv", index=False)

print("✅ Forecast saved to forecast.csv")

# Plot
plt.plot(y_test[:, 0], label="True Sensor 0")
plt.plot(predicted[:, 0], label="Predicted Sensor 0")
plt.legend()
plt.show()
