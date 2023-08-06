import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(42)
time = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.random.normal(loc=10, scale=2, size=100)

# Introduce an anomaly
values[75] = 20

# Create a pandas DataFrame with the time series data
data = pd.DataFrame({'Time': time, 'Value': values})
data.set_index('Time', inplace=True)

# Plot the original time series data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original Time Series Data')
plt.show()

# Calculate the rolling mean and rolling standard deviation
window_size = 7  # Adjust the window size based on the data characteristics
rolling_mean = data['Value'].rolling(window=window_size).mean()
rolling_std = data['Value'].rolling(window=window_size).std()

# Calculate the Z-Score for each data point
z_score = (data['Value'] - rolling_mean) / rolling_std

# Define a threshold for anomaly detection
threshold = 2.5  # Adjust the threshold based on the data characteristics

# Detect anomalies based on the Z-Score exceeding the threshold
anomalies = data[np.abs(z_score) > threshold]

# Plot the original data with detected anomalies
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Value'], label='Original Data')
plt.scatter(anomalies.index, anomalies['Value'], color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Anomaly Detection using Z-Score')
plt.legend()
plt.show()
