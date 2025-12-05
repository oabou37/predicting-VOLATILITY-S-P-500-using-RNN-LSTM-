"""
This program aims to predict VOLATILITY S&P 500 (^VIX) time series using LSTM.
The data is downloaded directly from Yahoo Finance using yfinance library.
"""
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix = []
    for i in range(len(vectorSeries) - sequence_length + 1):
        matrix.append(vectorSeries[i:i + sequence_length])
    return matrix

# random seed
np.random.seed(1234)

# Download VIX data using yfinance
print("Downloading VIX data from Yahoo Finance...")
vix_data = yf.download('^VIX', start='2005-01-02', end='2016-09-27')

# Extract the adjusted close prices
vector_vix = vix_data['Close'].values.tolist()
print(f"Downloaded {len(vector_vix)} data points")

# parameters
sequence_length = 20

# convert the vector to a 2D matrix
matrix_vix = convertSeriesToMatrix(vector_vix, sequence_length)

# shift all data by mean
matrix_vix = np.array(matrix_vix)
shifted_value = matrix_vix.mean()
matrix_vix -= shifted_value
print("Data shape:", matrix_vix.shape)

# split dataset: 90% for training and 10% for testing
train_row = int(round(0.9 * matrix_vix.shape[0]))
train_set = matrix_vix[:train_row, :]

# shuffle the training set (but do not shuffle the test set)
np.random.shuffle(train_set)

# the training set
X_train = train_set[:, :-1]
# the last column is the true value to compute the mean-squared-error loss
y_train = train_set[:, -1]

# the test set
X_test = matrix_vix[train_row:, :-1]
y_test = matrix_vix[train_row:, -1]

# the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# build the model
model = Sequential()
# layer 1: LSTM
model.add(LSTM(input_dim=1, units=50, return_sequences=True))
model.add(Dropout(0.2))
# layer 2: LSTM
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
# layer 3: dense
# linear activation: a(x) = x
model.add(Dense(units=1, activation='linear'))

# compile the model
model.compile(loss="mse", optimizer="rmsprop")

# train the model
print("Training the model...")
model.fit(X_train, y_train, batch_size=512, epochs=50, validation_split=0.05, verbose=1)

# evaluate the result
test_mse = model.evaluate(X_test, y_test, verbose=1)
print(f'\nThe mean squared error (MSE) on the test data set is {test_mse:.3f} over {len(y_test)} test samples.')

# get the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples, 1))

# plot the results
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test + shifted_value, label='Actual VIX')
plt.plot(predicted_values + shifted_value, label='Predicted VIX')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.title('VIX Prediction using LSTM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output_prediction.jpg', bbox_inches='tight')
plt.show()

# save the result into txt file
test_result = np.column_stack((predicted_values + shifted_value, y_test + shifted_value))
np.savetxt('output_result.txt', test_result, header='Predicted,Actual', comments='')
print("Results saved to output_result.txt and output_prediction.jpg")
