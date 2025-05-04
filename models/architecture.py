import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import Adam

def create_lstm_model(input_shape, lstm_units=64, output_dim=1, learning_rate=1e-4):
    model = Sequential()

    # First LSTM Layer
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Activation('swish'))

    # Second LSTM Layer
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Activation('swish'))

    # Final Dense Layer + ReLU Activation to ensure V0 and deltas are non-negative
    model.add(Dense(output_dim))
    model.add(Activation('relu'))  # Ensures positive outputs

    # Compile
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=None)  # Loss set at training

    return model
