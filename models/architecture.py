import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import Adam


def create_lstm_model(input_shape, lstm_units=30, output_dim=1, learning_rate=1e-4):
    model = Sequential()

    # First LSTM Layer
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Activation('swish'))

    # Second LSTM Layer
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Activation('swish'))

    # Final Dense Layer
    model.add(Dense(output_dim))
    model.add(Activation('swish'))

    # Compile
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=None)  # Loss will be added separately when calling model.fit()

    return model
