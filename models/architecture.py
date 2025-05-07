import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Activation # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def create_lstm_model(input_shape, lstm_units=30, output_dim=1, learning_rate=1e-4,):
    model = Sequential()

    # First LSTM Layer
    model.add(LSTM(units=lstm_units,
                   return_sequences=True,
                   input_shape=input_shape,
                   kernel_initializer='TruncatedNormal',
                   bias_initializer='TruncatedNormal'))
    model.add(Activation('swish'))

    # Second LSTM Layer
    model.add(LSTM(units=lstm_units,
                   return_sequences=True))
    model.add(Activation('swish'))

    # Final Dense Layer
    model.add(Dense(output_dim))
    model.add(Activation('relu'))

    # Compiling in training

    return model

