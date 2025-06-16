import tensorflow as tf
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (Input, LSTM, Dense, Activation, RepeatVector, TimeDistributed, Lambda)  # type: ignore

def create_two_head_model(input_shape, lstm_units=30):
    """
    Build a dual-output LSTM model for quantile hedging.

    Inputs:
        input_shape (tuple)
        lstm_units (int): Number of LSTM units in each recurrent layer

    Returns:
        tf.keras.Model: A model with two outputs:
            - v0: scalar estimate of initial capital
            - delta: vector of hedge ratios (Δ) over time
    """
    inp = Input(shape=input_shape)

    # Shared LSTM encoder (compresses full path into one vector)
    x = LSTM(lstm_units, return_sequences=True)(inp)
    x = Activation('swish')(x)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Activation('swish')(x)

    # Head for V₀ (scalar)
    v0_features = Lambda(lambda t: t[:, 0, :])(x)  # shape: (batch_size, lstm_units)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(x)  # shape: (batch_size, lstm_units)
    v0 = Dense(1, activation='swish')(pooled)
    v0 = Activation('swish')(v0)

    # Head for Δ: decode from shared representation
    # Repeat shared vector T-1 times (to match ΔS_t count)
    delta_features = Lambda(lambda t: t[:, 1:, :])(x)  # shape: (batch_size, T-1, lstm_units)
    delta = TimeDistributed(Dense(1, activation='swish'))(delta_features)  # shape: (batch_size, T-1, 1)

    model = Model(inputs=inp, outputs=[v0, delta])
    return model

class QuantileHedgeModel(tf.keras.Model):
    """
    Custom model wrapper implementing the quantile hedging objective for European call Option.

    The loss is a combination of:
    - L1: Squared initial capital (encouraging lower capital usage)
    - L2: Penalty for shortfall probability using a sigmoid-based proxy

    Args:
        base_model (Model): A Keras model returning (v0, delta)
        mu (float): Penalty weight on shortfall risk
        beta (float): Sharpness of sigmoid for quantile loss approximation
    """
    def __init__(self, base_model, mu, beta):
        super().__init__()
        self.model = base_model
        self.mu = mu
        self.beta = beta

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            v0_pred, delta_pred = self.model(x, training=True)
            v0_pred = tf.squeeze(v0_pred, axis=-1)

            # Compute portfolio value
            price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]
            gains = tf.reduce_sum(delta_pred * price_incr, axis=[1, 2])
            portfolio = v0_pred + gains

            K = 100.0
            H = tf.maximum(y_true[:, -1, 0] - K, 0.0)

            def sigmoid_indicator(portfolio, H, beta):
                return tf.square(tf.maximum(tf.sigmoid(beta * (H - portfolio)) - 0.5, 0.0))

            L1 = tf.reduce_mean(tf.square(v0_pred))
            L2 = self.mu * tf.reduce_mean(sigmoid_indicator(portfolio, H, self.beta))
            total_loss = L1 + L2

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": total_loss}
