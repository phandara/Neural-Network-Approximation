import tensorflow as tf

import tensorflow as tf

def prob_hedge(y_true, y_pred):
    V0 = y_pred[:, 0, 0]  # Initial capital
    delta = y_pred[:, 1:, :]  # Hedging strategy: shape (batch, T, 1)
    price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]  # Price increments

    # Asian option payoff: max(avg(S) - K, 0)
    avg_price = tf.reduce_mean(y_true[:, 1:, 0], axis=1)  # Arithmetic average of S_t, t=1,...,T
    K = tf.constant(100.0, dtype=tf.float32)
    H = tf.maximum(avg_price - K, 0.0)

    pnl = tf.reduce_sum(delta * price_incr, axis=1)  # Shape (batch, 1)
    portfolio = V0 + tf.squeeze(pnl, axis=-1)  # Shape (batch,)

    return tf.reduce_mean(tf.where(portfolio >= H, 1.0, 0.0))

def predicted_price(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 0, 0])

    