import tensorflow as tf

def prob_hedge(y_true, y_pred):
    V0 = y_pred[:, 0, 0]
    delta = y_pred[:, 1:, :]
    price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]
    K = tf.constant(100.0, dtype=tf.float32)
    H = tf.maximum(y_true[:, -1, 0] - K, 0.0)
    portfolio = V0 + tf.reduce_sum(tf.reduce_sum(delta * price_incr, axis=2), axis=1)
    return tf.reduce_mean(tf.where(portfolio >=H, 1.0, 0.0))

def predicted_price(y_true, y_pred):
    return tf.reduce_mean(y_pred[:, 0, 0])

    