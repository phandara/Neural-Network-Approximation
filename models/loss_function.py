import tensorflow as tf

def augmented_quantile_loss(q_target: float = 1, mu: float = 100):
    # L = |V_0|^2 + mu * max(0, q* - P(portfolio >= H))^2
    def sigmoid_indicator(portfolio, H, beta=10.0):
        return tf.square(tf.maximum(tf.sigmoid(beta * (portfolio - H)) - 0.5, 0.0))

    def loss(y_true, y_pred):
        V0 = y_pred[:, 0, 0]
        delta = y_pred[:, 1:, :]
        price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]
        K = tf.constant(100.0, dtype=tf.float32)
        H = tf.maximum(y_true[:, -1, 0] - K, 0.0)

        gains = tf.reduce_sum(tf.reduce_sum(delta * price_incr, axis=2), axis=1)
        portfolio = V0 + gains

        success_prob = tf.reduce_mean(sigmoid_indicator(portfolio, H))
        L1 = tf.reduce_mean(tf.square(V0))
        L2 = mu * tf.square(tf.maximum(0.0, q_target - success_prob))

        return L1 + L2

    return loss
