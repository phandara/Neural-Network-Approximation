import tensorflow as tf

def augmented_quantile_loss(mu: float):
    
    #Loss = |V_0|^2 + mu * P(portfolio >= H)
    
    def sigmoid_indicator(portfolio, H, beta=50.0):
        return tf.square(tf.maximum(tf.sigmoid(beta * (portfolio - H)) - 0.5, 0.0))

    def loss(y_true, y_pred):
        # Initial capital prediction (V0)
        V0 = y_pred[:, 0, 0]

        # Trading strategy deltas
        delta = y_pred[:, 1:, :]

        # Price increments
        price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]

        # Payoff: max(S_T - K, 0)
        K = tf.constant(100.0, dtype=tf.float32)
        H = tf.maximum(y_true[:, -1, 0] - K, 0.0)

        # Portfolio value
        gains = tf.reduce_sum(tf.reduce_sum(delta * price_incr, axis=2), axis=1)
        portfolio = V0 + gains

        # Soft success proxy
        success_prob = tf.reduce_mean(sigmoid_indicator(portfolio, H))

        # Loss components
        L1 = tf.reduce_mean(tf.square(V0))
        L2 = mu * success_prob

        return L1 + L2

    return loss
