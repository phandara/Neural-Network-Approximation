import tensorflow as tf

def augmented_quantile_loss(mu: float = 100):
    
    def sigmoid_indicator(portfolio, H, beta=10.0):
        return tf.square(tf.maximum(tf.sigmoid(beta * (portfolio - H)) - 0.5, 0.0))

    def loss(y_true, y_pred):
        V0 = y_pred[:, 0, 0]
        delta = y_pred[:, 1:, :]
        price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]
        # Payoff
        K = tf.constant(100.0, dtype=tf.float32)
        H = tf.maximum(y_true[:, -1, 0] - K, 0.0)

        gains = tf.reduce_sum(tf.reduce_sum(delta * price_incr, axis=2), axis=1)
        portfolio = V0 + gains

        
        L1 = tf.reduce_mean(tf.square(V0))
        L2 = mu * tf.reduce_mean(sigmoid_indicator(portfolio, H))

        return L1 + L2

    return loss


