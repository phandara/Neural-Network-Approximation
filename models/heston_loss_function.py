import tensorflow as tf

def augmented_quantile_loss_heston(mu: float = 100):
    
    def sigmoid_indicator(portfolio, H, beta=1.0):
        return tf.square(tf.maximum(tf.sigmoid(beta * (H - portfolio)) - 0.5, 0.0))

    def loss(y_true, y_pred):
        V0 = y_pred[:, 0, 0]
        delta = y_pred[:, 1:, :]
        price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]
        # Payoff
        K = 100
        avg_price = tf.reduce_mean(y_true[:, 1:, 0], axis=1)
        H = tf.maximum(avg_price - K, 0.0)

        gains = tf.reduce_sum(delta * price_incr, axis=[1,2])
        portfolio = V0 + gains
        
        L1 = tf.reduce_mean(tf.square(V0))
        L2 = mu * tf.reduce_mean(sigmoid_indicator(portfolio, H))

        return L1 + L2

    return loss


