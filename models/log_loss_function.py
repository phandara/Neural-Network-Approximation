import tensorflow as tf
# more sensitive to small success probabilities
def log_sigmoid_quantile_loss(mu: float = 1000.0, beta: float = 1.0):
    
    # L = |V_0|^2 - mu * log(P(portfolio >= H))
    # with sigmoid used to approximate indicator function for success probability
    
    def loss(y_true, y_pred):
        # Extract initial capital
        V0 = y_pred[:, 0, 0]  # shape: (batch,)
        delta = y_pred[:, 1:, :]  # shape: (batch, T-1, 1)

        # Compute price increments
        price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]  # (batch, T-1, 1)

        # Payoff: H = max(mean(S_t) - K, 0)
        K = tf.constant(100.0, dtype=tf.float32)
        # Compute average price
        avg_price = tf.reduce_mean(y_true[:, 1:, 0], axis=1)
        H = tf.maximum(avg_price - K, 0.0)
        

        # Portfolio value
        gains = tf.reduce_sum(delta * price_incr, axis=[1, 2])
        portfolio = V0 + gains

        # Smooth success probability
        success_prob = tf.reduce_mean(tf.sigmoid(beta * (portfolio - H)))

        # Loss terms
        L1 = tf.reduce_mean(tf.square(V0))
        L2 = mu * tf.math.log(success_prob + 1e-8)  # Add epsilon to avoid log(0)

        return L1 - L2  # Subtract since log(success_prob) increases with success

    return loss
