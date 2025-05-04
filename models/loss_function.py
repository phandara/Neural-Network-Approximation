import tensorflow as tf


def augmented_quantile_loss(q_target: float, mu: float):
    """
    Custom loss function for quantile hedging using an augmented Lagrangian style penalty.

    L = |V_0|^2 + mu * max(0, q* - P(success))^2
    
    where success means: portfolio >= H
    
    Args:
        q_target (float): Desired quantile hedging success probability (e.g. 0.95)
        mu (float): Penalty weight for constraint violation
    Returns:
        A callable loss function
    """

    def loss(y_true, y_pred):
        # Extract initial capital prediction (V0)
        V0 = y_pred[:, 0, 0]  # shape (batch_size,)

        # Extract delta strategies (shape: batch, time-1, assets)
        delta = y_pred[:, 1:, :]  

        # Compute price increments
        price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]  # shape: (batch, time-1, assets)

        # Final payoff: max(S_T - K, 0), assume K = 100 fixed
        K = tf.constant(100.0, dtype=tf.float32)
        H = tf.maximum(y_true[:, -1, 0] - K, 0.0)  # shape: (batch,)

        # Portfolio value = V0 + sum_k delta_k * (X_k - X_{k-1})
        gains = tf.reduce_sum(tf.reduce_sum(delta * price_incr, axis=2), axis=1)  # shape: (batch,)
        portfolio = V0 + gains

        # Success indicator: soft proxy using sigmoid approximation
        success_prob = tf.reduce_mean(tf.cast(portfolio >= H, tf.float32))

        # First term: squared initial capital
        L1 = tf.reduce_mean(tf.square(V0))

        # Second term: penalty for not reaching q_target
        constraint_violation = tf.maximum(0.0, q_target - success_prob)
        L2 = mu * tf.square(constraint_violation)

        return L1 + L2

    return loss
