import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def augmented_quantile_loss(mu=100):

    def sigmoid_indicator(portfolio, H, beta=0.5):
        return tf.square(tf.maximum(tf.sigmoid(beta * (H - portfolio)) - 0.5, 0.0))

    def loss(y_true, y_pred):
        v0_pred = y_pred[0]
        delta_pred = y_pred[1]
        v0_pred = tf.squeeze(v0_pred, axis=-1)  # (batch_size,)
        
        price_incr = y_true[:, 1:, :] - y_true[:, :-1, :]  # (batch_size, T-1, 1)
        gains = tf.reduce_sum(delta_pred * price_incr, axis=[1, 2])  # (batch_size,)
        portfolio = v0_pred + gains

        K = 100.0
        H = tf.maximum(y_true[:, -1, 0] - K, 0.0)

        L1 = tf.reduce_mean(tf.square(v0_pred))
        L2 = mu * tf.reduce_mean(sigmoid_indicator(portfolio, H))
        return L1 + L2

    return loss


# Plotting the scaled truncated sigmoid loss
if __name__ == "__main__":
    x = np.linspace(-10, 10, 500)
    betas = [0.5, 1.0, 2.0, 5.0]
    
    def l_beta(x, beta):
        sigmoid_term = 1 / (1 + np.exp(-beta * x))
        return np.square(np.maximum(sigmoid_term - 0.5, 0.0))
    
    plt.figure(figsize=(8, 6))
    for beta in betas:
        y = l_beta(x, beta)
        plt.plot(x, y, label=f"$\\beta = {beta}$")

    plt.title("Scaled Truncated Sigmoid Loss $l_\\beta(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$l_\\beta(x)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/loss_function_plot.png", dpi=300)
    plt.close()

