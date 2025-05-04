import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.generator import DataGenerator
from models.architecture import create_lstm_model

# -------------------------
# Load trained model weights
# -------------------------
weights_path = "models/lstm_quantile_bs.weights.h5"
assert os.path.exists(weights_path), "Weights file not found."

# -------------------------
# Data generation
# -------------------------
generator = DataGenerator(num_samples=1000, time_steps=30)
x_train, x_test, y_train, y_test = generator.generate_data()

# -------------------------
# Model setup
# -------------------------
input_shape = x_train.shape[1:]  # (time_steps, 1)
model = create_lstm_model(input_shape=input_shape)
model.load_weights(weights_path)

# -------------------------
# Predict on test set
# -------------------------
y_pred = model.predict(x_test)  # shape: (batch, time, 1)

# -------------------------
# Compute superhedging probability
# -------------------------
# Initial capital
V0 = y_pred[:, 0, 0]

# Trading strategy (delta)
delta = y_pred[:, 1:, :]  # shape: (batch, time-1, 1)

# Price increments
price_incr = y_test[:, 1:, :] - y_test[:, :-1, :]

# Portfolio terminal value
gains = tf.reduce_sum(delta * price_incr, axis=[1, 2])
portfolio = V0 + gains

# Payoff: H = max(S_T - K, 0)
K = 100.0
H = np.maximum(y_test[:, -1, 0] - K, 0)

# Probability of hedge
success_prob = np.mean(portfolio >= H)
print(f"Superhedging Probability: {success_prob:.4f}")
print(f"Average Initial Capital: {np.mean(V0):.4f}")

# -------------------------
# Plot histogram of portfolio surplus/deficit
# -------------------------
surplus = portfolio - H

plt.figure(figsize=(10, 6))
plt.hist(surplus, bins=50, alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', label='Breach Threshold')
plt.title("Histogram of Portfolio Surplus (Portfolio - Payoff)")
plt.xlabel("Surplus")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/portfolio_surplus_hist.png")
plt.show()
