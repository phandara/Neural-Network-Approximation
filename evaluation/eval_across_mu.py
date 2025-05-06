import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.generator import DataGenerator
from models.architecture import create_lstm_model
from models.loss_function import augmented_quantile_loss
from models.log_loss_function import log_sigmoid_quantile_loss

# Parameters
mu_values = [10, 100, 1000, 3000, 5000, 6000, 7500]
model_dir = "models"
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Load evaluation data
x_test = np.load("data/generated/x_test.npy")
y_test = np.load("data/generated/y_test.npy")
input_shape = x_test.shape[1:]

# Storage for results
V0_list = []
prob_success_list = []

for mu in mu_values:
    print(f"\nEvaluating for mu = {mu}...")
    
    # Build and compile model
    model = create_lstm_model(input_shape=input_shape)
    loss_fn = augmented_quantile_loss(mu=mu)
    #loss_fn = log_sigmoid_quantile_loss(mu=mu)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss_fn)

    # Load pre-trained weights
    weight_path = os.path.join(model_dir, f"lstm_quantile_mu_{mu}.weights.h5")
    if not os.path.exists(weight_path):
        print(f"❌ Weights for mu={mu} not found at {weight_path}, skipping...")
        continue
    model.load_weights(weight_path)

    # Predict
    y_pred = model.predict(x_test, verbose=0)

    # Extract initial capital
    V0 = y_pred[:, 0, 0]
    V0_mean = np.mean(V0)
    V0_list.append(V0_mean)

    # Compute portfolio
    delta = y_pred[:, 1:, :]
    price_incr = y_test[:, 1:, :] - y_test[:, :-1, :]
    gains = np.sum(delta * price_incr, axis=(1, 2))
    portfolio = V0 + gains

    # Payoff
    K = 100.0
    H = np.maximum(y_test[:, -1, 0] - K, 0.0)

    # Probability of successful hedge
    success_prob = np.mean(portfolio >= H)
    prob_success_list.append(success_prob)

# ---------- Plots ----------
plt.figure(figsize=(8, 6))
plt.plot(mu_values[:len(V0_list)], V0_list, marker='o')
plt.xlabel("Mu (penalty weight)")
plt.ylabel("Initial Capital V0")
plt.title("Initial Capital vs. Mu")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "v0_vs_mu.png"))

plt.figure(figsize=(8, 6))
plt.plot(mu_values[:len(prob_success_list)], prob_success_list, marker='o')
plt.xlabel("Mu (penalty weight)")
plt.ylabel("Success Probability")
plt.title("Success Probability vs. Mu")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "prob_vs_mu.png"))

plt.figure(figsize=(8, 6))
plt.plot(prob_success_list, V0_list, marker='o')
plt.xlabel("Success Probability")
plt.ylabel("Initial Capital V0")
plt.title("Pareto Frontier")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "pareto_frontier.png"))

# ---------- Save Results ----------
df = pd.DataFrame({
    "mu": mu_values[:len(V0_list)],
    "V0": V0_list,
    "prob_success": prob_success_list
})
df.to_csv(os.path.join(model_dir, "mu_results.csv"), index=False)

print("\n✅ Evaluation complete. Results saved.")

# Print helpful debug stats
print(f"Mean Initial Capital (V0): {V0.mean():.4f}")
print(f"Success Probability (portfolio ≥ H): {(portfolio >= H).mean():.4f}")
print(f"Min/Max portfolio: {portfolio.min():.2f} / {portfolio.max():.2f}")
print(f"Min/Max H: {H.min():.2f} / {H.max():.2f}")