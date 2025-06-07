import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from scipy.stats import norm

"""
Evaluate trained quantile hedging models on Heston data.

Steps:
1. Load pre-trained models for multiple penalty weights (mu).
2. Predict initial capital (V₀) and hedge ratios (Δ).
3. Compute final portfolio value and compare to payoff.
4. Assess hedge success probabilities.
5. Visualize results and compare with Monte Carlo - benchmark (imported from heston_monte_carlo.py).
"""

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.heston_architecture import create_two_head_model
from heston_monte_carlo import heston_monte_carlo

# Parameters
mu_values = [100, 1000, 10000, 50000, 75000]
model_dir = "models/Heston"
plot_dir = "plots/Heston"
os.makedirs(plot_dir, exist_ok=True)

# Load evaluation data
x_test = np.load("data/generated/Heston/x_test_heston.npy")
y_test = np.load("data/generated/Heston/y_test_heston.npy")
input_shape = x_test.shape[1:]

# Storage for results
V0_list = []
V0_std_list = []
prob_success_list = []

for mu in mu_values:
    print(f"\nEvaluating for mu = {mu}...")

    model = create_two_head_model(input_shape=input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    weight_path = os.path.join(model_dir, f"lstm_heston_mu_{mu}.weights.h5")
    if not os.path.exists(weight_path):
        print(f" Weights not found for mu = {mu}, skipping...")
        continue
    model.load_weights(weight_path)

    # Predict
    v0_pred, delta_pred = model.predict(x_test, verbose=0)
    V0 = v0_pred.squeeze()
    delta = delta_pred 

    V0_mean = np.mean(V0)
    V0_std = np.std(V0)
    V0_list.append(V0_mean)
    V0_std_list.append(V0_std)

    price_incr = y_test[:, 1:, :] - y_test[:, :-1, :]
    gains = np.sum(delta * price_incr, axis=(1, 2))
    portfolio = V0 + gains

    K = 100.0
    avg_price = np.mean(y_test[:, :, 0], axis=1)
    H = np.maximum(avg_price - K, 0.0)


    success_prob = np.mean(portfolio >= H)
    prob_success_list.append(success_prob)

    # Plot distribution
    plt.figure(figsize=(8, 6))
    plt.hist(portfolio - H, bins=50)
    plt.title(f"Portfolio - Asian Payoff, mu = {mu}")
    plt.xlabel("Portfolio - H")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"heston_portfolio_minus_H_mu_{mu}.png"))
    plt.close()

# Monte Carlo theoretical price
mc_price, mc_error = heston_monte_carlo()
print(f"\n MC Estimated Asian Option Price: {mc_price:.4f} ± {1.96 * mc_error:.4f}")

# Summary plots
plt.figure(figsize=(8, 6))
plt.plot(mu_values[:len(V0_list)], V0_list, marker='o', label='NN Approximation')
plt.axhline(mc_price, color='red', linestyle='--', label='MC Price')
plt.fill_between(mu_values[:len(V0_list)], mc_price - 1.96*mc_error, mc_price + 1.96*mc_error, 
                 color='red', alpha=0.2, label='95% CI')
plt.xlabel("Mu")
plt.ylabel("Initial Capital V0")
plt.title("Initial Capital vs. Mu (Heston)")
plt.ylim(bottom=0.25)  # Start y-axis at 0.25
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "heston_v0_vs_mu.png"))

plt.figure(figsize=(8, 6))
plt.plot(mu_values[:len(prob_success_list)], prob_success_list, marker='o')
plt.xlabel("Mu")
plt.ylabel("Success Probability")
plt.title("Success Probability vs. Mu (Heston)")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "heston_prob_vs_mu.png"))

plt.figure(figsize=(8, 6))
plt.plot(prob_success_list, V0_list, marker='o')
plt.xlabel("Success Probability")
plt.ylabel("Initial Capital V0")
plt.title("Pareto Frontier (Heston)")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "heston_pareto_frontier.png"))

# Save results
df = pd.DataFrame({
    "mu": mu_values[:len(V0_list)],
    "V0": V0_list,
    "V0_std": V0_std_list,
    "prob_success": prob_success_list
})
df.to_csv(os.path.join(model_dir, "heston_mu_results.csv"), index=False)

print("\n Heston evaluation complete.")
