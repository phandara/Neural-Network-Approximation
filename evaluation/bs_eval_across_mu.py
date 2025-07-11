import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import pandas as pd
from scipy.stats import norm

"""
Evaluate trained quantile hedging models on Black-Scholes data.

Steps:
1. Load pre-trained models for multiple penalty weights (mu).
2. Predict initial capital (V₀) and hedge ratios (Δ).
3. Compute final portfolio value and compare to payoff.
4. Assess hedge success probabilities.
5. Visualize results and compare with Black–Scholes benchmark.
"""

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.architecture import create_two_head_model

# Parameters
mu_values = [10, 100, 500, 1000, 3000, 5000, 7500, 15000]
model_dir = "models/BS"
plot_dir = "plots/BS"
os.makedirs(plot_dir, exist_ok=True)

# Load evaluation data
x_test = np.load("data/generated/BS/x_test.npy")
y_test = np.load("data/generated/BS/y_test.npy")
input_shape = x_test.shape[1:]

# Storage for results
V0_list = []
V0_std_list = []
prob_success_list = []

for mu in mu_values:
    print(f"\nEvaluating for mu = {mu}...")
    # Build and compile model
    model = create_two_head_model(input_shape=input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))#,loss=loss_fn, metrics=metrics_fn

    # Load pre-trained weights
    weight_path = os.path.join(model_dir, f"lstm_quantile_mu_{mu}.weights.h5")
    if not os.path.exists(weight_path):
        print(f" Weights for mu={mu} not found at {weight_path}, skipping...")
        continue
    model.load_weights(weight_path)

    # Predict
    v0_pred, delta_pred = model.predict(x_test, verbose=0)
    V0 = v0_pred.squeeze()                    # (batch_size,)
    delta = delta_pred 

    V0_mean = np.mean(V0)
    V0_std = np.std(V0)
    V0_list.append(V0_mean)
    V0_std_list.append(V0_std)

    # Compute portfolio
    print("Δ min:", delta.min(), "max:", delta.max(), "mean:", delta.mean())
    price_incr = y_test[:, 1:, :] - y_test[:, :-1, :]
    gains = np.sum(delta * price_incr, axis=(1, 2))
    portfolio = V0 + gains
    

    # Payoff
    K = 100.0
    H = np.maximum(y_test[:, -1, 0] - K, 0.0)
    plt.hist(portfolio - H, bins=50)
    # Probability of successful hedge
    success_prob = np.mean(portfolio >= H)
    prob_success_list.append(success_prob)
    plt.figure(figsize=(8, 6))
    plt.hist(portfolio - H, bins=50)
    plt.title(f'Portfolio - Payoff, μ = {mu}')
    plt.xlabel("Portfolio - H")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"hist_portfolio_minus_H_mu_{mu}.png"))
    plt.close()

# ---------- Plots ----------
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
    "V0_std": V0_std_list,
    "prob_success": prob_success_list
})
df.to_csv(os.path.join(model_dir, "mu_results.csv"), index=False)

print("\n Evaluation complete. Results saved.")

# Print helpful debug stats
print("For last mu")
print(f"Mean Initial Capital (V0): {V0.mean():.4f}")
print(f"Success Probability (portfolio ≥ H): {(portfolio >= H).mean():.4f}")
print(f"Min/Max portfolio: {portfolio.min():.2f} / {portfolio.max():.2f}")
print(f"Min/Max H: {H.min():.2f} / {H.max():.2f}")

# Black-Scholes reference price for a European call option
def bs_call_price(S, K, T, sigma, r=0):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Compute BS price
S0 = 100
K = 100
sigma = 0.1
T = 30 / 250
bs_price = bs_call_price(S0, K, T, sigma)
print(f"Black-Scholes price for the call: {bs_price:.4f}")

# --- Plot Initial Capital vs Mu with BS Benchmark ---
plt.figure(figsize=(8, 6))
plt.plot(mu_values[:len(V0_list)], V0_list, marker='o', label="NN Approximation")
plt.axhline(y=bs_price, color='red', linestyle='--', label=f"BS Benchmark ({bs_price:.2f})")
plt.xlabel("Mu (penalty weight)")
plt.ylabel("Initial Capital $V_0$")
plt.title("Initial Capital vs. Mu")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "v0_vs_mu.png"))