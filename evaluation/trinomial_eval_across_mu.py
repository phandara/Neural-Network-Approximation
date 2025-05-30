import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import pandas as pd
from scipy.stats import norm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.architecture import create_two_head_model

# Parameters
mu_values = [100, 1000, 5000, 12500, 15000,16000]
model_dir = "models/Trinomial"
plot_dir = "plots/Trinomial"
os.makedirs(plot_dir, exist_ok=True)

# Load evaluation data
x_test = np.load("data/generated/Trinomial/x_test_trinomial.npy")
y_test = np.load("data/generated/Trinomial/y_test_trinomial.npy")
input_shape = x_test.shape[1:]

# Storage for results
V0_list = []
V0_std_list = []
prob_success_list = []


for mu in mu_values:
    print(f"\nEvaluating for mu = {mu} on trinomial model...")

    # Build and compile model
    model = create_two_head_model(input_shape=input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # Load pre-trained weights
    weight_path = os.path.join(model_dir, f"lstm_trinomial_mu_{mu}.weights.h5")
    if not os.path.exists(weight_path):
        print(f"❌ Weights for mu={mu} not found at {weight_path}, skipping...")
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
    price_incr = y_test[:, 1:, :] - y_test[:, :-1, :]
    gains = np.sum(delta * price_incr, axis=(1, 2))
    portfolio = V0 + gains

    # Payoff
    K = 100.0
    H = np.maximum(y_test[:, -1, 0] - K, 0.0)
    success_prob = np.mean(portfolio >= H)
    prob_success_list.append(success_prob)

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(portfolio - H, bins=50)
    plt.title(f'Portfolio - Payoff, μ = {mu}')
    plt.xlabel("Portfolio - H")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"hist_trinomial_portfolio_minus_H_mu_{mu}.png"))
    plt.close()

# ---------- Plots ----------
plt.figure(figsize=(8, 6))
plt.plot(mu_values[:len(V0_list)], V0_list, marker='o', label="NN Approximation")
plt.axhline(y=2.17, color='red', linestyle='--', label="MC Benchmark (2.17)")
plt.xlabel("Mu (penalty weight)")
plt.ylabel("Initial Capital $V_0$")
plt.title("Initial Capital vs. Mu (Trinomial)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "v0_vs_mu_trinomial.png"))

plt.figure(figsize=(8, 6))
plt.plot(mu_values[:len(prob_success_list)], prob_success_list, marker='o')
plt.xlabel("Mu (penalty weight)")
plt.ylabel("Success Probability")
plt.title("Success Probability vs. Mu (Trinomial)")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "prob_vs_mu_trinomial.png"))

plt.figure(figsize=(8, 6))
plt.plot(prob_success_list, V0_list, marker='o')
plt.xlabel("Success Probability")
plt.ylabel("Initial Capital V0")
plt.title("Pareto Frontier (Trinomial)")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "pareto_frontier_trinomial.png"))

# ---------- Save Results ----------
df = pd.DataFrame({
    "mu": mu_values[:len(V0_list)],
    "V0": V0_list,
    "V0_std": V0_std_list,
    "prob_success": prob_success_list
})
df.to_csv(os.path.join(model_dir, "mu_results_trinomial.csv"), index=False)
# ---------- Combined Histogram ----------
plt.figure(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(mu_values)))
bins = np.linspace(-2.5, 2.5, 60)  # fixed bin range for consistency

for i, mu in enumerate(mu_values):
    weight_path = os.path.join(model_dir, f"lstm_trinomial_mu_{mu}.weights.h5")
    if not os.path.exists(weight_path):
        continue

    # Recompute portfolio - H for overlay histogram
    model = create_two_head_model(input_shape=input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    model.load_weights(weight_path)
    y_pred = model.predict(x_test, verbose=0)

    V0 = y_pred[0].squeeze()
    delta = y_pred[1]
    price_incr = y_test[:, 1:, :] - y_test[:, :-1, :]
    gains = np.sum(delta * price_incr, axis=(1, 2))
    portfolio = V0 + gains
    H = np.maximum(y_test[:, -1, 0] - K, 0.0)
    diff = portfolio - H

    plt.hist(diff, bins=bins, alpha=0.4, label=f"$\mu$={mu}", color=colors[i], density=True)

plt.xlabel("Portfolio - Payoff")
plt.ylabel("Density")
plt.title("Portfolio - H for Different $\mu$ (Trinomial Model)")
plt.xlim(-2.5, 2.5)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "histogram_overlay_trinomial.png"))
plt.close()

print("\n✅ Trinomial evaluation complete. Results saved.")
