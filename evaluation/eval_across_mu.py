import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.generator import DataGenerator
from models.architecture import create_lstm_model
from models.loss_function import augmented_quantile_loss

# Parameters
mu_values = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
q_target = 0.95
epochs = 20
batch_size = 512
model_dir = "models_mu"
os.makedirs(model_dir, exist_ok=True)

# Data generation
generator = DataGenerator(num_samples=5000, time_steps=30)
x_train, x_test, y_train, y_test = generator.generate_data()
input_shape = x_train.shape[1:]

# Storage
V0_list = []
prob_success_list = []

# Loop over mu
for mu in mu_values:
    print(f"Training model for mu = {mu}")
    model = create_lstm_model(input_shape=input_shape)
    loss_fn = augmented_quantile_loss(q_target=q_target, mu=mu)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss_fn)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    # Save weights
    model_path = os.path.join(model_dir, f"model_mu_{mu}.weights.h5")
    model.save_weights(model_path)

    # Predict
    y_pred = model.predict(x_test)

    # Extract V0
    V0 = y_pred[:, 0, 0]
    V0_mean = np.mean(V0)
    V0_list.append(V0_mean)

    # Portfolio value
    delta = y_pred[:, 1:, :]
    price_incr = y_test[:, 1:, :] - y_test[:, :-1, :]
    gains = np.sum(delta * price_incr, axis=(1, 2))
    portfolio = V0 + gains

    # Payoff H
    K = 100.0
    H = np.maximum(y_test[:, -1, 0] - K, 0.0)

    success_prob = np.mean(portfolio >= H)
    prob_success_list.append(success_prob)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(mu_values, V0_list, marker='o')
plt.xlabel("Mu (penalty weight)")
plt.ylabel("Initial Capital V0")
plt.title("V0 vs Mu")
plt.grid(True)
plt.savefig("plots/v0_vs_mu.png")

plt.figure(figsize=(8, 6))
plt.plot(mu_values, prob_success_list, marker='o')
plt.xlabel("Mu (penalty weight)")
plt.ylabel("Success Probability")
plt.title("Success Probability vs Mu")
plt.grid(True)
plt.savefig("plots/prob_vs_mu.png")

plt.figure(figsize=(8, 6))
plt.plot(prob_success_list, V0_list, marker='o')
plt.xlabel("Success Probability")
plt.ylabel("Initial Capital V0")
plt.title("Pareto Frontier: Capital vs Probability")
plt.grid(True)
plt.savefig("plots/pareto_frontier.png")

import pandas as pd
df = pd.DataFrame({
    "mu": mu_values,
    "V0": V0_list,
    "prob_success": prob_success_list
})
df_path = os.path.join(model_dir, "mu_results.csv")
df.to_csv(df_path, index=False)

