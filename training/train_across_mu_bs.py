import os
import tensorflow as tf
import sys
import numpy as np

"""
Train and save two-head LSTM models under the BS model.

This script:
1. Generates synthetic price path data using our BS DataGenerator.
2. Trains a custom QuantileHedgeModel with different penalization factors (mu).
3. Saves the resulting model weights for each mu value.
"""

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.generator_bs import DataGenerator
from models.architecture import create_two_head_model, QuantileHedgeModel

# Parameters
num_samples = 100000*5
time_steps = 30
learning_rate = 1e-4
epochs = 70
batch_size = 256*2

# List of mu values to train over
mu_values = [10, 100, 500, 1000, 3000, 5000, 7500, 15000]

# Prepare data
print("Generating data...")
np.random.seed(1)
generator = DataGenerator(num_samples=num_samples, time_steps=time_steps)
x_train, x_test, y_train, y_test = generator.generate_data()
input_shape = (time_steps, 1)
# Save test set
os.makedirs("data/generated", exist_ok=True)
np.save("data/generated/BS/x_test.npy", x_test)
np.save("data/generated/BS/y_test.npy", y_test)

# Training loop
for mu in mu_values:
    print(f"\n=== Training model with mu = {mu} ===")

    base_model = create_two_head_model(input_shape=input_shape)
    model = QuantileHedgeModel(base_model, mu, beta = 0.75, option="European")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the model weights
    weight_path = f"models/BS/lstm_quantile_mu_{mu}.weights.h5"
    base_model.save_weights(weight_path)
    print(f"Saved weights to: {weight_path}")

print("\n All trainings complete.")
