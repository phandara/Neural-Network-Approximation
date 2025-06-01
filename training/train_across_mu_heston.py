import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.heston_architecture import create_two_head_model, QuantileHedgeModel
from data.generator_heston import HestonDataGenerator

# Parameters
num_samples = 500000*4
time_steps = 60
learning_rate = 1e-4 
epochs = 70
batch_size = 512

# List of mu values to train over
mu_values = [75000]
# Data generation
print("Generating Heston data...")
np.random.seed(1)
generator = HestonDataGenerator(num_samples=num_samples, time_steps=time_steps)
x_train, x_test, y_train, y_test = generator.generate_data()
input_shape = (time_steps, 1)

# Save test data for evaluation
os.makedirs("data/Heston/generated", exist_ok=True)
np.save("data/generated/Heston/x_test_heston.npy", x_test)
np.save("data/generated/Heston/y_test_heston.npy", y_test)

# Training loop
for mu in mu_values:
    print(f"\n=== Training model with mu = {mu} ===")

    base_model = create_two_head_model(input_shape=input_shape)
    model = QuantileHedgeModel(base_model, mu, beta = 0.75)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save model weights
    os.makedirs("models", exist_ok=True)
    base_model.save_weights(f"models/Heston/lstm_heston_mu_{mu}.weights.h5")
    print(f"Saved weights for mu = {mu}")

print("\n✅ All Heston trainings complete.")
