import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.generator_trimonial import TrinomialDataGenerator
from models.architecture import create_two_head_model, QuantileHedgeModel

# Parameters
num_samples = 500000*2*2
time_steps = 30
learning_rate = 1e-4
epochs = 40
batch_size = 512*2
#100, 1000, 5000, 10000, 22500, 27500, 30000, 40000,
# List of mu values to train over
mu_values = [25000]

# Prepare data
print("Generating trinomial data...")
np.random.seed(1)
generator = TrinomialDataGenerator(num_samples=num_samples, time_steps=time_steps)
x_train, x_test, y_train, y_test = generator.generate_data()
input_shape = (time_steps, 1)

# Save test set
os.makedirs("data/generated/Trinomial", exist_ok=True)
np.save("data/generated/Trinomial/x_test_trinomial.npy", x_test)
np.save("data/generated/Trinomial/y_test_trinomial.npy", y_test)

# Training loop
for mu in mu_values:
    print(f"\n=== Training model on trinomial data with mu = {mu} ===")

    # Build and compile model
    base_model = create_two_head_model(input_shape=input_shape)
    model = QuantileHedgeModel(base_model, mu, beta = 1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save weights
    os.makedirs("models", exist_ok=True)
    weight_path = f"models/Trinomial/lstm_trinomial_mu_{mu}.weights.h5"
    base_model.save_weights(weight_path)
    print(f"Saved weights to: {weight_path}")

print("\n✅ All trinomial trainings complete.")
