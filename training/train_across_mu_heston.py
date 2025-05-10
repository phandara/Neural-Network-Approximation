import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.architecture import create_lstm_model
from models.log_loss_function import augmented_quantile_loss_heston
from models.metrics import prob_hedge, predicted_price
from data.generator_heston import HestonDataGenerator

# Parameters
num_samples = 100000
time_steps = 30
learning_rate = 1e-4
epochs = 70
batch_size = 512

# List of mu values to train over
mu_values = [10, 100, 200, 500, 1000, 3000, 5000, 7500]

# Data generation
print("Generating Heston data...")
np.random.seed(1)
generator = HestonDataGenerator(num_samples=num_samples, time_steps=time_steps)
x_train, x_test, y_train, y_test = generator.generate_data()
input_shape = (time_steps, 1)

# Save test data for evaluation
os.makedirs("data/Heston/generated", exist_ok=True)
np.save("data/generated/Heston/gx_test_heston.npy", x_test)
np.save("data/generated/Heston/y_test_heston.npy", y_test)

# Training loop
for mu in mu_values:
    print(f"\n=== Training model with mu = {mu} ===")

    loss_fn = augmented_quantile_loss_heston(mu=mu)
    model = create_lstm_model(input_shape=input_shape, learning_rate=learning_rate)
    metrics_fn = [prob_hedge, predicted_price]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss_fn, metrics=metrics_fn)

    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=2)

    # Save model weights
    os.makedirs("models", exist_ok=True)
    model.save_weights(f"models/Heston/lstm_heston_mu_{mu}.weights.h5")
    print(f"Saved weights for mu = {mu}")

print("\n✅ All Heston trainings complete.")
