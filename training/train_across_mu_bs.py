import os
import tensorflow as tf
import sys
import numpy as np
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.generator import DataGenerator
from models.loss_function import augmented_quantile_loss
from models.log_loss_function import log_sigmoid_quantile_loss
from models.architecture import create_lstm_model
from models.metrics import prob_hedge, predicted_price

# Parameters
num_samples = 100000
time_steps = 30
learning_rate = 1e-4
epochs = 70
batch_size = 256*2

# List of mu values to train over
mu_values = [1, 10, 100, 1000, 3000, 5000, 6000, 7500]

# Prepare data once
print("Generating data...")
np.random.seed(42)
generator = DataGenerator(num_samples=num_samples, time_steps=time_steps)
x_train, x_test, y_train, y_test = generator.generate_data()
input_shape = (time_steps, 1)
# Save test set
os.makedirs("data/generated", exist_ok=True)
np.save("data/generated/x_test.npy", x_test)
np.save("data/generated/y_test.npy", y_test)

# Training loop
for mu in mu_values:
    print(f"\n=== Training model with mu = {mu} ===")

    # Build and compile model
    loss_fn = augmented_quantile_loss(mu=mu)
    model = create_lstm_model(input_shape=input_shape, learning_rate=learning_rate)
    metrics_fn = [prob_hedge, predicted_price]
    #loss_fn = log_sigmoid_quantile_loss(mu=mu)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss_fn, metrics=metrics_fn)

    # Train model
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size)

    # Save weights
    os.makedirs("models", exist_ok=True)
    weight_path = f"models/lstm_quantile_mu_{mu}.weights.h5"
    model.save_weights(weight_path)
    print(f"Saved weights to: {weight_path}")

print("\n✅ All trainings complete.")
