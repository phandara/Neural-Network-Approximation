import os
import tensorflow as tf
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.generator import DataGenerator
from models.loss_function import augmented_quantile_loss
from models.architecture import create_lstm_model

# Parameters
num_samples = 50000
time_steps = 30
learning_rate = 1e-4
q_target = 0.99  # target hedging probability
epochs = 100
batch_size = 128

# List of mu values to train over
mu_values = [10, 100, 1000, 5000, 10000, 20000]

# Prepare data once
print("Generating data...")
generator = DataGenerator(num_samples=num_samples, time_steps=time_steps)
x_train, x_test, y_train, y_test = generator.generate_data()
input_shape = (time_steps, 1)

# Training loop
for mu in mu_values:
    print(f"\n=== Training model with mu = {mu} ===")

    # Build and compile model
    model = create_lstm_model(input_shape=input_shape, learning_rate=learning_rate)
    loss_fn = augmented_quantile_loss(mu=mu, q_target=q_target)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss_fn)

    # Train model
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.2,
              verbose=2)

    # Save weights
    os.makedirs("models", exist_ok=True)
    weight_path = f"models/lstm_quantile_mu_{mu}.weights.h5"
    model.save_weights(weight_path)
    print(f"Saved weights to: {weight_path}")

print("\n✅ All trainings complete.")
