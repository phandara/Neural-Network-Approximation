import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.generator_trimonial import TrinomialDataGenerator
from models.trinomial_loss_function import augmented_quantile_loss
from models.architecture import create_lstm_model
from models.metrics import prob_hedge, predicted_price

# Parameters
num_samples = 500000*2   
time_steps = 30
learning_rate = 1e-4
epochs = 40
batch_size = 512*2

# List of mu values to train over
mu_values = [100, 1000, 5000, 10000, 22500, 27500, 30000, 40000, 50000]

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
    loss_fn = augmented_quantile_loss(mu=mu)
    model = create_lstm_model(input_shape=input_shape, learning_rate=learning_rate)
    metrics_fn = [prob_hedge, predicted_price]
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss_fn, metrics=metrics_fn)

    # Train model
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=2)

    # Save weights
    os.makedirs("models", exist_ok=True)
    weight_path = f"models/Trinomial/lstm_trinomial_mu_{mu}.weights.h5"
    model.save_weights(weight_path)
    print(f"Saved weights to: {weight_path}")

print("\n✅ All trinomial trainings complete.")
