import os
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.generator import DataGenerator
from models.loss_function import augmented_quantile_loss
from models.architecture import create_lstm_model

# Parameters
num_samples = 50000
time_steps = 30
learning_rate = 1e-4
mu = 1e4  # weight for loss
q_target = 0.99  # target success probability
epochs = 40
batch_size = 128

# Load Data
generator = DataGenerator(num_samples=num_samples, time_steps=time_steps)
x_train, x_test, y_train, y_test = generator.generate_data()

# Define input shape
input_shape = (time_steps, 1)  # (timesteps, features)

# Build model
model = create_lstm_model(input_shape=input_shape, learning_rate=learning_rate)

# Compile with custom loss
loss = augmented_quantile_loss(mu=mu)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss)

# Train
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Save weights
os.makedirs("models", exist_ok=True)
model.save_weights("models/lstm_quantile_bs.weights.h5")

print("Training completed and weights saved.")
