import os
import tensorflow as tf
from data.generator import DataGenerator
from models.loss_function import augmented_quantile_loss
from models.architecture import create_lstm_model

# Set training parameters
NUM_SAMPLES = 10000
TIME_STEPS = 30
Q_TARGET = 0.9  # Desired quantile level
MU = 10.0        # Penalty weight
BATCH_SIZE = 512
EPOCHS = 40
LEARNING_RATE = 1e-3

# 1. Load data
generator = DataGenerator(num_samples=NUM_SAMPLES, time_steps=TIME_STEPS)
x_train, x_test, y_train, y_test = generator.generate_data()

# 2. Define model, loss, and optimizer
loss_fn = augmented_quantile_loss(q_target=Q_TARGET, mu=MU)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model = create_lstm_model(
    input_shape=(TIME_STEPS, 1),
    loss_fn=loss_fn,
    optimizer=optimizer
)

# 3. Train the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# 4. Save model weights
weights_dir = os.path.abspath("weights")
os.makedirs(weights_dir, exist_ok=True)
model.save_weights(os.path.join(weights_dir, f"lstm_BS_q{int(Q_TARGET*100)}.h5"))

print("Training completed and weights saved.")
