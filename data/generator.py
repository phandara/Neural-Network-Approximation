from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

class DataGenerator:
    # BS model parameters
    def __init__(self, num_samples: int, time_steps: int, init_price: float = 100.0,
                 mu: float = 0, sigma: float = 0.1, dt: float = 1.0 / 250):
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.init_price = init_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    # Simulate paths using the BS model
    def simulate_bs_paths(self) -> np.ndarray:
        Z = np.random.normal(0, 1, size=(self.num_samples, self.time_steps - 1))
        increments = (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z
        increments = np.concatenate([np.zeros((self.num_samples, 1)), increments], axis=1)
        log_paths = np.cumsum(increments, axis=1)
        paths = self.init_price * np.exp(log_paths)
        return paths

    # Generate data for training and testing
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = self.simulate_bs_paths()
        x = np.zeros_like(y)
        x[ : , 0] = y[ :, 0]
        x[ : , 1] = y[ :, 0]
        x[: ,2: ] = y[ : ,1:-1]

        x = x.reshape(self.num_samples, self.time_steps, 1)
        y = y.reshape(self.num_samples, self.time_steps, 1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        return x_train.astype(np.float32), x_test.astype(np.float32), \
               y_train.astype(np.float32), y_test.astype(np.float32)

if __name__ == "__main__":
# Run the generator to verify output shapes
    generator = DataGenerator(num_samples=1000, time_steps=30)
    x_train, x_test, y_train, y_test = generator.generate_data()

    x_train.shape, y_train.shape, x_test.shape, y_test.shape
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Plot a few sample paths from the training data
    plt.figure(figsize=(12, 6))
    for i in range(10):  # plot 10 random sample paths
        plt.plot(y_train[i, :, 0], label=f"Path {i+1}")
    plt.title("Sample BS Price Paths from Training Data")
    plt.xlabel("Time Step")
    plt.ylabel("Asset Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()