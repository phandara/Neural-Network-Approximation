import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
import matplotlib.pyplot as plt

class TrinomialDataGenerator:
    def __init__(self, num_samples: int, time_steps: int, init_price: float = 100.0,
                 u: float = 0.05, d: float = -0.05, m: float = 0):
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.init_price = init_price
        self.u = 1+u
        self.d = 1+d
        self.m = 1+m

    def simulate_trinomial_paths(self) -> np.ndarray:
        # Equal probabilities
        p_u = 1/3
        p_d = p_u
        p_m = p_d

        prices = np.zeros((self.num_samples, self.time_steps))
        prices[:, 0] = self.init_price

        for t in range(1, self.time_steps):
            rnd = np.random.rand(self.num_samples)
            up_mask = rnd < p_u
            down_mask = rnd >= (1 - p_d)
            mid_mask = ~(up_mask | down_mask)

            prices[up_mask, t] = prices[up_mask, t-1] * self.u
            prices[mid_mask, t] = prices[mid_mask, t-1] * self.m
            prices[down_mask, t] = prices[down_mask, t-1] * self.d

        return prices

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = self.simulate_trinomial_paths()
        x = np.zeros_like(y)
        x[:, 0] = y[:, 0]
        x[:, 1] = y[:, 0]
        x[:, 2:] = y[:, 1:-1]

        x = x.reshape(self.num_samples, self.time_steps, 1)
        y = y.reshape(self.num_samples, self.time_steps, 1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        return x_train.astype(np.float32), x_test.astype(np.float32), \
               y_train.astype(np.float32), y_test.astype(np.float32)

if __name__ == "__main__":
    generator = TrinomialDataGenerator(num_samples=1000, time_steps=30)
    x_train, x_test, y_train, y_test = generator.generate_data()

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Plot sample paths
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.plot(y_train[i, :, 0], label=f"Path {i+1}")
    plt.title("Sample Trinomial Price Paths from Training Data")
    plt.xlabel("Time Step")
    plt.ylabel("Asset Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
