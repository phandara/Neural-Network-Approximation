import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
import matplotlib.pyplot as plt

"""
Generate synthetic asset price paths using the trinomial model.
This class is used to create training and testing data for our model.
"""
class TrinomialDataGenerator:
    # Trinomial model parameters
    def __init__(self, num_samples: int, time_steps: int, init_price: float = 100.0,
                 u: float = 0.01, d: float = -0.01, m: float = 0):
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.init_price = init_price
        self.u = 1+u
        self.d = 1+d
        self.m = 1+m

    # Simulate paths using the trinomial model
    def simulate_trinomial_paths(self) -> np.ndarray:
        NumOfSamples = self.num_samples
        TimeSteps = self.time_steps
        InitPrice = self.init_price

        u = self.u
        d = self.d
        m = self.m

        Z = np.random.choice([d, m, u], size=(NumOfSamples, TimeSteps), p = [1/3, 1/3, 1/3])
        S = np.zeros([NumOfSamples, TimeSteps])
        S[:, 0] = InitPrice
        for t in range(1, TimeSteps):
            S[:, t] = S[:, t - 1] * Z[:, t]
        return S

    # Generate data for training and testing
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = self.simulate_trinomial_paths()
        x = np.zeros_like(y)
        x[:, 0] = y[:, 0]
        x[:, 1] = y[:, 0]
        x[:, 2:] = y[:, 1:-1]

        x = x.reshape(self.num_samples, self.time_steps, 1)
        y = y.reshape(self.num_samples, self.time_steps, 1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        return x_train.astype(np.float32), x_test.astype(np.float32), \
               y_train.astype(np.float32), y_test.astype(np.float32)

# Example usage
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
