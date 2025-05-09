import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
import matplotlib.pyplot as plt

class HestonDataGenerator:
    def __init__(self, num_samples: int, time_steps: int, init_price: float = 100.0,
                 v0: float = 0.01, kappa: float = 2.0, theta: float = 0.01, # kappa == mean reversion speed, thetha == Long-term average vol
                 xi: float = 0.3, rho: float = -0.8, dt: float = 1.0 / 250): # xi == vol of vol, rho == correlation price vs vol
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.init_price = init_price
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt

    def simulate_heston_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        S = np.zeros((self.num_samples, self.time_steps))
        V = np.zeros((self.num_samples, self.time_steps))

        S[:, 0] = self.init_price
        V[:, 0] = self.v0

        for t in range(1, self.time_steps):
            Z1 = np.random.normal(0, 1, self.num_samples)
            Z2 = np.random.normal(0, 1, self.num_samples)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2

            V_prev = V[:, t - 1]
            V[:, t] = np.maximum(
                V_prev + self.kappa * (self.theta - V_prev) * self.dt + self.xi * np.sqrt(np.maximum(V_prev, 0)) * np.sqrt(self.dt) * W2,
                0
            )

            S_prev = S[:, t - 1]
            S[:, t] = S_prev * np.exp(( -0.5 * V_prev) * self.dt + np.sqrt(V_prev * self.dt) * W1)

        return S, V

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y, _ = self.simulate_heston_paths()
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
    np.random.seed(1)
    generator = HestonDataGenerator(num_samples=100, time_steps=30)
    S, V = generator.simulate_heston_paths()
    x_train, x_test, y_train, y_test = generator.generate_data()

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Create subplot for price paths and price vs volatility
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot sample price paths
    for i in range(10):
        axs[0].plot(S[i], label=f"Path {i+1}")
    axs[0].set_title("Sample Heston Price Paths from Training Data")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Asset Price")
    axs[0].grid(True)

    # Plot price vs volatility for one sample
    axs[1].plot(S[0], label='Price', color='tab:blue')
    axs[1].set_ylabel('Price', color='tab:blue')
    axs[1].tick_params(axis='y', labelcolor='tab:blue')

    ax2 = axs[1].twinx()
    ax2.plot(np.sqrt(V[0]), label='Volatility', color='tab:red', linestyle='--')
    ax2.set_ylabel('Volatility', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    axs[1].set_xlabel("Time Step")
    axs[1].set_title("Price and Volatility Path (Sample 0)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()