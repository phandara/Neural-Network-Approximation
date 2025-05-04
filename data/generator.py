import numpy as np
from typing import Tuple

class DataGenerator:
    def __init__(self, num_samples: int, time_steps: int, init_price: float = 100.0,
                 mu: float = 0.0, sigma: float = 0.2, dt: float = 1.0 / 30):
        """
        Simulate asset paths using the geometric Brownian motion (Black-Scholes model).

        Parameters:
            num_samples (int): Number of price paths to generate
            time_steps (int): Number of discrete time steps per path
            init_price (float): Initial price of the asset
            mu (float): Drift coefficient
            sigma (float): Volatility coefficient
            dt (float): Time increment (e.g., 1/30 for monthly steps over 1 year)
        """
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.init_price = init_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def simulate_bs_paths(self) -> np.ndarray:
        """
        Generate paths using geometric Brownian motion.

        Returns:
            paths (np.ndarray): shape (num_samples, time_steps), simulated price paths
        """
        Z = np.random.normal(0, 1, size=(self.num_samples, self.time_steps - 1))
        increments = (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z
        increments = np.concatenate([np.zeros((self.num_samples, 1)), increments], axis=1)
        log_paths = np.cumsum(increments, axis=1)
        paths = self.init_price * np.exp(log_paths)
        return paths

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare reshaped train/test sets for LSTM.

        Returns:
            x_train, x_test, y_train, y_test: np.ndarrays
        """
        from sklearn.model_selection import train_test_split

        y = self.simulate_bs_paths()
        x = np.zeros_like(y)
        x[:, 0] = y[:, 0]  # Broadcast initial price as first two entries
        x[:, 1] = y[:, 0]
        x[:, 2:] = y[:, :-2]

        x = x.reshape(self.num_samples, self.time_steps, 1)
        y = y.reshape(self.num_samples, self.time_steps, 1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        return x_train.astype(np.float32), x_test.astype(np.float32), \
               y_train.astype(np.float32), y_test.astype(np.float32)
