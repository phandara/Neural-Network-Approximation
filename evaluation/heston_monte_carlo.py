import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

# Monte Carlo for Asian Call option under Heston Model (Euler-Maruyama)
def heston_monte_carlo(num_paths: int = 100000, time_steps: int = 30,
                       S0: float = 100.0, K: float = 100.0,
                       v0: float = 0.04, kappa: float = 2.0, theta: float = 0.05,
                       xi: float = 0.4, rho: float = -0.8, T: float = 30/250) -> Tuple[float, float]:

    dt = T / time_steps
    S = np.zeros((num_paths, time_steps + 1))
    V = np.zeros((num_paths, time_steps + 1))

    S[:, 0] = S0
    V[:, 0] = v0

    for t in range(1, time_steps + 1):
        Z1 = np.random.normal(size=num_paths)
        Z2 = np.random.normal(size=num_paths)
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        V_prev = V[:, t - 1]
        V[:, t] = np.maximum(
            V_prev + kappa * (theta - V_prev) * dt + xi * np.sqrt(np.maximum(V_prev, 0)) * np.sqrt(dt) * W2,
            0
        )

        S_prev = S[:, t - 1]
        S[:, t] = S_prev * np.exp(-0.5 * V_prev * dt + np.sqrt(V_prev * dt) * W1)

    # Compute average price and payoff for Asian call
    avg_price = np.mean(S[:, 1:], axis=1)  # exclude S[:,0]
    payoff = np.maximum(avg_price - K, 0.0)

    price_estimate = np.mean(payoff)
    std_error = np.std(payoff) / np.sqrt(num_paths)
    return price_estimate, std_error

if __name__ == "__main__":
    price, stderr = heston_monte_carlo()
    print(f"Monte Carlo estimated price (Asian Call under Heston): {price:.4f} ± {1.96*stderr:.4f}")
