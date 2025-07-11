# Neural-Network-Approximation

This repository contains a numerical framework to approximate superhedging prices of financial derivatives using LSTM-based neural networks under different market models. The framework is implemented for:
- Black–Scholes (BS) Model
- Trinomial Tree Model
- Heston Stochastic Volatility Model

It evaluates the initial capital vs. success probability tradeoff for superhedging via customized loss functions.

## Structure
```
quantile-hedging-nn/
│
├── data/
│   ├── generated/                                # consisting generated data
│   │   ├── BS/
│   │   ├── Trinomial/                          
│   │   └── Heston/                             
│   ├── generator_bs.py                           # respective data generator         
│   ├── generator_trinomial.py
│   └── generator_heston.py
│
├── training/                                     # training files
│   ├── train_across_mu_bs.py
│   ├── train_across_mu_trinomial.py
│   └── train_across_mu_heston.py
│
├── evaluation/                                   # evaluation files
│   ├── bs_eval_across_mu.py
│   ├── trinomial_eval_across_mu.py
│   ├── heston_eval_across_mu.py
│   └── heston_monte_carlo.py                     # MC price as benchmark
│
├── models/                                        
│   ├── BS/                                       # respective weights
│   ├── Trinomial/
│   ├── Heston/
│   └── architecture.py                           # NN architecture
│
├── plots/                                        # relevant plots and visualization
│   ├── BS/
│   ├── Trinomial/
│   └── Heston/
│
└── README.md

```

## Checklist
| Component              |   Done?    | Notes                                               |
| ---------------------- | ---------- | --------------------------------------------------- |
| Model generators       |     ✅     | Re-implement from scratch                           |
| LSTM model             |     ✅     | Modular, test on synthetic data                     |
| Loss Function          |     ✅     | Careful TensorFlow implementation                   |
| Training loop          |     ✅     | Save models for different mu                        |
| Evaluation pipeline    |     ✅     | Plots, performance stats, CSV export                |
| Visualizations         |     ✅     | Quantile plot, probability plot, histograms         |
