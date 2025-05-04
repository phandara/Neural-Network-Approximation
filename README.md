# Neural-Network-Approximation

## Structure
```
quantile-hedging-nn/
│
├── data/
│   ├── generator.py            # Black-Scholes / Heston simulation
│   └── utils.py                # Scaling, batching, train/test split helpers
│
├── models/
│   ├── architecture.py         # Model architecture definitions (LSTM, FFN, etc.)
│   ├── loss_functions.py       # Truncated sigmoid-based losses
│   ├── metrics.py              # Hedge probability, pricing metrics, etc.
│   └── trainer.py              # Model compilation, training loop, callbacks
│
├── experiments/
│   ├── train_bs.py             # Training script on BS data
│   ├── train_heston.py         # Training on Heston (later)
│   └── analyze_results.py      # Post-training metrics and visualizations
│
├── plots/
│   └── ...                     # Saved plot outputs
│
├── notebooks/
│   └── dev_tests.ipynb         # Prototyping notebook (for exploratory use)
│
├── saved_models/
│   └── bs_lambda_1000.h5       # Trained model weights
│
├── config/
│   └── config.yaml             # Config file for paths, training params, model hyperparams
│
├── README.md
├── requirements.txt
└── .gitignore

```

## Checklist
| Component              | Done? | Notes                                               |
| ---------------------- | ----- | --------------------------------------------------- |
| Model generator        | ⬜     | Re-implement from scratch or re-use cleaned version |
| LSTM model             | ⬜     | Modular, test on synthetic data                     |
| Loss Function          | ⬜     | Careful TensorFlow implementation                   |
| Training loop          | ⬜     | Save models for different λ                         |
| Evaluation pipeline    | ⬜     | Plots, performance stats, CSV export                |
| Visualizations         | ⬜     | Quantile plot, probability plot, histograms         |
