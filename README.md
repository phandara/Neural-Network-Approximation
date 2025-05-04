# Neural-Network-Approximation

## Structure
quantile_hedging_nn/
├── data/
│   └── generator.py               # Market simulation (e.g. trinomial model)
├── models/
│   └── hedging_model.py          # Neural network architectures
├── training/
│   ├── train.py                  # Training script
│   └── callbacks.py              # TensorBoard, early stopping, model checkpointing
├── evaluation/
│   ├── evaluate.py               # Backtest & performance metrics
│   └── plot_utils.py             # Custom plotting functions
├── utils/
│   └── losses.py                 # Custom loss (truncated sigmoid) and metrics
│   └── config.py                 # Central config with hyperparameters
├── notebooks/
│   └── exploratory.ipynb         # For prototyping
├── main.py                       # Entrypoint for full training/testing cycle
└── README.md

| Component              | Done? | Notes                                               |
| ---------------------- | ----- | --------------------------------------------------- |
| Trinomial generator    | ⬜     | Re-implement from scratch or re-use cleaned version |
| LSTM model             | ⬜     | Modular, test on synthetic data                     |
| Truncated sigmoid loss | ⬜     | Careful TensorFlow implementation                   |
| Training loop          | ⬜     | Save models for different λ                         |
| Evaluation pipeline    | ⬜     | Plots, performance stats, CSV export                |
| Visualizations         | ⬜     | Quantile plot, probability plot, histograms         |
