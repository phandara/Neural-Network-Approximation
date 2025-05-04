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
