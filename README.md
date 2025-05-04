# Neural-Network-Approximation

## Structure
```
quantile_hedging_nn/
├── data/
│   └── generator.py
├── models/
│   └── hedging_model.py
├── training/
│   ├── train.py
│   └── callbacks.py
├── evaluation/
│   ├── evaluate.py
│   └── plot_utils.py
├── utils/
│   └── losses.py
│   └── config.py
├── notebooks/
│   └── exploratory.ipynb
├── main.py
└── README.md
```


| Component              | Done? | Notes                                               |
| ---------------------- | ----- | --------------------------------------------------- |
| Model generator        | ⬜     | Re-implement from scratch or re-use cleaned version |
| LSTM model             | ⬜     | Modular, test on synthetic data                     |
| Truncated sigmoid loss | ⬜     | Careful TensorFlow implementation                   |
| Training loop          | ⬜     | Save models for different λ                         |
| Evaluation pipeline    | ⬜     | Plots, performance stats, CSV export                |
| Visualizations         | ⬜     | Quantile plot, probability plot, histograms         |
