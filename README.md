# Neural-Network-Approximation

## Structure
```
quantile-hedging-nn/
│
├── data/
│   ├── generated/                                 # consisting generated data
│   │   ├── BS/
│   │   ├── Trinomial/                          
│   │   └── Heston/                             
│   ├── generator_bs.py                            # respective data generator         
│   ├── generator_trinomial.py
│   └── generator_heston.py
│
├── training/                                      # training files
│   ├── train_across_mu_bs.py
│   ├── train_across_mu_trinomial.py
│   └── train_across_mu_heston.py
│
├── evaluation/                                    # evaluation files
│   ├── bs_eval_across_mu.py
│   ├── trinomial_eval_across_mu.py
│   ├── heston_eval_across_mu.py
│   └── heston_monte_carlo.py
│
├── models/                                        
│   ├── BS/                                        # respective weights
│   ├── Trinomial/
│   ├── Heston/
│   ├── architecture.py                            # NN architecture
│   ├── loss_function.py                           # loss function for BS and trimonial model
│   ├── heston_loss_function.py                    # Heston model loss
│   └── metrics.py                                 # metrics file for nn model
│
├── plots/                                         # relevant plots and visualization
│   ├── BS/
│   ├── Trinomial/
│   └── Heston/
│
└── README.md

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
