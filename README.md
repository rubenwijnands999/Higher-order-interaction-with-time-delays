# Volterra CPD

Tensor regression assuming the model weights follow a CP-model.

## Overview
This package provides tools for tensor kernel regression using a CP (CANDECOMP/PARAFAC) model for the weights. It is designed for higher-order interaction modeling, such as Volterra series, and includes efficient estimation routines and utilities for model evaluation.

## Installation
Install the package using pip:

```bash
pip install volterra-cpd
```

Or, for local development:

```bash
git clone https://github.com/yourusername/volterra-cpd.git
cd volterra-cpd
pip install -e .
```

## Usage Example
Here is a minimal example using the VolterraCPD model:

```python
import numpy as np
from higher_order_ML_package.volterra_cpd import VolterraCPD

def generate_synthetic_data(N, T):
    X = np.random.randn(N, T)
    y = np.sum(X, axis=0)  # Simple linear target for demo
    return X.T, y

X, y = generate_synthetic_data(N=5, T=100)
model = VolterraCPD(R=3, D=3, reg_lambda=0.001, max_iter=50, runs=1, tol=1e-4, verbose=True)
model.fit(X, y)
predictions = model.predict(X)
print(predictions)
```

For a more advanced example, see `examples/demo_volterra.py`.

## API Reference

### VolterraCPD
A scikit-learn compatible estimator for tensor regression with CPD weights.

**Parameters:**
- `R` (int): Rank of the CP decomposition (default: 3)
- `D` (int): Order of interactions (default: 3)
- `reg_lambda` (float): Regularization parameter (default: 0.001)
- `max_iter` (int): Maximum number of ALS iterations (default: 100)
- `runs` (int): Number of random initializations (default: 1)
- `tol` (float): Convergence tolerance (default: 1e-4)
- `verbose` (bool): Verbosity flag (default: False)

**Methods:**
- `fit(X, y)`: Fit the model to data.
- `predict(X_new)`: Predict using the fitted model.

### Utilities
- `compute_metric`, `compute_MSE_metric`, `compute_ROC_AUC_metric`: Functions for model evaluation.

## Contributing
Contributions are welcome! Please open issues or submit pull requests. For major changes, open an issue first to discuss your ideas.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
