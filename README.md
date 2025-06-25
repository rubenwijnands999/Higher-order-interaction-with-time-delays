# Volterra CPD

Tensor regression assuming the model weights follow a CP-model.

## Overview
This package provides tools for tensor kernel regression using a CP (CANDECOMP/PARAFAC) model for the weights. It is designed for higher-order interaction modeling, such as Volterra series, and includes efficient estimation routines and utilities for model evaluation.

## Installation

### From PyPI (Recommended)
```bash
pip install higher-order-ml-package
```

### From Source (Development)
For local development or latest features:

```bash
git clone https://github.com/rubenwijnands999/PhD-package.git
cd PhD-package
pip install -e .
```

**Note**: The local installation will automatically install all required dependencies (numpy, scikit-learn, scipy, matplotlib, networkx, pandas).

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

## Constrained Models and Data Preparation

### ConstrainedVolterraCPD
The package also provides constrained Volterra models that incorporate temporal dependencies and structural constraints. These models are particularly useful for time series data and multi-channel signals.

#### Data Preparation for Constrained Models

**Important**: Constrained models require specially formatted input data that includes time-lagged versions of your original features. The data must be reshaped to include `M+1` copies of each variable representing different time delays.

**Data Format Requirements:**
- Original data: `X` of shape `(T, N-1)` where `T` is time samples, `N-1` is number of input variables
- Constrained data: `X_constrained` of shape `(T-M, (N-1)*(M+1))` where `M` is the maximum time delay

**Step-by-step Data Preparation:**

```python
import numpy as np
from higher_order_ML_package.volterra_cpd import ConstrainedVolterraCPD

# You can also use the function from examples/simulations.py:
# from examples.simulations import reshape_for_constrained

def prepare_constrained_data(X, y, M):
    """
    Prepare data for constrained Volterra models with time lags.
    
    Parameters
    ----------
    X : ndarray of shape (T, N-1)
        Original input data
    y : ndarray of shape (T,)
        Target values
    M : int
        Maximum time delay
        
    Returns
    -------
    X_new : ndarray of shape (T-M, (N-1)*(M+1))
        Reshaped data with time delays
    y_new : ndarray of shape (T-M,)
        Corresponding target values
    """
    T, N_features = X.shape
    X_new = np.zeros((T, N_features * (M + 1)))
    
    # Create time-delayed versions
    for i in range(N_features):
        for j in range(M + 1):
            X_new[:, i * (M + 1) + j] = np.roll(X[:, i], j)
    
    # Remove initial samples affected by time delays
    return X_new[M:, :], y[M:]

# Example usage
T, N = 1000, 5
X_original = np.random.randn(T, N-1)  # Shape: (1000, 4)
y = np.random.randn(T)                # Shape: (1000,)

M = 2  # Use 2 time delays
X_constrained, y_constrained = prepare_constrained_data(X_original, y, M)
print(f"Original shape: {X_original.shape}")      # (1000, 4)
print(f"Constrained shape: {X_constrained.shape}") # (998, 12) = (1000-2, 4*(2+1))
```

#### Using Constrained Models

```python
# ALS-LR Algorithm (with Low-Rank constraints)
model_lr = ConstrainedVolterraCPD(
    R=3,                    # CP rank
    D=3,                    # Maximum interaction order
    M=2,                    # Time delay parameter
    algorithm="ALS-LR",     # Algorithm choice
    reg_lambda=0.001,       # Regularization
    max_iter=100,           # Maximum iterations
    max_inner_iter=2,      # Inner iterations for ALS-LR
    runs=5,                 # Multiple random initializations
    verbose=True
)

# Fit the constrained model
model_lr.fit(X_constrained, y_constrained)

# Make predictions
predictions = model_lr.predict(X_constrained[:10])

# ALS-SVD Algorithm (with SVD-based constraints)
model_svd = ConstrainedVolterraCPD(
    R=3, D=3, M=2,
    algorithm="ALS-SVD",    # SVD-based constraints
    reg_lambda=0.001,
    max_iter=100,
    runs=5,
    verbose=True
)

model_svd.fit(X_constrained, y_constrained)
predictions_svd = model_svd.predict(X_constrained[:10])
```

#### Understanding the M Parameter

The `M` parameter controls the temporal structure:
- `M=0`: No time delays (similar to basic model)
- `M=1`: Include current and 1 previous time step
- `M=2`: Include current and 2 previous time steps
- Higher M: More temporal context but larger feature space

#### Complete Example with Real Data Pipeline

```python
import numpy as np
from higher_order_ML_package.volterra_cpd import VolterraCPD, ConstrainedVolterraCPD
from higher_order_ML_package.utils import get_unique_interactions, plot_graph

# 1. Generate synthetic data with known interactions
def generate_volterra_data(N, T, interactions, SNR=10):
    """Generate synthetic Volterra series data"""
    X = np.random.randn(N-1, T)
    y = np.zeros(T)
    
    # Add true interactions
    for interaction in interactions:
        indices = np.array(interaction) - 1  # Convert to 0-based
        y += np.prod(X[indices, :], axis=0)
    
    # Add noise
    if SNR < np.inf:
        noise_power = np.var(y) / (10**(SNR/10))
        y += np.sqrt(noise_power) * np.random.randn(T)
    
    return X.T, y

# 2. Define true higher-order interactions
N = 6  # Total variables (including target)
T = 2000  # Time samples
true_interactions = [[2, 3], [4, 5, 6], [1]]  # True interaction patterns

# 3. Generate data
X, y = generate_volterra_data(N, T, true_interactions, SNR=15)

# 4. Split into train/test
split = int(0.8 * T)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Compare basic vs constrained models
print("=== Basic VolterraCPD ===")
basic_model = VolterraCPD(R=3, D=3, runs=10, verbose=True)
basic_model.fit(X_train, y_train)
basic_pred = basic_model.predict(X_test)
basic_mse = np.mean((y_test - basic_pred)**2)
print(f"Basic model MSE: {basic_mse:.6f}")

# 6. Prepare data for constrained model
M = 1  # Use 1 time delay
X_train_const, y_train_const = prepare_constrained_data(X_train, y_train, M)
X_test_const, y_test_const = prepare_constrained_data(X_test, y_test, M)

print(f"Constrained data shapes: {X_train_const.shape}, {y_train_const.shape}")

# 7. Fit constrained model
print("=== Constrained VolterraCPD (ALS-LR) ===")
const_model = ConstrainedVolterraCPD(
    R=3, D=3, M=M, 
    algorithm="ALS-LR", 
    max_inner_iter=5,
    runs=10, 
    verbose=True
)
const_model.fit(X_train_const, y_train_const)
const_pred = const_model.predict(X_test_const)
const_mse = np.mean((y_test_const - const_pred)**2)
print(f"Constrained model MSE: {const_mse:.6f}")

# 8. Extract and visualize interactions
basic_interactions = get_unique_interactions(basic_model.weights_, D=3)
const_interactions = get_unique_interactions(const_model.weights_, D=3)

print(f"Basic model found {len(basic_interactions)} interactions")
print(f"Constrained model found {len(const_interactions)} interactions")

# 9. Visualize interaction networks
plot_graph({0: basic_interactions}, N, weight_threshold=0.1)
plot_graph({0: const_interactions}, N, weight_threshold=0.1)
```

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

### ConstrainedVolterraCPD
A constrained version that incorporates temporal dependencies and structural constraints.

**Parameters:**
- `R` (int): Rank of the CP decomposition (default: 3)
- `D` (int): Order of interactions (default: 3)
- `M` (int): Maximum time delay parameter (default: 1)
- `algorithm` (str): Algorithm choice: 'ALS-LR' or 'ALS-SVD' (default: 'ALS-LR')
- `reg_lambda` (float): Regularization parameter (default: 0.001)
- `max_iter` (int): Maximum number of iterations (default: 100)
- `max_inner_iter` (int): Maximum inner iterations for ALS-LR (default: 2)
- `runs` (int): Number of random initializations (default: 1)
- `tol` (float): Convergence tolerance (default: 1e-4)
- `verbose` (bool): Verbosity flag (default: False)

**Methods:**
- `fit(X, y)`: Fit the constrained model to data.
- `predict(X_new)`: Predict using the fitted model.

### Utilities
- `get_unique_interactions(weights, D)`: Extract interaction patterns from fitted models
- `plot_graph(interactions, N, weight_threshold)`: Visualize interaction networks
- `compute_MSE_metric(estimated, ground_truth)`: Compute MSE between estimated and true interactions
- `compute_ROC_AUC_metric(estimated, ground_truth)`: Compute ROC-AUC for interaction detection
- `gen_ground_truth_interactions(HOI, D, N)`: Generate ground truth reference for evaluation

## Contributing
Contributions are welcome! Please open issues or submit pull requests. For major changes, open an issue first to discuss your ideas.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
