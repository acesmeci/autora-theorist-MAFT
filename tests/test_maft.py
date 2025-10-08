## For us an example to check

def test_sanity_check():
    """Simple sanity test to verify CI runs correctly."""
    assert 1 + 1 == 2


"""
Unit tests for the MAFT Theorist.

Test 1: Global linear recovery
- Fits a simple linear function and expects low MSE using the real
  ChunkedPolynomialRegressorSparse (forced to k=0 by max_chunks=0).

Test 2: FeatureLibrary log-safety
- Ensures transform() handles non-positive inputs via learned shifts
  (no NaNs/-inf, names reflect shifts).
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from autora.theorist.maft import ChunkedPolynomialRegressorSparse, FeatureLibrary


def test_global_linear_recovery_low_mse():
    """Theorist should recover a simple linear mapping with low MSE (global model)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(240, 2))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.05 * rng.normal(size=X.shape[0])

    model = ChunkedPolynomialRegressorSparse(
        max_chunks=0,        # force global model (k=0)
        max_degree=2,
        max_symbols=36,
        include_interactions=True,
        include_logs=True,
        allowed_powers=(0.5, 1.5),
        lambda_complexity=1e-3,
        random_state=0,
    )
    model.fit(X, y)
    yhat = model.predict(X)

    mse = mean_squared_error(y, yhat)
    assert mse < 0.05, f"MSE too high: {mse:.4f}"
    assert model.best_k == 0, f"Expected global (k=0), got k={model.best_k}"
    assert model.best_degree in (1, 2), f"Unexpected degree: {model.best_degree}"


def test_feature_library_log_features_safe_for_nonpositive():
    """Log features should be finite even when inputs are non-positive (shifts applied)."""
    X = np.array([
        [-3.0, 0.0],
        [-1.0, -2.0],
        [ 0.0, 1.0],
        [ 2.0, -0.5],
    ])

    lib = FeatureLibrary(include_interactions=False, include_logs=True, allowed_powers=(0.5, 1.5))
    lib.fit(X)
    F, names = lib.transform(X, degree=1)  # linear + logs + small power set

    # 1️⃣ No NaNs or infs from logs/powers on non-positive inputs
    assert np.isfinite(F).all(), "Found non-finite values in feature matrix."

    # 2️⃣ There are log features
    log_names = [n for n in names if n.startswith("ln(")]
    assert log_names, "Expected log features in names."

    # 3️⃣ At least one log feature uses a positive shift (string contains '+')
    assert any('+' in n for n in log_names), "Expected at least one shifted log feature name."

    # 4️⃣ Programmatic check: any feature with min <= 0 must have a positive learned shift
    mins = X.min(axis=0)
    need_shift = mins <= 0
    assert np.all(lib.log_shifts_[need_shift] > 0), "Expected positive log shifts for non-positive columns."

