"""
Simple and reliable unit tests for ChunkedPolynomialRegressorSparse
"""
import numpy as np
from autora.theorist.maft import ChunkedPolynomialRegressorSparse


def test_linear_fit():
    """Test that the theorist can fit and predict a simple linear function."""
    # Create simple linear data: y = 2x + 1
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = (2 * X.ravel() + 1)
    
    # Fit the model
    theorist = ChunkedPolynomialRegressorSparse(
        max_chunks=0,  # Global model only for simplicity
        max_degree=1,
        max_symbols=10
    )
    theorist.fit(X, y)
    
    # Make predictions
    y_pred = theorist.predict(X)
    
    # Check that predictions are reasonable (MSE < 0.1)
    mse = np.mean((y_pred.ravel() - y) ** 2)
    assert mse < 0.1, f"MSE too high: {mse}"


def test_quadratic_fit():
    """Test that the theorist can fit a quadratic function."""
    # Create quadratic data: y = x^2 + 1
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = (X.ravel() ** 2 + 1)
    
    # Fit the model
    theorist = ChunkedPolynomialRegressorSparse(
        max_chunks=0,  # Global model only
        max_degree=2,
        max_symbols=10
    )
    theorist.fit(X, y)
    
    # Make predictions
    y_pred = theorist.predict(X)
    
    # Check that predictions are reasonable (MSE < 0.1)
    mse = np.mean((y_pred.ravel() - y) ** 2)
    assert mse < 0.1, f"MSE too high: {mse}"
    
    # Check that the model has expected attributes
    assert theorist.best_degree is not None
    assert theorist.blocks is not None
    assert len(theorist.blocks) > 0
