import numpy as np
from autora.theorist.maft import ChunkedPolynomialRegressorSparse
from sklearn.metrics import mean_squared_error


def test_chunked_regressor_instantiation():
	"""Simple smoke test: the object can be instantiated and has expected defaults."""
	model = ChunkedPolynomialRegressorSparse()
	# basic attribute existence checks
	assert hasattr(model, "max_chunks")
	assert hasattr(model, "max_symbols")
	assert model.max_chunks >= 0


def test_chunked_regressor_fit_predict_small_linear():
	"""Fit the model on a tiny linear dataset and check predict shape and reasonable MSE."""
	# create a simple linear problem y = 2*x + 1 with a tiny dataset
	rng = np.random.RandomState(1)
	X = rng.randn(30, 1)
	y = 2.0 * X[:, 0] + 1.0
	y = y.reshape(-1, 1)

	# use a global model (k=0) and degree=1 so it's fast and simple
	model = ChunkedPolynomialRegressorSparse(max_chunks=0, max_degree=1, max_symbols=6)
	model.fit(X, y)
	preds = model.predict(X)

	# shape checks
	assert preds.shape == y.shape

	# MSE should be reasonably small for this simple problem
	mse = mean_squared_error(y, preds)
	assert mse < 1.0
