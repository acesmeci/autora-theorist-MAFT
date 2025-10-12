from autora.theorist.maft import ChunkedPolynomialRegressorSparse
import numpy as np

def test_fit_predict_consistency():
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = (2 * X + 1).ravel()
    theorist = ChunkedPolynomialRegressorSparse()
    theorist.fit(X, y)
    y_pred = theorist.predict(X)
    assert y_pred.ravel().shape == y.shape

def test_basic_functionality():
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = (3 * X ** 2 + 2).ravel()
    theorist = ChunkedPolynomialRegressorSparse()
    theorist.fit(X, y)
    theorist.print_eqn()  # エラーが出なければOK
    assert theorist.best_degree is not None
    assert theorist.blocks is not None
