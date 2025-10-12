
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error

# change "your_module_file" to the filename that contains the classes + chunk_theorist you pasted
from autora_theorist_yourtheorist import ChunkedPolynomialRegressorSparse, chunk_theorist

def test_instance_fit_predict_consistency_linear_global():
    """
    Uses the provided module-level instance `chunk_theorist`.
    Checks shape and that a simple global linear relation is learned with low MSE and k=0.
    """
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    y = 2 * X + 1

    # fit the pre-instantiated theorist
    chunk_theorist.fit(X, y)
    y_pred = chunk_theorist.predict(X)

    assert y_pred.shape == y.shape
    mse = mean_squared_error(y, y_pred)
    assert mse < 1e-3, f"MSE too high for linear: {mse:.6g}"
    assert chunk_theorist.best_k == 0, f"Expected k=0 for global linear data, got {chunk_theorist.best_k}"


def test_class_print_eqn_emits_human_readable_summary(capsys):
    """
    Instantiates the class directly, fits a quadratic target to exercise degree=2,
    then verifies that `print_eqn()` prints a non-empty, informative summary.
    """
    X = np.linspace(-1, 1, 160).reshape(-1, 1)
    y = 3 * (X ** 2) + 2

    model = ChunkedPolynomialRegressorSparse(
        max_chunks=4,
        max_degree=2,
        max_symbols=36,
        include_interactions=True,
        include_logs=True,
        allowed_powers=(0.5, 1.5),
        lambda_complexity=1e-3,
        random_state=0,
    )
    model.fit(X, y)

    # print_eqn prints to stdout; capture it
    model.print_eqn()
    out = capsys.readouterr().out

    assert isinstance(out, str) and len(out.strip()) > 0
    assert "Final selection" in out
    assert "degree=" in out and "k=" in out
