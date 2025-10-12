from autora_theorist_yourtheorist import YourTheorist
import numpy as np

def test_fit_predict_consistency():
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = 2 * X + 1
    theorist = YourTheorist()
    theorist.fit(X, y)
    y_pred = theorist.predict(X)
    assert y_pred.shape == y.shape

def test_equation_extraction():
    X = np.linspace(0, 1, 10).reshape(-1, 1)
    y = 3 * X ** 2 + 2
    theorist = YourTheorist()
    theorist.fit(X, y)
    eqn = theorist.print_eqn()
    assert isinstance(eqn, str)
    assert len(eqn) > 0
