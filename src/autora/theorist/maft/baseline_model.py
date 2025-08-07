import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class BaselineTheorist:
    def __init__(self, reg_lambda=1e-3):
        self.chosen = None
        self.params = {}
        self.λ = reg_lambda
        self.model = None

    def fit(self, X, y):
        x = np.asarray(X).ravel()
        y = np.asarray(y).ravel()
        best_score = np.inf

        # 1) Power law
        mask = x > 0
        if mask.sum() >= 2:
            A, B = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
            y_pred = np.exp(A) * x**B
            score = mean_squared_error(y, y_pred) + self.λ*2
            if score < best_score:
                best_score = score
                self.chosen = "power"
                self.params = {"a": np.exp(A), "b": B}

        # 2) Log law
        for c in [1e-3, 1e-2, 1e-1, 0.5, 1.0]:
            Z = np.log(x + c).reshape(-1,1)
            lr = LinearRegression().fit(Z, y)
            y_pred = lr.predict(Z)
            score2 = mean_squared_error(y, y_pred) + self.λ*3
            if score2 < best_score:
                best_score = score2
                self.chosen = "log"
                self.model = lr
                self.params = {"c": c}

        # 3) Linear
        lr2 = LinearRegression().fit(x.reshape(-1,1), y)
        y_pred = lr2.predict(x.reshape(-1,1))
        score3 = mean_squared_error(y, y_pred) + self.λ*2
        if score3 < best_score:
            self.chosen = "linear"
            self.model = lr2
            self.params = {}

        return self

    def predict(self, X):
        x = np.asarray(X).ravel()
        if self.chosen == "power":
            a, b = self.params["a"], self.params["b"]
            y = a * x**b
        elif self.chosen == "log":
            c = self.params["c"]
            Z = np.log(x + c).reshape(-1,1)
            y = self.model.predict(Z)
        else:
            y = self.model.predict(x.reshape(-1,1))
        return y.reshape(-1,1)

    def print_eqn(self):
        if self.chosen == "power":
            a,b = self.params["a"], self.params["b"]
            return f"y = {a:.3f}·x^{b:.3f}"
        elif self.chosen == "log":
            c = self.params["c"]
            intercept, slope = self.model.intercept_, self.model.coef_[0]
            return f"y = {intercept:.3f} + {slope:.3f}·ln(x + {c})"
        else:
            intercept, slope = self.model.intercept_, self.model.coef_[0]
            return f"y = {intercept:.3f} + {slope:.3f}·x"
