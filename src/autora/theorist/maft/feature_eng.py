import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model    import Lasso
from sklearn.metrics         import mean_squared_error

class FeatureEngTheorist:
    """
    A single unified solver that picks a custom feature set
    depending on the number of inputs:
    
    1) 1‐D (Stevens): polynomial + log1p(x) + exp(x)
    2) 2‐D (Weber):   ln(1 + ΔS/S0) + polynomial on ΔS, S0
    3) 4‐D (EV):      ΔEV = V_A*P_A − V_B*P_B plus sigmoid(ΔEV)
    """

    def __init__(self, degree=3, alpha=1e-3, tol=1e-6):
        self.degree = degree
        self.alpha  = alpha
        self.tol    = tol

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def fit(self, X, y):
        # support DataFrame or array
        if isinstance(X, pd.DataFrame):
            X_arr       = X.values
            self.names  = list(X.columns)
        else:
            X_arr       = np.asarray(X)
            self.names  = [f"x{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y).ravel()
        n, d  = X_arr.shape

        # Reset
        self.feature_names_ = []
        Φ_parts = []

        if d == 1:
            # Stevens‐style: x>0 so log1p is safe
            x = X_arr[:,0].reshape(-1,1)
            # 1) degree‐d polynomial on x
            poly = PolynomialFeatures(degree=self.degree, include_bias=True)
            Φp   = poly.fit_transform(x)
            self.feature_names_ += poly.get_feature_names_out(self.names).tolist()
            Φ_parts.append(Φp)
            # 2) log1p(x)
            Φl  = np.log1p(x)
            self.feature_names_.append(f"ln(1+{self.names[0]})")
            Φ_parts.append(Φl)
            # 3) exp(x)
            Φe  = np.exp(x)
            self.feature_names_.append(f"exp({self.names[0]})")
            Φ_parts.append(Φe)

        elif d == 2:
            # Weber‐style: ΔS = x0−x1, S0 = x1
            ΔS = (X_arr[:,0] - X_arr[:,1]).reshape(-1,1)
            S0 =  X_arr[:,1].reshape(-1,1)
            # 1) log ratio
            Φw = np.log1p(ΔS/S0)
            self.feature_names_.append(f"ln(1+({self.names[0]}−{self.names[1]})/{self.names[1]})")
            Φ_parts.append(Φw)
            # 2) polynomial on (ΔS, S0)
            poly = PolynomialFeatures(degree=self.degree, include_bias=True)
            Φp   = poly.fit_transform(np.hstack([ΔS, S0]))
            names = [f"{t.replace(' ', '').replace('x0', self.names[0]).replace('x1', self.names[1])}"
                     for t in poly.get_feature_names_out()]
            self.feature_names_ += names
            Φ_parts.append(Φp)

        elif d == 4:
            # EV‐style: ΔEV = V_A*P_A − V_B*P_B
            VA,PA,VB,PB = X_arr.T
            ΔEV = (VA*PA - VB*PB).reshape(-1,1)
            # 1) sigmoid(ΔEV)
            Φs = self._sigmoid(ΔEV)
            self.feature_names_.append("sigmoid(VA*PA - VB*PB)")
            Φ_parts.append(Φs)
            # 2) polynomial on ΔEV
            poly = PolynomialFeatures(degree=self.degree, include_bias=True)
            Φp   = poly.fit_transform(ΔEV)
            names = [t if t=="1" else f"{t}".replace("x0", "ΔEV") for t in poly.get_feature_names_out()]
            self.feature_names_ += names
            Φ_parts.append(Φp)

        else:
            # Fallback: raw polynomial on all dims
            poly = PolynomialFeatures(degree=self.degree, include_bias=True)
            Φp   = poly.fit_transform(X_arr)
            names = poly.get_feature_names_out(self.names).tolist()
            self.feature_names_ = names
            Φ_parts = [Φp]

        # Stack and fit Lasso
        Φ_all = np.hstack(Φ_parts)
        self.model = Lasso(alpha=self.alpha, max_iter=10_000).fit(Φ_all, y_arr)
        self.intercept_ = self.model.intercept_
        self.coefs      = self.model.coef_
        return self

    def predict(self, X):
        # Rebuild features exactly as in fit()
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
        n, d = X_arr.shape

        Φ_parts = []
        if d == 1:
            x = X_arr[:,0].reshape(-1,1)
            poly = PolynomialFeatures(degree=self.degree, include_bias=True)
            Φp   = poly.fit_transform(x)
            Φl   = np.log1p(x)
            Φe   = np.exp(x)
            Φ_parts = [Φp, Φl, Φe]

        elif d == 2:
            ΔS = (X_arr[:,0] - X_arr[:,1]).reshape(-1,1)
            S0 =  X_arr[:,1].reshape(-1,1)
            poly = PolynomialFeatures(degree=self.degree, include_bias=True)
            Φw   = np.log1p(ΔS/S0)
            Φp   = poly.fit_transform(np.hstack([ΔS, S0]))
            Φ_parts = [Φw, Φp]

        elif d == 4:
            VA,PA,VB,PB = X_arr.T
            ΔEV = (VA*PA - VB*PB).reshape(-1,1)
            poly = PolynomialFeatures(degree=self.degree, include_bias=True)
            Φs   = self._sigmoid(ΔEV)
            Φp   = poly.fit_transform(ΔEV)
            Φ_parts = [Φs, Φp]

        else:
            poly  = PolynomialFeatures(degree=self.degree, include_bias=True)
            Φp    = poly.fit_transform(X_arr)
            Φ_parts = [Φp]

        Φ_all = np.hstack(Φ_parts)
        y_pred = self.model.predict(Φ_all)
        return y_pred.reshape(-1,1)

    def print_eqn(self):
        terms = []
        for coef, name in zip(self.coefs, self.feature_names_):
            if abs(coef) > self.tol:
                terms.append(f"({coef:.3g})·{name}")
        eq = " + ".join(terms)
        return f"y = {self.intercept_:.3g}" + (f" + {eq}" if eq else "")