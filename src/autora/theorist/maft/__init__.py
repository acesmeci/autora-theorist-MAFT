"""
Example Theorist
"""
from typing import Union
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Our theorist (chunk_theorist)

# -------------------------------
# Feature library with allowed ops
# -------------------------------
class FeatureLibrary:
    """
    Allowed ops:
      - linear: x_i
      - degree<=2 polys: x_i^2, (optional) x_i*x_j
      - logs: ln(x_i + shift_i)
      - powers: (x_i + shift_i)^c for c in allowed_powers
    Safe for non-positive inputs via learned shifts.
    """
    def __init__(
        self,
        include_interactions=True,
        include_logs=True,
        allowed_powers=(0.5, 1.5),
        eps=1e-8,
        random_state=0,
    ):
        self.include_interactions = include_interactions
        self.include_logs = include_logs
        self.allowed_powers = tuple(float(c) for c in allowed_powers)
        self.eps = float(eps)
        self.random_state = random_state
        self.log_shifts_ = None
        self.pow_shifts_ = None
        self.n_features_in_ = None

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X):
        X = self._ensure_2d(X)
        self.n_features_in_ = X.shape[1]
        mins = np.nanmin(X, axis=0)
        self.log_shifts_ = np.maximum(self.eps - mins, 0.0)
        self.pow_shifts_ = np.maximum(self.eps - mins, 0.0)
        return self

    def transform(self, X, degree=2):
        assert self.n_features_in_ is not None, "Call fit() first."
        X = self._ensure_2d(X)
        n, d = X.shape
        assert d == self.n_features_in_, "Feature dimension mismatch."

        feats, names = [], []

        # linear
        for i in range(d):
            feats.append(X[:, i:i+1])
            names.append(f"x{i}")

        # degree-2
        if degree >= 2:
            for i in range(d):
                feats.append(X[:, i:i+1] ** 2)
                names.append(f"x{i}^2")
            if self.include_interactions and d > 1:
                for i, j in combinations(range(d), 2):
                    feats.append((X[:, i:i+1] * X[:, j:j+1]))
                    names.append(f"x{i}*x{j}")

        # logs
        if self.include_logs:
            for i in range(d):
                shifted = np.maximum(X[:, i] + self.log_shifts_[i], self.eps)
                feats.append(np.log(shifted).reshape(-1, 1))
                s = self.log_shifts_[i]
                names.append(f"ln(x{i}+{s:.3g})" if s > 0 else f"ln(x{i})")

        # small power set
        for c in self.allowed_powers:
            if abs(c - 1.0) < 1e-12:
                continue
            for i in range(d):
                shifted = np.maximum(X[:, i] + self.pow_shifts_[i], self.eps)
                feats.append((shifted ** c).reshape(-1, 1))
                s = self.pow_shifts_[i]
                names.append(f"(x{i}+{s:.3g})^{c:.3g}" if s > 0 else f"x{i}^{c:.3g}")

        F = np.hstack(feats) if feats else np.empty((n, 0))
        return F, names


# -----------------------------------------------
# Sparse linear model (OMP) + OLS refit on support
# -----------------------------------------------
class SparseLinearBlock:
    """
    Select up to term_cap features via OMP (on standardized features),
    then refit OLS on the selected features (original scale).
    Handles multi-output by unioning supports and pruning to term_cap.
    """
    def __init__(self, term_cap=6, random_state=0):
        self.term_cap = int(term_cap)
        self.random_state = random_state
        self.active_idx_ = None
        self.lr_ = None
        self.feature_names_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, F, y, feature_names):
        F = np.asarray(F); y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # standardize F (mean-0, unit-var), center y (mean-0)
        scalerF = StandardScaler(with_mean=True, with_std=True)
        Fy = scalerF.fit_transform(F)
        y_mean = y.mean(axis=0, keepdims=True)
        y_c = y - y_mean

        # OMP per target -> union support
        max_atoms = max(1, min(self.term_cap, Fy.shape[1], max(1, Fy.shape[0] - 1)))
        supports = []
        coef_store = []  # to score features if we need to prune
        for t in range(y_c.shape[1]):
            omp = OrthogonalMatchingPursuit(
                n_nonzero_coefs=max_atoms,
                fit_intercept=False,  # we centered y and standardized X
            )
            omp.fit(Fy, y_c[:, t])
            coef = getattr(omp, "coef_", None)
            if coef is None:
                continue
            coef_store.append(np.abs(coef))
            supports.append(set(np.flatnonzero(np.abs(coef) > 0).tolist()))

        if supports:
            active = sorted(list(set.union(*supports)))
        else:
            active = []

        # If nothing selected, pick best by correlation
        if len(active) == 0:
            corr = np.abs(Fy.T @ y_c).sum(axis=1)  # aggregate over targets
            best = int(np.argmax(corr))
            active = [best]

        # Prune union to term_cap if needed (by aggregate correlation with y)
        if len(active) > self.term_cap:
            corr = np.abs(Fy.T @ y_c).sum(axis=1)
            order = np.argsort(-corr[active])  # descending
            active = [active[i] for i in order[: self.term_cap]]

        self.active_idx_ = np.array(active, dtype=int)

        # OLS refit on selected features (original scale)
        F_sel = F[:, self.active_idx_]
        lr = LinearRegression()
        lr.fit(F_sel, y)
        self.lr_ = lr
        self.feature_names_ = [feature_names[j] for j in self.active_idx_]

        coef = lr.coef_
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        self.coef_ = coef
        self.intercept_ = lr.intercept_.ravel()
        return self

    def predict(self, F):
        F = np.asarray(F)
        F_sel = F[:, self.active_idx_]
        return self.lr_.predict(F_sel)

    def n_terms(self):
        return 0 if self.active_idx_ is None else len(self.active_idx_)


# --------------------------------------------------------
# Chunked sparse polynomial regressor with symbol budgeting
# --------------------------------------------------------
class ChunkedPolynomialRegressorSparse:
    """
    Piecewise sparse regression with global symbol budget.
    Search: degree ∈ {1,2}, k ∈ {0..max_chunks}.
    """
    def __init__(
        self,
        max_chunks=4,
        max_degree=2,
        max_symbols=40,
        include_interactions=True,
        include_logs=True,
        allowed_powers=(0.5, 1.5),
        lambda_complexity=1e-4,
        random_state=0,
    ):
        self.max_chunks = int(max_chunks)
        self.max_degree = int(max_degree)
        self.max_symbols = int(max_symbols)
        self.lambda_complexity = float(lambda_complexity)
        self.random_state = random_state

        self.library = FeatureLibrary(
            include_interactions=include_interactions,
            include_logs=include_logs,
            allowed_powers=allowed_powers,
            random_state=random_state,
        )

        self.best_degree = None
        self.best_k = None
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_scaler_ = None  # <-- store scaler used for clustering
        self.blocks = []
        self.feature_names_ = None
        self.mse_per_config = {}

        self.X_fit = None
        self.y_fit = None

    def _symbol_count(self, blocks):
        # 1 intercept per chunk + #selected terms
        return sum(1 + b.n_terms() for b in blocks)

    def _fit_blocks_for_k(self, X, y, degree, k):
        n = X.shape[0]

        if k == 0:
            labels = np.zeros(n, dtype=int)
            clusterer = None
            self.cluster_scaler_ = None
            per_chunk_cap = max(1, self.max_symbols - 1)
        else:
            # scaler & kmeans on training data; store scaler
            self.cluster_scaler_ = StandardScaler(with_mean=True, with_std=True).fit(X)
            Xs = self.cluster_scaler_.transform(X)
            clusterer = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = clusterer.fit_predict(Xs)
            # conservative per-chunk cap so total <= max_symbols
            per_chunk_cap = max(1, (self.max_symbols - k) // k)

        blocks = []
        preds = np.zeros_like(y, dtype=float)
        n_chunks = 1 if k == 0 else k

        for cid in range(n_chunks):
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                return None, None, np.inf, np.inf

            F, names = self.library.transform(X[idx], degree=degree)
            if F.shape[1] == 0:
                return None, None, np.inf, np.inf

            # Cap term_cap to available features/samples (extra safety for OMP)
            safe_cap = max(1, min(per_chunk_cap, F.shape[1], max(1, F.shape[0] - 1)))
            block = SparseLinearBlock(term_cap=safe_cap, random_state=self.random_state)
            block.fit(F, y[idx], names)

            preds[idx] = block.predict(F)
            blocks.append(block)

        # budget check
        scount = self._symbol_count(blocks)
        if scount > self.max_symbols:
            return None, None, np.inf, np.inf

        mse = mean_squared_error(y, preds)
        return blocks, (clusterer, labels), mse, scount

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        if y.ndim == 1: y = y.reshape(-1, 1)

        self.X_fit = X
        self.y_fit = y
        self.library.fit(X)

        best = None
        best_score = np.inf
        self.mse_per_config = {}

        max_k = min(self.max_chunks, X.shape[0])

        for degree in range(1, self.max_degree + 1):
            for k in range(0, max_k + 1):
                blocks, pack, mse, scount = self._fit_blocks_for_k(X, y, degree, k)
                self.mse_per_config[(degree, k)] = (mse, scount)
                if not np.isfinite(mse):
                    continue
                score = mse + self.lambda_complexity * scount
                if score < best_score - 1e-12:
                    best_score = score
                    best = (degree, k, blocks, pack)

        if best is None:
            raise RuntimeError("❌ No feasible (degree, k) under the symbol budget.")

        self.best_degree, self.best_k, self.blocks, pack = best
        if self.best_k == 0:
            self.clusterer, self.cluster_labels = None, np.zeros(X.shape[0], dtype=int)
            self.cluster_scaler_ = None
        else:
            self.clusterer, self.cluster_labels = pack

        # cache names for printing (consistent for a given degree)
        _, names_tmp = self.library.transform(X, degree=self.best_degree)
        self.feature_names_ = names_tmp
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1: X = X.reshape(-1, 1)

        if self.best_k == 0:
            F, _ = self.library.transform(X, degree=self.best_degree)
            return self.blocks[0].predict(F)

        # use the SAME scaler used at fit-time for clustering
        Xs_new = self.cluster_scaler_.transform(X)
        labels = self.clusterer.predict(Xs_new)

        yhat = np.zeros((X.shape[0], self.y_fit.shape[1]))
        for cid, block in enumerate(self.blocks):
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                continue
            F, _ = self.library.transform(X[idx], degree=self.best_degree)
            yhat[idx] = block.predict(F)
        return yhat

    # Print the clusters
    def print_clusters(self, feature_names=None, quantiles=(0, 50, 100), digits=5):
        """
        Describe KMeans clusters in original input units.
        - feature_names: list like ['S1','S2'] (defaults to ['x0','x1',...])
        - quantiles: which percentiles to report per feature (e.g., (5,50,95))
        """
        if self.best_k == 0 or self.clusterer is None:
            print("No clustering: global model (k=0).")
            return

        X = self.X_fit
        n, d = X.shape
        names = feature_names or [f"x{i}" for i in range(d)]

        # Labels for training data using the stored scaler
        labels = self.clusterer.predict(self.cluster_scaler_.transform(X))

        # Centroids back in original units
        centers_std = self.clusterer.cluster_centers_
        centers_orig = self.cluster_scaler_.inverse_transform(centers_std)

        print(f"\n🔎 Cluster summary (k={self.best_k}):")
        for cid in range(self.best_k):
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                print(f"\n  • Region {cid+1}: [empty]")
                continue

            Xc = X[idx]
            c = centers_orig[cid]
            print(f"\n  • Region {cid+1}: n={len(idx)}")
            print("    center: " + ", ".join(f"{names[j]}={c[j]:.{digits}g}" for j in range(d)))

            # Per-feature quantile ranges in original units
            for j in range(d):
                qs = np.percentile(Xc[:, j], q=list(quantiles))
                qstr = " | ".join([f"q{int(q)}={val:.{digits}g}" for q, val in zip(quantiles, qs)])
                print(f"    {names[j]}: {qstr}")

            # Mean radius in standardized space (how tight the cluster is)
            dists = np.linalg.norm(self.cluster_scaler_.transform(Xc) - centers_std[cid], axis=1)
            print(f"    mean radius (std units): {np.mean(dists):.{digits}g} ± {np.std(dists):.{digits}g}")

        # For 1D, also print approximate cutpoints (midpoints between sorted centroids)
        if d == 1:
            order = np.argsort(centers_orig[:, 0])
            cuts = [(centers_orig[order[i], 0] + centers_orig[order[i+1], 0]) / 2
                    for i in range(len(order) - 1)]
            if cuts:
                print("\n  approx cutpoints (1D): " + ", ".join(f"{v:.{digits}g}" for v in cuts))

    # ------------- pretty printing -------------
    def _equation_str(self, block, out_idx=0):
        coef = block.coef_[out_idx]
        if coef.size == 0:
            return f"y{out_idx+1} = {block.intercept_[out_idx]:.6g}"
        terms = [f"({c:.6g})*{name}" for c, name in zip(coef, block.feature_names_)]
        return f"y{out_idx+1} = {block.intercept_[out_idx]:.6g} + " + " + ".join(terms)

    def print_eqn(self, max_outputs=1):
        print("\n📊 Scores for (degree, chunks):")
        for (deg, k), (mse, scount) in sorted(self.mse_per_config.items()):
            mark = " (global)" if k == 0 else ""
            print(f"  - degree={deg}, k={k}{mark}: MSE={mse:.6g}, symbols={scount}")

        print(f"\n✅ Final selection: degree={self.best_degree}, chunk count={self.best_k}")
        total_symbols = self._symbol_count(self.blocks)
        print(f"   Total symbols: {total_symbols} (budget ≤ {self.max_symbols})")

        # brief cluster summary so the regions are interpretable
        if self.best_k > 0:
            # pass raw feature names if you have them, e.g. ['S1','S2']
            self.print_clusters(feature_names=None, quantiles=(5, 50, 95))

        if self.best_k == 0:
            preds = self.predict(self.X_fit)
            mse = mean_squared_error(self.y_fit, preds)
            print(f"\n🔹 Global Model: n={len(self.X_fit)}, MSE={mse:.6g}")
            for j in range(min(max_outputs, self.y_fit.shape[1])):
                print(self._equation_str(self.blocks[0], out_idx=j))
            return

        # piecewise: recompute training labels with stored scaler
        labels = self.clusterer.predict(self.cluster_scaler_.transform(self.X_fit))
        for cid, block in enumerate(self.blocks):
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                print(f"\n🔹 Region {cid+1}: [Empty]")
                continue
            preds = self.predict(self.X_fit[idx])
            mse = mean_squared_error(self.y_fit[idx], preds)
            print(f"\n🔹 Region {cid+1}: n={len(idx)}, MSE={mse:.6g}")
            for j in range(min(max_outputs, self.y_fit.shape[1])):
                print(self._equation_str(block, out_idx=j))


# -----------------------------
# Instantiate your new theorist
# -----------------------------
chunk_theorist = ChunkedPolynomialRegressorSparse(
    max_chunks=4,
    max_degree=2,
    max_symbols=36,
    include_interactions=True,
    include_logs=True,
    allowed_powers=(0.5, 1.5),
    lambda_complexity=1e-3,
    random_state=0,
)

