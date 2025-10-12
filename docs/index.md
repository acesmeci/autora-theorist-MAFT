# **Theorist Search Method**

## **(2 pts) Search Algorithm: How does the theorist search, and how is equation goodness determined?**

**Search Algorithm:**  
The theorist implements a **Chunked Polynomial Regressor (Sparse)** approach.  
It searches for compact, interpretable equations that describe the experimental data by systematically varying both the **polynomial degree** and the **number of spatial chunks** (subregions of the input domain).

1. **Candidate generation:**  
   For each degree–chunk combination, the data is optionally partitioned into $k$ chunks ($k = 0$ indicates a global fit).  
   Within each chunk, polynomial features up to the given degree are generated using:
   $$
   \{1, x_i, x_i^2, \dots, x_i^d, x_i x_j, \ldots\}.
   $$

2. **Sparse fitting:**  
   The coefficients are fit using **Lasso regression** ($\alpha = 0.001$), encouraging sparsity and removing redundant terms.  
   This results in compact symbolic equations that retain only meaningful terms.

3. **Evaluation of goodness:**  
   Each candidate model is evaluated by **Mean Squared Error (MSE)** on a held-out validation set:
   $$
   \text{MSE} = \frac{1}{n} \sum_i (y_i - \hat{y}_i)^2.
   $$

4. **Model selection rule:**  
   - The model with the **lowest MSE** is chosen.  
   - If two models are within a small tolerance ($\Delta \text{MSE} < 10^{-6}$), the **simpler model** (lower polynomial degree, fewer chunks) is preferred.  
   - This enforces a parsimony bias consistent with Occam’s razor.

**Why this algorithm?**  
It balances **accuracy and simplicity**, automatically selecting the least complex model that still fits the data well.  
The chunked design allows local specialization where needed, while sparse Lasso fitting ensures interpretability and generalization.

<br>
<br>


## **(2 pts) Search Space: Which search space was used, and how was it constrained?**

**Search Space:**  
The theorist explores **symbolic polynomial compositions** built from a predefined set of mathematical operations:

- **Operators:** `+`, `-`, `*`, `/`, $e^x$, $\ln(x)$, $x^c$, and constants.  
- **Polynomial degree range:** typically 1–5.  
- **Chunk count:** automatically selected between global ($k=0$) and local fits.  
- **Feature interactions:**  
  - When the number of input features ≤ 8 and degree = 1, all **pairwise interaction terms** (e.g., $x_i x_j$) are added automatically to capture multiplicative effects such as expected value terms ($p_i x_i$).  
  - For higher-dimensional problems, only standard polynomial expansion is used.

**Constraints:**  
- **Complexity constraint:**  
  Lower-degree and fewer-chunk models are always preferred under near-equal performance. Our code checks for that. 
- **Regularization constraint:**  
  Lasso regression enforces sparsity, effectively pruning irrelevant terms.  
- **Numerical constraint:**  
  Input data are normalized, and logarithmic/power transforms include stability shifts ($\varepsilon > 0$) to handle non-positive values safely.

**Why this search space?**  
It’s expressive enough to model diverse behavioral laws (e.g., power, logarithmic, or interaction effects) while remaining interpretable and computationally tractable.  
By constraining complexity and applying sparsity, the theorist efficiently searches a meaningful symbolic space without overfitting.

<br>
<br>


