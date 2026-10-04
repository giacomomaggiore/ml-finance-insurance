# Project 1 — The course theory you need, question by question

"Notes 2.3" means section 2.3 of `Machine Learning in Finance and Insurance Notes` (slide sets 1 and 2 cover everything here). The notation is the course's: m observations, d features, design matrix A with a first column of ones, l = d + 1 parameters.

| Question | Theory used | Notes |
| --- | --- | --- |
| 1.b | loss → regression function; why transform the target | 1.2, 2.1 |
| 1.d | numerical vs categorical data; one-hot vs dummy; standardization | 2.7, 2.8, 2.9 |
| 2.a | training vs test error; R² and out-of-sample R² | 1.6, 1.8 |
| 2.b | normal equation; distribution of β̂ and standard errors; rank; SVD; pseudoinverse | 2.1, 2.2 |
| 3.a | overfitting; bias-variance; the dummy trap | 1.5, 1.8, 2.9 |
| 3.b | truncated pseudoinverse; ridge; LASSO; Elastic Net; cross-validation | 1.9, 2.3–2.6 |
| 3.b (intercept) | intercept from the means | 2.7 |
| 3.c | cross-validation, then an independent test set | 2.6 |
| 3.d | sparsity (L1) vs dropping singular directions | 2.3, 2.5 |
| 3.e | all of the above, plus feature engineering for the limits | 1.5, 2.10 |

---

## 1. Why transform SalePrice (1.b, 2.a)

**Square loss → conditional mean** (Notes 1.2). With the square loss, the best possible predictor is the regression function, the conditional mean:

$$\bar f(x)=\mathbb{E}[Y\mid X=x]$$

**The linear model assumes additive Gaussian noise** (Notes 2.1):

$$Y=\beta_0+\beta_1X_1+\dots+\beta_dX_d+\varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma^2)$$

House prices break this. Errors grow with the price level (a 10% error is $10k on a cheap house and $50k on an expensive one), the distribution is right-skewed, and a few expensive houses dominate the squared loss. On the log scale the errors become roughly **relative (percentage) errors** with a constant spread. That is closer to the Gaussian, same-variance noise the model assumes.

**Why it matters, in four points:**

- the OLS **standard errors, t-statistics and confidence intervals** rely on normal, homoscedastic errors (footnote 1 of the assignment)
- MSE on raw prices is **dominated by the most expensive houses**; on the log scale every house counts in relative terms
- effects become **multiplicative**: on the log scale β̂ⱼ ≈ the % change in price for one unit (here: one standard deviation) more of feature j. That is economically plausible: a garage adds more dollars to an expensive house
- after `exp`, a prediction can **never be negative**

**Careful with the back-transformation.** The model estimates the mean of log Y. Since exp is increasing, and the log-errors are about symmetric,

$$\exp\big(\widehat{\mathbb E}[\log Y\mid x]\big)\approx\text{median of }Y\text{ given }x
\qquad\text{but}\qquad
\mathbb E[Y\mid x]=\exp\big(\mu(x)+\tfrac{\sigma^2}{2}\big)\ \text{ if }\log Y\mid x\sim\mathcal N(\mu(x),\sigma^2)$$

So `np.exp(prediction)` slightly **under-predicts the mean price**, by a factor of about exp(σ²/2) ≈ 1.01 here. This is a good point for *Limitations*. A known fix is Duan's smearing factor, mean(exp(training residuals)).

## 2. Metrics (2.a, 3.a, 3.b)

**Training and test error** (Notes 1.8), with the square loss:

$$E_{tr}=\frac1m\sum_{i=1}^{m}\big(\hat f_m(X_i)-Y_i\big)^2\ \ (\text{in-sample MSE}),\qquad
E_{te}=\frac1n\sum_{i=m+1}^{m+n}\big(\hat f_m(X_i)-Y_i\big)^2\ \ (\text{out-of-sample MSE})$$

↳ training error much smaller than test error → the model has most likely **overfitted** (Notes 1.8, 1.9).

**R²** (Notes 1.6): R² = 1 − SSR/SST. In-sample it measures the fit; out of sample it measures **prediction power**. The course defines the out-of-sample SST with the **training** mean:

$$R^2_{os}=1-\frac{\sum_{\text{test}}(Y_i-\hat f_m(X_i))^2}{\sum_{\text{test}}(Y_i-\bar Y_{\text{train}})^2}$$

`sklearn.metrics.r2_score(y_test, pred)` uses the **test** mean instead. The difference is tiny here, because the training and test means are close (see `4_Plan_and_Decisions.md`). Use `r2_score` as the template suggests, and mention the course definition once.

**Two scales, two messages:**

- **log scale:** MSE ≈ mean squared **relative** error (log-MSE 0.02 → RMSE 0.14 → typical error about 14–15% of the price). It is the scale the model was trained on, and it treats cheap and expensive houses alike.
- **USD scale:** MSE in dollars², driven by the expensive houses. It answers "how many dollars am I off?" RMSE = √MSE is easier to read than MSE.

## 3. OLS with matrix algebra (2.b)

**Normal equation** (Notes 2.1). OLS minimizes ‖Ab − y‖². Every minimizer solves

$$A^\top A\,b=A^\top y$$

- **Case 1:** the columns of A are linearly independent → AᵀA is invertible → a unique solution β̂ = (AᵀA)⁻¹Aᵀy.
- **Case 2:** the columns are linearly dependent → AᵀA is **singular** (det = 0) → **infinitely many solutions**, which all give the **same fitted values** Aβ̂ (the projection of y onto the column space of A) but different coefficients.

**Sampling distribution and standard errors** (Notes 2.1). In Case 1, with Gaussian noise:

$$\hat\beta\sim\mathcal N_{d+1}\big(\beta,\ \sigma^2(A^\top A)^{-1}\big),\qquad
\widehat{\sigma}^2=\frac{1}{m-(d+1)}\sum_{i=1}^m(y_i-\hat y_i)^2,\qquad
\mathrm{SE}(\hat\beta_j)=\sqrt{\widehat\sigma^2\,[(A^\top A)^{-1}]_{jj}}$$

- the t-statistic β̂ⱼ / SE(β̂ⱼ) follows a t-distribution with m − (d + 1) degrees of freedom
- **when A has rank r < d + 1**, the unbiased variance estimate divides by **m − r**, not m − (d + 1). This is exactly what `statsmodels` does (`df_resid = nobs − rank`)
- in Case 2, a single coefficient in a dependent group is **not identified**: the data cannot tell how to split the effect between the dependent columns. Its "standard error" has no meaning

**Rank, singular values, SVD** (Notes 2.2):

$$A=U\Sigma V^\top=\sum_{i=1}^{r}\sigma_i u_i v_i^\top,\qquad \sigma_1\ge\dots\ge\sigma_r>0,\qquad r=\mathrm{rank}(A)$$

- rank = number of non-zero singular values. On a computer "zero" means "below a tolerance". `np.linalg.matrix_rank` uses σ_max · max(m, l) · machine-epsilon
- **condition number** κ(A) = σ_max / σ_min. It measures how much errors in the data (and rounding errors) can be amplified. Also κ(AᵀA) = κ(A)², so the normal equation **squares** the problem. Roughly, you lose log₁₀ κ digits of the 16 digits a computer has
- the right singular vectors vᵢ of the zero singular values span the **null space**, the directions b with Ab = 0. Their non-zero entries show **which columns are dependent**

**Pseudoinverse** (Notes 2.1 Lemma, 2.2 (v)–(vi)):

$$A^{\dagger}=V\Sigma^{\dagger}U^\top=\sum_{i=1}^{r}\sigma_i^{-1}v_iu_i^\top,\qquad \hat\beta=A^{\dagger}y$$

- β̂ = A†y is the least-squares solution with the **smallest 2-norm**. Of all solutions, it has no part in the null space
- full column rank → A† = (AᵀA)⁻¹Aᵀ → **identical** to the normal equation
- rank-deficient → the normal equation has many solutions and A† picks one. `np.linalg.solve` returns a different one, fixed by rounding errors
- `np.linalg.pinv` sets σᵢ to zero when σᵢ ≤ 1e-15 · σ_max (the default cutoff). Near-zero singular values are then **treated as exactly zero**
- the Notes 2.2 toolkit example: a singular value of 0.01 becomes a factor 100 in A†. Tiny singular values amplify noise

## 4. Overfitting and the bias-variance trade-off (3.a, 3.b, 3.e)

**Bias-variance** (Notes 1.5). A more flexible model has less bias but more variance. Too flexible → overfitting (a very small training error, a larger test error). Too rigid → underfitting.

**One-hot encoding and the dummy trap** (Notes 2.9). One-hot gives k columns for k levels. With an intercept, the k columns of every categorical variable **add up to the column of ones** → exact collinearity → Case 2 again for OLS. Dummy encoding (k − 1 columns) avoids this for OLS. The assignment asks for one-hot, so OLS on all features is **rank-deficient** and relies on a minimum-norm solution. Regularized models are unique anyway (see ridge below).

**Rare levels** are a classic variance problem. A level seen in only 1–3 training houses gets a coefficient fitted to those few prices, so OLS can fit them almost perfectly. The coefficient is mostly noise, and it hurts the test error. Shrinkage (ridge, LASSO) pulls such coefficients towards 0.

## 5. The four regularization methods (3.b, 3.d)

**Standardization first** (Notes 2.7). The penalties depend on the scale of each feature. Without standardization, a feature in square feet would be penalized differently from the same feature in square metres. Standardized features make the coefficients comparable, and the penalty fair.

**Truncated pseudoinverse** (Notes 2.3). Keep only the k largest singular directions and drop the rest:

$$A_c^{\dagger}=\sum_{i=1}^{k}\sigma_i^{-1}v_iu_i^\top,\qquad k=\#\{i:\sigma_i>c\}$$

- directions with a small σᵢ carry little signal but amplify noise by 1/σᵢ → dropping them adds some bias and removes a lot of variance
- larger c (fewer directions k) → more bias, less variance → choose k by CV
- **in sklearn:** `TruncatedSVD(n_components=k)` computes Z = X V_k (the scores of the data on the top k right singular vectors). Then `LinearRegression` fits y ≈ b₀ + Zγ. In the original features this is β = V_k γ
- two small differences from the lecture formula: (1) `TruncatedSVD` does **not centre** the data (the one-hot columns are not centred); (2) the intercept is fitted **outside** the SVD, so it is never truncated. The template asks for this set-up

**Ridge** (Notes 2.4):

$$\min_b\ \lVert Ab-y\rVert_2^2+\lambda\lVert b\rVert_2^2,\qquad
\hat\beta_\lambda=(A^\top A+\lambda I)^{-1}A^\top y=\sum_{i=1}^{r}\frac{\sigma_i}{\sigma_i^2+\lambda}v_iu_i^\top y$$

- AᵀA + λI is **always invertible** (every eigenvalue is raised by λ) → a **unique** solution even when A is rank-deficient (dummy trap, identical columns)
- shrinkage factor per direction: OLS uses 1/σᵢ, ridge uses σᵢ/(σᵢ² + λ). Large directions are almost untouched and small ones are tamed. It is a "soft" version of truncation
- coefficients shrink but are **never exactly zero**
- sklearn `Ridge(alpha=λ)` minimizes ‖y − Xw‖² + α‖w‖² (no 1/m) and does **not** penalize the intercept

**LASSO** (Notes 2.5):

$$\min_b\ \lVert Ab-y\rVert_2^2+\lambda\sum_{j=1}^{d}|b_j|\qquad(\text{the sum starts at }j=1:\ \text{the intercept is not penalized})$$

- the L1 ball is a diamond with corners on the axes → the solution often sits on a corner → **some coefficients are exactly 0** → a sparse model that also **selects features**
- no closed form; solved numerically (sklearn: coordinate descent → set `max_iter` large enough)
- sklearn `Lasso` minimizes (1/(2m))‖y − Xw‖² + α‖w‖₁. The **1/(2m)** means Lasso's α and Ridge's α are on **different scales**. Never compare their sizes directly

**Elastic Net** (Notes 1.9: "λ₁‖f‖₁ + λ₂‖f‖₂² = ElasticNet"). sklearn uses a strength α and a mixing parameter ρ = `l1_ratio`:

$$\frac{1}{2m}\lVert y-Xw\rVert_2^2+\alpha\rho\lVert w\rVert_1+\frac{\alpha(1-\rho)}{2}\lVert w\rVert_2^2
\ \ \Longleftrightarrow\ \
\frac{1}{m}\lVert y-Xw\rVert_2^2+\underbrace{2\alpha\rho}_{\lambda_1}\lVert w\rVert_1+\underbrace{\alpha(1-\rho)}_{\lambda_2}\lVert w\rVert_2^2$$

- ρ = 1 → LASSO; ρ → 0 → ridge-like
- with **correlated features** (GarageCars/GarageArea, the one-hot blocks), LASSO tends to keep one feature of a group at random. The L2 part makes Elastic Net keep or drop **correlated features together** (the "grouping effect")

**Sparsity vs truncation (3.d).** LASSO/EN set **individual feature coefficients** to 0. Which ones depends on y. The result is a model that uses fewer of the original features. TPI drops **directions in the feature space** (the columns of V, which mix all features). These are chosen from X alone, without looking at y. After truncation, β = V_k γ is usually **dense**: every original feature keeps a non-zero coefficient. So TPI reduces the effective dimension but does **not** select features. The TA slide says it this way: "Lasso sparsifies feature coefficients; truncated SVD removes directions in the design space."

## 6. Cross-validation, refit and the test set (3.b, 3.c)

**k-fold CV** (Notes 2.6). Split the *training* data into k parts. For each part: fit on the other k − 1, compute the MSE on this part. Average over the k parts, and choose the hyperparameter with the smallest average. Then the chosen model **must still be evaluated on an independent test set**, because the validation data already made a choice.

**Leakage.** Every learned preprocessing quantity (means, standard deviations, modes, the list of categories) must be learned **inside** each training fold. Otherwise the validation fold has helped to build the model and the CV error is too optimistic. Putting the preprocessor inside the `Pipeline` guarantees this.

**`step__parameter`.** In a `Pipeline`, `model__alpha` means "the parameter `alpha` of the step named `model`". Deeper nesting works the same way, e.g. `preprocess__columns__num__...`.

**`refit=True`.** After CV has chosen the best values, `GridSearchCV` fits the **whole pipeline once more on the full training set** (all 8 folds together), with those values. That becomes `best_estimator_`, which `search.predict` uses. It is useful because the final model:

- learns from all the training data (each CV model saw only 7/8 of it)
- has its preprocessing statistics estimated on all the training data
- is **one** model, ready for **one** honest test evaluation, and the test set was never touched before

## 7. Why the intercept is not penalized (3.b)

**Intercept from the means** (Notes 2.7). The first normal equation gives

$$b_0=\bar y-\sum_{j=1}^{d}\bar x_j b_j,\qquad\text{so with centred features }b_0=\bar y .$$

- the intercept only sets the **average level** of the predictions. It is not a source of complexity or of overfitting
- penalizing it would pull it towards 0, an **arbitrary** point that depends on the units: on the log scale 0 means a price of e⁰ = $1, and the average log price is about 12
- without a penalty, shifting all y by a constant just shifts the intercept (**shift invariance**), and the residuals keep a zero mean (the model is right on average)
- sklearn's `Ridge`, `Lasso` and `ElasticNet` with `fit_intercept=True` centre the data and leave the intercept unpenalized. For TPI the intercept is estimated by `LinearRegression` after the SVD

## 8. Limits of linear models (3.e, report)

**Feature engineering** (Notes 2.10). "Linear" regression only needs to be linear **in the coefficients**. The inputs can be transformed first: log of skewed areas (LotArea skew 12), quality × size interactions, neighbourhood × quality, age at sale = YrSold − YearBuilt. None of this is asked. It belongs in *Limitations* as a possible next step, together with nonlinear models (trees, boosting — slide sets 7–9).
