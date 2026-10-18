# Assignment

ETH Zurich, Fall 2026 

Machine Learning in Finance & Insurance 

# Project 1: Linear Regression and Regularization 

(First discussion: Sept 23; Last questions: Oct 7; Deadline: Oct 14; In charge: Zhexin Wu) 

This project explores linear regression and regularization techniques in the context of predicting house prices. You will use the _Housing_ dataset from Ames, Iowa (USA), which contains 1,460 observations of residential properties. The dataset includes one target variable, _SalePrice_ (the sale price of the house in U.S. dollars), and 79 explanatory variables describing different aspects of the homes. Of these predictors, 35 are numerical and 44 are categorical. 

The goal of the project is to build models that both explain and predict _SalePrice_ , while comparing the strengths and limitations of different regression approaches. You can refer to the file `data description.txt` for a detailed description of all variables. 

This dataset is widely used in predictive modeling competitions and mimics real-world challenges such as skewed target distributions, missing values, multicollinearity among predictors, and many categorical features. Addressing these issues is part of the learning objective of the project. 

**Submission and report.** Submit one fully executed Jupyter notebook. At the end of the notebook, include a concise report in Markdown cells under the heading _Project Report_ . The report should summarize your approach, model selection, key results, interpretation, robustness, and limitations. Detailed computations and intermediate results should remain in the main body of the notebook. 

**Code quality.** Keep your analysis readable and reproducible. Avoid unnecessary duplication and use small reusable helper functions where appropriate. Classes or object-oriented design are not required. **Grading.** The implementation and results account for 80% of the project grade, and the Project Report accounts for 20%. 

1. To start, you need to import the dataset into Python in the correct format. 

   - a) Import the dataset `Housing.csv` in Python as a pandas DataFrame. Separate the explanatory features _X_ ( _Housing_ ) from the target variable _y_ ( _SalePrice_ ). 

   - b) Graphically determine whether the target variable _SalePrice_ is approximately Gaussian. If not, suggest a suitable transformation to bring _SalePrice_ closer to a Gaussian distribution and apply this transformation to the dataset. Why is it important to consider such potential transformations? 

   - c) Since the regression models have to be evaluated on a different dataset than the one used for training, split the data into two subsets: ( _X, y_ ) _train_ and ( _X, y_ ) _test_ . Randomly assign 70% of the observations to the training set and the remaining 30% to the test set. Keep an unprocessed copy of these training and test splits, since you will return to the raw features in Question 3 when constructing a cross-validation pipeline. 

   - d) Replace missing values in _X_ using the training data statistics only: 

      - For numerical features, replace missing values with the mean of the column (computed from the training set). 

      - For categorical features, replace missing values with the most frequent category in the corresponding column (computed from the training set). 

1 

- Some categorical variables admit `’NA’` (or `’None’` ) as a valid category, which should be treated as an actual level and not as missing. 

Next, standardize all numerical features by applying a z-score transform using the training mean and standard deviation, and apply the same transformation (using these training set statistics) to the test data. Finally, use one-hot encoding for all categorical features (e.g., with `pd.get` ~~`d`~~ `ummies` ). After encoding, ensure that the test set has the same number of columns as the training set and the features appear in the correct order. 

**Hint:** If `pd.get dummies` is applied separately to the training and test sets, they may produce different dummy-variable columns because some categorical levels occur in only one of the two sets. The regression model must receive the same feature columns at training and prediction time. You can use 

```
Xtest=Xtest.reindex(columns=Xtrain.columns,fillvalue=0)
```

to make the encoded test columns match the training columns and their order. Dummy variables corresponding to training categories that are absent from the test set are then filled with zeros. 

2. This question focuses on building a linear regression model to predict the variable _SalePrice_ using only the 35 numerical features. 

   - a) Restrict your analysis to the numerical predictors only (i.e., exclude all categorical features from _X_ ). Fit a linear regression model on the training dataset using the `sklearn` Python package. The regression should include an intercept term. Present a table with the regression coefficients for each feature. Compare the in-sample and out-of-sample Mean Squared Error (MSE) and _R_<sup>2</sup> . 

      - If you transformed the target variable, you have to inverse-transform the predictions before computing MSE and _R_<sup>2</sup> , so that these metrics are reported on the original _SalePrice_ scale. In addition, also report the MSE and _R_<sup>2</sup> on the transformed scale, and comment on the differences between the two. What does each set of metrics tell you about model performance? 

   - b) The `sklearn` package does not provide standard errors for the estimated regression coefficients, which are essential tools to assess the statistical precision of an estimate.<sup>1</sup> Therefore, you will now use matrix algebra in Python with the `numpy` package to compute the standard errors of the estimated coefficients _β_<sup>ˆ</sup> . All computations in this part should be performed using the training set only. Let _A ∈_ R<sup>_m×_(</sup><sup>_d_+1)</sup> denote the design matrix (including a column of ones for the intercept term), and let _y ∈_ R<sup>_m_</sup> denote the observed target values. 

      - (i) Compute the estimated coefficients _β_<sup>ˆ</sup> using 



Note that _β_<sup>ˆ</sup> 0 denotes the estimate of the intercept. 

**Practical note:** In code, do not form matrix inverses explicitly; instead use a numerically stable equivalent such as `np.linalg.solve(A.T @ A, A.T @ y)` whenever the system is numerically well behaved. 

> 1Under the classical linear model with homoscedastic, normally distributed errors, the OLS estimator _β_ ˆ is exactly normal, and each t-statistic _β_<sup>ˆ</sup> _j/_ SE( _β_<sup>ˆ</sup> _j_ ) follows a _t_ -distribution with _m−_ ( _d_ +1) degrees of freedom. This enables confidence intervals and significance tests. 

2 

(ii) Compute the standard error of each coefficient _β_<sup>ˆ</sup> _j_ , _j_ = 0 _, . . . , d_ , using 



and 

SE( _β_<sup>ˆ</sup> _j_ ) = ~~√~~ _σ_ ˆ︁<sup>2</sup> _·_ <u>[(</u> _A_<sup>_⊤_</sup> _A_ )<sup>_−_1]</sup> _jj_<sup>_,_</sup> 

where [( _A_<sup>_⊤_</sup> _A_ )<sup>_−_1</sup> ] _jj_ denotes the _j_ -th diagonal element. Recall that _j_ = 0 corresponds to the intercept. 

   - (iii) Using your matrix-algebra implementation, verify that the in-sample MSE and _R_<sup>2</sup> agree with the corresponding results from Question 2.a), up to numerical rounding. 

   - (iv) Inspect the numerical structure of the design matrix _A_ . Compute its rank and singular values, and comment on whether _A_ is full column rank and whether the smallest singular values indicate potential numerical instability. Relate your findings to the OLS calculations above. 

   - (v) Replace ( _A_<sup>_⊤_</sup> _A_ )<sup>_−_1</sup> _A_<sup>_⊤_</sup> by the Moore–Penrose pseudoinverse _A_<sup>+</sup> (use `np.linalg.pinv` with its default cutoff). Do _β_<sup>ˆ</sup> , _σ_ ˆ︁<sup>2</sup> , and the standard errors change? Briefly explain when the results are identical and when they can differ, using your findings from part (iv). 

   - (vi) Confirm your results using the OLS function from the `statsmodels` package. Report the coefficient table and standard errors, and check that they match your matrix-algebra results up to numerical rounding. If they differ, relate the discrepancy to the rank and numerical-stability diagnostics above. 

3. In this question, you will implement regularization techniques and compare their performance to ordinary least squares. Use the same training and test split as before to ensure consistency across questions. In Question 3.a), use the prepared _Housing_ dataset from Question 1. For the cross-validation analysis in Question 3.b), return to the unprocessed training and test features saved in Question 1.c). 

In this project, all learned preprocessing steps used during cross-validation should be fitted within the corresponding cross-validation training fold. This convention allows you to construct the complete model-selection procedure using `Pipeline` and `GridSearchCV` . 

- a) Fit an OLS regression of the potentially transformed target variable _SalePrice_ on all explanatory variables in the prepared _Housing_ dataset. Report the in-sample and out-of-sample MSE and _R_<sup>2</sup> on the original _SalePrice_ scale. If the target was transformed, inverse-transform the predictions before computing these metrics. 

   - Compare these results with the regression using only numerical features from Question 2.a). Does including the categorical features improve out-of-sample predictive performance? How does the gap between in-sample and out-of-sample performance change? 

- b) Implement the Truncated Pseudoinverse, Ridge, Lasso, and Elastic Net regressions. Use 8- fold cross-validation on the training set to tune the hyperparameters of each method, using MSE as the selection criterion. Use the same cross-validation folds for all methods. For this part, construct the complete model-selection workflow using `sklearn` ’s 

   - `ColumnTransformer` , 

   - `Pipeline` , and 

   - `GridSearchCV` . 

A minimal example of this workflow is provided in the project template. Follow the steps below. 

3 

- (i) Start from the unprocessed training and test features saved in Question 1.c). Construct a preprocessing component that reproduces Question 1.d), including imputation, standardization, one-hot encoding, and the special treatment of `’NA’` / `’None’` when these are valid categorical levels. 

   - Place this preprocessing component inside the pipeline so that its learned quantities are fitted separately using the training observations of each cross-validation fold. Make sure that categories not observed in a particular training fold can still be handled when transforming the corresponding validation fold. 

- (ii) Construct one pipeline for each regularization method, using the same preprocessing component. For Ridge, Lasso, and Elastic Net, place the regression model directly after the preprocessor. 

   - For the Truncated Pseudoinverse model, insert `TruncatedSVD` after the preprocessor and before `LinearRegression` . 

- (iii) Specify reasonable hyperparameter grids and tune them with `GridSearchCV` . In particular, consider: 

   - the regularization strength for Ridge and Lasso; 

   - the regularization strength and mixing parameter for Elastic Net; and 

   - the number of retained singular directions for the Truncated Pseudoinverse model. 

   - Pipeline parameters follow the usual naming convention. For example, `model` ~~`a`~~ `lpha` refers to the parameter `alpha` of the pipeline step named `model` . For the Truncated Pseudoinverse model, tune the number of retained singular directions through `svd n components` . This hyperparameter specifies the number of retained singular components rather than a singular-value cutoff. 

- (iv) Inspect whether the selected hyperparameter values lie at or near the boundary of the corresponding search grids. If so, extend the relevant range and repeat the search. Briefly justify your final hyperparameter ranges. 

- (v) Use the selected and refitted models to compute in-sample and out-of-sample MSE and _R_<sup>2</sup> on the original _SalePrice_ scale. If the target was transformed, inverse-transform the predictions before computing these metrics. 

Compare the results with the OLS models from Questions 2.a) and 3.a). 

Summarize your results in a table. Include the four regularized models as well as the OLS benchmarks from Questions 2.a) and 3.a). For quantities that do not apply to an OLS benchmark, use “–”. The table should report: 

- the selected hyperparameter(s); 

- for each regularized model, the mean cross-validation MSE across the 8 validation folds for the selected hyperparameter configuration, together with its standard deviation; 

- the in-sample MSE and _R_<sup>2</sup> on the original _SalePrice_ scale; and 

- the out-of-sample MSE and _R_<sup>2</sup> on the original _SalePrice_ scale. 

If _SalePrice_ was transformed, the cross-validation MSE used for model selection is computed on the transformed target scale. 

**Hint:** For `GridSearchCV` , the cross-validation quantities for the selected hyperparameter configuration can be obtained from the selected row of `cv results` , for example using `best index` ~~,~~ `mean test` ~~`s`~~ `core` , and `std test` ~~`s`~~ `core` . If you use `neg` ~~`m`~~ `ean` ~~`s`~~ `quared` ~~`e`~~ `rror` as the scoring rule, remember to reverse the sign of `mean` ~~`t`~~ `est score` when reporting MSE. All regressions should include an intercept term. The intercept must not be penalized during regularization. Why is it important not to penalize the intercept? 

4 

- c) Explain the meaning of the pipeline parameter notation (e.g., `model alpha` ) and what `refit=True` in `GridSearchCV` does after the best hyperparameters have been selected. Why is this refitting step useful before the final evaluation on the held-out test set? 

- d) For the Lasso and Elastic Net regularization techniques, how many estimated feature coefficients are non-zero ( _β_<sup>ˆ</sup> _j̸_ = 0), excluding the intercept? Compare these results with Ridge, which generally shrinks coefficients without setting them exactly to zero. 

For the Truncated Pseudoinverse model, report the selected number of retained singular directions instead of counting non-zero regression coefficients. Explain why truncating singular directions is conceptually different from coefficient sparsity in Lasso or Elastic Net. 

- e) Based on your findings from Questions 2 and 3, which model would you recommend for predicting house prices? Justify your choice not only by comparing performance metrics, but also by discussing the nature of the problem (e.g., number of features, presence of categorical variables, potential collinearity, sparsity, nonlinearity). Explain how the strengths and limitations of the chosen method align with the problem structure. 

5 

