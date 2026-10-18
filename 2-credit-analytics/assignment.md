ETH Zurich, Fall 2026 

Machine Learning in Finance & Insurance 

# Project 2: Credit Analytics 

(First discussion: Oct 14; Last questions: Oct 28; Deadline: Nov 4; In charge: Zhexin Wu) 

This project is about credit analytics for consumer loans. The goal is to estimate risk profiles of individuals applying for a loan. For simplicity, we work with artificially generated data and only consider three borrower characteristics: age, monthly income and employment status. In reality, the availability of good data is important, and typically, many more features are taken into account. 

**Submission and report.** Submit one fully executed Jupyter notebook in which all requested quantities are printed or shown in tables/plots. Briefly interpret your results where asked. At the end of the notebook, include a concise report in Markdown cells under the heading _Project Report_ . The report should summarize your approach, model selection (here: your comparison of the classifiers and of the lending strategies), key results, interpretation, robustness, and limitations. Detailed computations and intermediate results should remain in the main body of the notebook. 

**Code quality.** Keep your analysis readable and reproducible. Avoid unnecessary duplication and use small reusable helper functions where appropriate. Classes or object-oriented design are not required. 

**Grading.** The implementation and results account for 80% of the project grade, and the Project Report accounts for 20%. 

1. Let $m=20000$ and $n=10000$. Simulate $m+n$ independent vectors $x_i=(x_{i1},x_{i2},x_{i3})\in\mathbb{R}^3$, $i=1,\ldots,m+n$, with:

   - $x_{i1}$ = age in $[18,80]$ (continuous uniform distribution).
   - $x_{i2}$ = monthly income in CHF 1000 in $[1,15]$ (continuous uniform distribution).
   - $x_{i3}\in\{0,1\}$ indicates salaried/self-employed, where 0 = salaried and 1 = self-employed (probability of being self-employed is 10%).

Assume $x_{i1},x_{i2},x_{i3}$ are independent.

   - a) Compute the empirical means and standard deviations of $x_{i1},x_{i2},x_{i3}$ over $i=1,\ldots,m$.
   - b) Can you think of additional features (besides age, income, salaried/self-employed) that could be relevant in reality?

2. Let $\xi_i$, $i=1,\ldots,m+n$, be independent random variables uniformly distributed on $(0,1)$, and let $\psi:\mathbb{R}\to(0,1)$ be the logistic (sigmoid) function

   $$
   \psi(t)=\frac{1}{1+e^{-t}}.
   $$

Consider two functions $p_1,p_2:\mathbb{R}^3\to(0,1)$ defined by

   $$
   \begin{aligned}
   p_1(x_i) &= \psi\left(13.3-0.33x_{i1}+3.5x_{i2}-3x_{i3}\right),\\
   p_2(x_i) &= \psi\left(5-10\left[\mathbb{1}_{(-\infty,25)}(x_{i1})+\mathbb{1}_{(75,\infty)}(x_{i1})\right]+1.1x_{i2}-x_{i3}\right).
   \end{aligned}
   $$

Generate two artificial data sets $(x_i,y_i^{(1)})$ and $(x_i,y_i^{(2)})$, $i=1,\ldots,m+n$, by setting

   $$
   y_i^{(s)}=\mathbb{1}\{\xi_i\le p_s(x_i)\},\qquad s=1,2.
   $$

(We use the convention that $y_i^{(s)}=1$ denotes a good borrower and $y_i^{(s)}=0$ denotes a delinquent borrower. Thus, $p_1$ and $p_2$ are the conditional probabilities that loans will be paid back in the two data-generating regimes.)

For both data sets, $s=1,2$, do the following:

- a) Fit a logistic regression model $\hat p_s^{\mathrm{log}}:\mathbb{R}^3\to(0,1)$ on the training data $(x_i,y_i^{(s)})$, $i=1,\ldots,m$. Calculate the cross-entropy loss of $\hat p_s^{\mathrm{log}}$ on the training and test data. You can use `sklearn.linear_model.LogisticRegression`.

- b) For SVM classification, denote by $\hat\sigma_j$ the empirical standard deviation of $(x_{ij})_{i=1}^m$ and use the normalized data $\tilde x_{ij}=x_{ij}/\hat\sigma_j$ for both training and evaluation.

   - (i) Fit an SVM $\hat f_s^{\mathrm{svm}}:\mathbb{R}^3\to\mathbb{R}$ on the training data by minimizing the regularized hinge loss. With $\tilde y_i^{(s)}=2y_i^{(s)}-1$, define

      $$
      (\hat w_s,\hat b_s)\in\operatorname*{arg\,min}_{w,b}
      \left\{\lambda\lVert w\rVert^2+\frac{1}{m}\sum_{i=1}^{m}\max\left(0,1-\tilde y_i^{(s)}\left[\langle w,\Phi(\tilde x_i)\rangle+b\right]\right)\right\},
      \qquad
      \hat f_s^{\mathrm{svm}}(x)=\langle\hat w_s,\Phi(x)\rangle+\hat b_s.
      $$

      where $\Phi$ is the feature map associated with the kernel

      $$
      k(x,x')=\exp\left(-10^{-1}\lVert x-x'\rVert_2^2\right),
      $$

      and $\lambda=2.5/m$. You can use `sklearn.svm.SVC`; this choice of $\lambda$ corresponds to $C=1/(2\lambda m)=0.2$ in `sklearn.svm.SVC`.

   - (ii) On top of $\hat f_s^{\mathrm{svm}}$, fit a logistic function $\hat g_s:\mathbb{R}\to(0,1)$ of the form

      $$
      \hat g_s(t)=\frac{1}{1+\exp\left(-\hat a_s t-\hat b_s\right)}.
      $$

     Then $\hat p_s^{\mathrm{svm}}:=\hat g_s\circ\hat f_s^{\mathrm{svm}}$ predicts the conditional probability that a loan is paid back; see Platt (1999)<sup>1</sup>. You may use `probability=True` in `sklearn.svm.SVC`.

      - (iii) Compute the cross-entropy loss of $\hat p_s^{\mathrm{svm}}$, $s=1,2$, on both the normalized training and test data.

      - (iv) Would the results change if we used standardized data $\tilde z_{ij}=(x_{ij}-\hat\mu_j)/\hat\sigma_j$ instead of normalized data $\tilde x_{ij}=x_{ij}/\hat\sigma_j$, where $\hat\mu_j$ is the empirical mean of $(x_{ij})_{i=1}^m$? Explain why or why not.

   - c) Generate FDR/TPR curves (with TPR on the horizontal axis and FDR on the vertical axis) from the test data for $\hat p_s^{\mathrm{log}}$ and $\hat p_s^{\mathrm{svm}}$, and compute the AUC of each curve. Briefly explain how you construct the curves and what the AUC means in this setting. Compare the two models on both data sets.

3. Let us now focus on the second data set $(x_i,y_i^{(2)})$, $i=1,\ldots,m+n$. The goal is to find “good investment opportunities” in the test data based on the features $x_i$, $i=m+1,\ldots,m+n$. We assume that loans are either completely repaid with interest or fully delinquent. In reality, a lender tries to recover parts of delinquent loans.

We compare three different lending strategies: 

- (i) We give out a loan to every person in the test data set in the amount of CHF 1000 charging an interest rate of 5.5%. 

- (ii) We only charge an interest rate of 1%, but we selectively choose the applicants who are awarded a loan (in the amount of CHF 1000) using the selection criterion

  $$
  \hat p_2^{\mathrm{log}}(x_i)\ge 0.95.
  $$

> 1 `https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf` 

2 

- (iii) We only charge an interest rate of 1% but selectively choose the applicants who are awarded a loan (in the amount of CHF 1000) using the selection criterion

  $$
  \hat p_2^{\mathrm{svm}}(\tilde x_i)\ge 0.95,
  $$

where $\tilde x_i=(\tilde x_{i1},\tilde x_{i2},\tilde x_{i3})$ contains the normalized features from 2.b).

To estimate the performance of the strategies (i)–(iii) above, we simulate different market scenarios according to the conditional probabilities $p_2(x_{m+i})$, $i=1,\ldots,n$. Using independent random variables $\xi_{i,k}\sim\mathrm{Unif}(0,1)$, $i=1,\ldots,n$, $k=1,\ldots,50000$, generate the $n\times 50000$ matrix $D\in\{0,1\}^{n\times 50000}$ by setting

$$
D_{i,k}=\mathbb{1}\{\xi_{i,k}\le p_2(x_{m+i})\}.
$$

Here, $D_{i,k}=1$ means that in scenario $k$, the $i$th loan is paid back with interest. Thus, the $k$th column of $D$ describes which loans are paid back in scenario $k$.

Now, for each of the strategies (i)–(iii) above ... 

- a) plot a histogram of the profits & losses over the different market scenarios and estimate the expected profit & loss. 

- b) Estimate the 95% VaR of the profit-and-loss distribution, defined as the negative of its 5% quantile: $\operatorname{VaR}_{0.95}=-\operatorname{Quantile}_{0.05}(\mathrm{P\&L})$.

Report all profits & losses and VaRs in CHF. 

- c) Compare the three strategies and briefly interpret your results. 

3 

