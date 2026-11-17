# Assignment

Transcribed from `slides/Pr3_26.pdf` (Project 3: Deep Hedging). First discussion: Nov 4; last questions: Nov 18; deadline: Nov 25. In charge: Arishi Orra.

The goal of this project is to implement the deep hedging method of Buehler et al. (2019) from scratch and test it on data simulated from a Black–Scholes and a Heston model. Submit your solution using the template `template3.ipynb`. One submission per team is enough.

You can implement your model either in PyTorch (by adapting the code in `demo.ipynb`) or TensorFlow. In either case your model must be implemented from scratch: you are not allowed to use third-party repositories with ready-made implementations of deep hedging.

## 1. Black–Scholes model

$$dS_t = rS_t\,dt + \sigma S_t\,dW_t, \quad S_0 = s_0 \in \mathbb{R}_+, \tag{1}$$

where $W$ is a Brownian motion under a risk-neutral measure $\mathbb{Q}$, $\sigma$ the volatility and $r = 0$ the risk-free rate.

For an option with payoff $g(S_T)$, a hedging strategy solves
$\inf_{H \in \mathcal{H}} \mathbb{E}\big[(p + \int_0^T H_u\,dS_u - g(S_T))^2\big]$,
where $p = \mathbb{E}[g(S_T)]$ is the fair price and $\mathcal{H}$ the set of predictable processes. On a uniform grid $0 = t_0 < \dots < t_N = T$, train $N$ networks $H_0, \dots, H_{N-1}$ jointly by minimizing

$$\frac{1}{m}\sum_{i=1}^m \Big(p + \sum_{j=0}^{N-1} H_j \cdot \big(s^{(i)}_{t_{j+1}} - s^{(i)}_{t_j}\big) - g\big(s^{(i)}_T\big)\Big)^2 \tag{2}$$

on a training set of $m$ simulated paths of $S$.

- **a)** Use Itô's formula to check that $S_t = s_0 \exp(\sigma W_t - \frac12\sigma^2 t)$ solves SDE (1) (geometric Brownian motion).
- **b)** Simulate a training set of 100,000 paths and a test set of 10,000 paths with $N = 30$, $s_0 = 1$, $T = 1/4$, $\sigma = 0.35$, using the exact update
  $S_{t_{j+1}} = S_{t_j}\exp\big(-\frac{\sigma^2}{2}\frac{T}{N} + \sigma\sqrt{T/N}\,Z_{j+1}\big)$ with iid standard normal $Z_j$.
- **c)** Implement the model, with each $H_j: \mathbb{R} \to \mathbb{R}$ a neural network with input $S_{t_j}$.
- **d)** Train the hedge for a call $g(S_T) = (S_T - K)^+$ with $K = 1$, $T = 1/4$ by minimizing loss (2). The Black–Scholes call price is
  $C(t, S_t) = \Phi(d_+)S_t - \Phi(d_-)Ke^{-r(T-t)}$ (3), with
  $d_+ = \frac{1}{\sigma\sqrt{T-t}}\big(\log(S_t/K) + (r + \frac{\sigma^2}{2})(T-t)\big)$ and $d_- = d_+ - \sigma\sqrt{T-t}$.
  Compute $p = C(0, S_0)$ with formula (3).
- **e)** Evaluate the P&L $p + \sum_{j=0}^{N-1} H_j \cdot (S_{t_{j+1}} - S_{t_j}) - g(S_T)$ on the test set. Plot the histogram and print the empirical mean and standard deviation.
- **f)** The analytical solution is $H^{\text{BS}}_t(s) = \partial C(t, s)/\partial s$.
  - **i)** Derive a closed-form formula for $H^{\text{BS}}_t(s)$ from (3).
  - **ii)** Evaluate the hedging losses on the test set with $H^{\text{BS}}$ rebalanced at $t_0, \dots, t_{N-1}$. Plot the histogram, print mean and standard deviation, and compare with (e).
- **g)** For $j \in \{0, 10, 20, 29\}$, plot $s \mapsto H_j(s)$ against $s \mapsto H^{\text{BS}}_{t_j}(s)$ for $s \in [0.5, 1.5]$. For which times $t_j$ are the two functions most similar? Why?

## 2. Heston model

$$dS_t = \sqrt{V_t}\,S_t\,dW_t,\ S_0 = s_0; \qquad dV_t = a(b - V_t)\,dt + \sigma\sqrt{V_t}\,dW'_t,\ V_0 = v_0, \tag{4}$$

with correlation $\rho$ between $W$ and $W'$. The model is incomplete, so look for a hedge minimizing

$$p = \inf_{H \in \mathcal{H}} R\Big(g(S_T) - \int_0^T H_u\,dS_u\Big) \tag{5}$$

for a risk measure $R$; $p$ is the price charged for selling the option. With $R = \text{AVaR}_\alpha$, $\text{AVaR}_\alpha(X) = \frac{1}{1-\alpha}\int_\alpha^1 q_u(X)\,du = \inf_{w \in \mathbb{R}}\big(w + \frac{\mathbb{E}[(X - w)^+]}{1-\alpha}\big)$, problem (5) becomes the minimization over $w$ and $H_0, \dots, H_{N-1}$ of

$$\frac{1}{m}\sum_{i=1}^m \Big(w + \frac{1}{1-\alpha}\Big(g\big(s^{(i)}_T\big) - \sum_{j=0}^{N-1} H_j \cdot \big(s^{(i)}_{t_{j+1}} - s^{(i)}_{t_j}\big) - w\Big)^+\Big). \tag{6}$$

- **a)** Simulate 100,000 training and 10,000 test paths of $S$ and $V$ with $N = 30$, $s_0 = 1$, $T = 1/4$, $\rho = -0.3$, $a = 4$, $b = 0.5$, $v_0 = 0.5$, $\sigma = 1$.
  - **i)** Simulate $V$ exactly: $V_{t_{j+1}} = c \cdot C_j$ with $c = \frac{\sigma^2}{4a}(1 - e^{-aT/N})$ and $C_j$ non-central chi-square with $\frac{4ab}{\sigma^2}$ degrees of freedom and non-centrality $e^{-aT/N}(V_{t_j}/c)$ (`scipy.stats.ncx2`).
  - **ii)** Simulate $S$ with the simplified Broadie–Kaya scheme:
    $S_{t_{j+1}} = S_{t_j}\exp\Big(\frac{\rho}{\sigma}\big((V_{t_{j+1}} - V_{t_j}) - ab\frac{T}{N}\big) + \big(\frac{a\rho}{\sigma} - \frac12\big)V_{t_j}\frac{T}{N} + \sqrt{(1-\rho^2)V_{t_j}\frac{T}{N}}\,Z_{j+1}\Big)$.
- **b)** Implement the approach with each $H_j$ a neural network with input $(S_{t_j}, V_{t_j})$.
- **c)** Train the model for the call of 1.d) by minimizing loss (6) for $\alpha = 0.5$ and $\alpha = 0.9$.
- **d)** Compare the prices $p$ for both $\alpha$ by evaluating loss (6) on the test set. Which one is higher? Why?
- **e)** Compare the P&L $p + \sum_{j=0}^{N-1} H_{t_j}(S_{t_{j+1}} - S_{t_j}) - g(S_T)$ on the test set for both $\alpha$. Plot the two histograms and report the empirical means and standard deviations.

## Template requirements (`template3.ipynb`)

- Make sure all requested results are visible in the executed notebook.
- A concise **Project Report** (Approach, Key results, Interpretation, Robustness, Limitations) is required at the end of the notebook.
