# Project 3 status

- Project: Deep Hedging (Buehler et al., 2019).
- Models: Black–Scholes (Exercise 1) and Heston (Exercise 2).
- Notebook: `3-main.ipynb` (based on `template3.ipynb`).
- Deadline: 25 November 2026 (last questions: 18 November).

## Completed

- Exercise 1 is finished (Questions 1.a–1.g). Koki promised to make it prettier by tonight.
- Question 1.a: Itô's formula shows that $S_t = s_0 e^{\sigma W_t - \frac12\sigma^2 t}$ solves $dS_t = \sigma S_t\,dW_t$.
- Question 1.b: simulated 100,000 training and 10,000 test paths with the exact GBM update ($N = 30$, $s_0 = 1$, $T = 1/4$, $\sigma = 0.35$); arrays are 100,000 × 31 and 10,000 × 31.
- Question 1.c: 30 separate networks $H_0, \dots, H_{29}$ (`nn.ModuleList`), each 1 → 32 → 32 → 1; 1,153 parameters per network, 34,590 in total. Each $H_j$ sees only $S_{t_j}$.
- Question 1.d: fixed premium $p = C(0, s_0) \approx 0.069725$ (Black–Scholes formula, not trained). Trained jointly on loss (2) with the `demo.ipynb` setup: Adam, `lr=1e-2`, `batch_size=1000`, 100 epochs, MultiStepLR at epoch 75.
- Questions 1.e–1.f: the deep hedge matches the analytical delta hedge $H^{\text{BS}}_t(s) = \Phi(d_+)$ on the test set.
  - Deep hedge P&L: mean −2.29e-06, std 0.0111.
  - BS delta P&L: mean 5.85e-06, std 0.0111.
  - Both are centered at zero; the remaining error comes from discrete rebalancing (30 dates), not from the network.
- Question 1.g: the networks match delta best at later dates. $H_0$ is trained only at $s = 1$ (all paths start there); later networks see a wider price range because the spread of $S_{t_j}$ grows like $\sigma\sqrt{t_j}$. Supporting plot of training prices seen per network: `output/1g_training_data_seen.png`.

## Experiments

- Tried a different activation function (tanh instead of ReLU) and different numbers of epochs. All variants reached similar performance (P&L std ≈ 0.0111, matching delta).

## To do

- Exercise 2 (Heston, AVaR at α = 0.5 and α = 0.9): Questions 2.a–2.e.
  - 2.a: simulate V with `scipy.stats.ncx2` (vectorized over paths, one time step at a time), then S with the simplified Broadie–Kaya scheme.
- Try some variations around performance (architecture, learning rate, epochs) where useful.
- Optional (appendix work only): a short analysis of which changes help and which do not. Nice to have, not required.
- Project Report at the end of the notebook.
