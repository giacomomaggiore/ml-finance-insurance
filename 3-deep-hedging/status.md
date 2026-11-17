# Project 3 status

- Project: Deep Hedging (Buehler et al., 2019).
- Models: Black–Scholes (Exercise 1) and Heston (Exercise 2).
- Notebook: `3-main.ipynb` (copy of `template3.ipynb` with the review fixes; `template3.ipynb` is kept unchanged as the previous version).
- `3-main.ipynb` is executed top to bottom in a fresh kernel (cells 1–14 in order, about 8 minutes on CPU). Ex 1 uses `torch.manual_seed(1)` and numpy seed 0; Ex 2 uses its own seeds (numpy 1, torch 2).
- Deadline: 25 November 2026 (last questions: 18 November).

## Completed

- Exercise 1 (Questions 1.a–1.g):
  - 1.a: Itô's formula shows that $S_t = s_0 e^{\sigma W_t - \frac12\sigma^2 t}$ solves $dS_t = \sigma S_t\,dW_t$.
  - 1.b: 100,000 training and 10,000 test paths with the exact GBM update ($N = 30$, $s_0 = 1$, $T = 1/4$, $\sigma = 0.35$).
  - 1.c: 30 separate networks $H_0, \dots, H_{29}$, each 1 → 32 → 32 → 1 with tanh; each $H_j$ sees only $S_{t_j}$.
  - 1.d: fixed premium $p = C(0, s_0) = 0.069726$ (not trained). `demo.ipynb` setup: Adam, `lr=1e-2`, `batch_size=1000`, 100 epochs, MultiStepLR at epoch 75. The loss plot shows the epoch-average loss.
  - 1.e–1.f: deep hedge P&L mean 8.9e-06, std 0.01115; BS delta P&L mean 5.9e-06, std 0.01105; std of the path-wise difference 0.00131. The remaining error comes from discrete rebalancing (30 dates). Answer written in Markdown.
  - 1.g: the networks match delta best at later dates; $H_0$ only sees $s = 1$. Supporting plot: `output/1g_training_data_seen.png`.
- Exercise 2 (Questions 2.a–2.e):
  - 2.a: V simulated exactly with noncentral chi-square draws (`rng.noncentral_chisquare`, same distribution as `scipy.stats.ncx2`), S with the simplified Broadie–Kaya scheme. The Heston vol of variance is `sigma_v` (does not overwrite Ex 1's `sigma`). Checks: min V = 0.0059, mean $S_T$ = 0.999 (martingale).
  - 2.b: `hedged_loss` (X = payoff − trading gain) shared by the AVaR objective (6) and the P&L in 2.e.
  - 2.c: Adam `lr=1e-3`, batch 1000, 100 epochs, w trained jointly; loss curves are flat after about 40 epochs.
  - 2.d: p = 0.1606 (α = 0.5) and 0.1937 (α = 0.9). Higher at 0.9 because AVaR is non-decreasing in α. Trained w ≈ test α-quantile of X (0.1349 vs 0.1353; 0.1760 vs 0.1763).
  - 2.e: P&L mean 0.0229 / 0.0560, std 0.0312 / 0.0330 (α = 0.5 / 0.9). Mean P&L = p − E[payoff] (mean trading gain ≈ 0, std. error 0.0024). The mean difference (0.0330) equals the price difference (0.0331); the α = 0.9 hedge has a thinner left tail (1% quantile of P&L − p: −0.217 vs −0.232) but a slightly larger std.
- Review fixes (items 1–18) applied in `3-main.ipynb`: professor comments and checklist restored, answers moved from `print` to Markdown, dead code and wrong comments removed.
- `assignment.md` now contains the transcribed assignment from `slides/Pr3_26.pdf`.

## Experiments

- Earlier (not in the notebook): tanh instead of ReLU and different numbers of epochs reached similar performance (P&L std ≈ 0.011). Not claimed in the notebook, since no cell computes it.

## To do

- Project Report at the end of the notebook (use the numbers above, all computed in `3-main.ipynb`).
