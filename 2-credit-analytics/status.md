# Status

- Exercises 1, 2.a, 2.b done; 2.c done (FDR/TPR curves, AUC, interpretation cell).
- Fixed SVM `gamma` to 0.1 (assignment kernel `exp(-10^-1 ||x-x'||^2)`); was 1.0.
- Exercise 3 code done: boolean scenario matrix D in column blocks; P&L histograms, expected P&L and 95%-VaR for strategies (i)–(iii).
- Review (2026-10-18): no bugs; all numbers verified independently; fresh-kernel run reproduces outputs (~2 min, ~1.5 GB RAM).
- Project Report written (Approach, Model selection, Key results, Interpretation, Robustness, Limitations), appended below the professor's instruction lines.
- Report uses rounded expected P&L (≈ 22,000 / 4,900 / 62,000 CHF) because the saved Ex. 3 output comes from a second run of the cell in the same kernel (22,380 / 4,889 / 62,007); a fresh Restart & Run All gives 22,460 / 4,881 / 61,961. VaRs are identical in both (-2,455 / 6,840 / -54,070).
- Report's Robustness section cites a separate 3-seed check (seeds 1–3, run outside the notebook): (iii) best every time; (i) vs (ii) not stable.
- Open: Ex 3.c Markdown interpretation; 2.c plot readability; Ex 3 helper function and grid-aligned histogram bins; restore removed template comments; group names; final Restart & Run All, then re-check the two tables in "Key results".
