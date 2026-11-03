# Status

- Exercises 1, 2.a, 2.b done; 2.c done (FDR/TPR curves, AUC, interpretation cell).
- Fixed SVM `gamma` to 0.1 (assignment kernel `exp(-10^-1 ||x-x'||^2)`); was 1.0.
- Exercise 3 code done: boolean scenario matrix D in column blocks; P&L histograms, expected P&L and 95%-VaR for strategies (i)–(iii).
- Review (2026-10-18): no bugs; all numbers verified independently; fresh-kernel run reproduces outputs (~2 min, ~1.5 GB RAM).
- Project Report written (Approach, Model selection, Key results, Interpretation, Robustness, Limitations), appended below the professor's instruction lines.
- Report uses rounded expected P&L (≈ 22,000 / 4,900 / 62,000 CHF). Saved outputs now come from a fresh top-to-bottom run: 22,459.80 / 4,880.55 / 61,960.54; VaRs -2,455 / 6,840 / -54,070.
- Report's Robustness section cites a separate 3-seed check (seeds 1–3, run outside the notebook): (iii) best every time (CHF 66–68k); (i) vs (ii) not stable. Limitations cites a separate age-band logistic check (test cross-entropy 0.043).
- Audit fixes (2026-11-03): Ex 3.c Markdown; FDR 0/0 convention; Platt randomness in 2.b(iv) and Robustness; Ex 3 helper, summary table, grid-aligned histogram bins; template comments restored; `probability` warning filtered; small comment and name cleanups.
- SVM fits use the global random state: removing or reordering a fit changes D, so re-check "Key results" after any such change.
- Fresh-kernel run after the audit fixes (2026-11-03): all numbers unchanged, no warnings, execution numbers 1–8. Group names added in the first cell (no re-run needed).
- Open: submit on Moodle by Nov 4.
