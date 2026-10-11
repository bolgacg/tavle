# Held-out results, January 2024 to September 2026

The single held-out run of the New York study, scored once on 10 and 11 October 2026 from the code frozen at f0636cc (research/us/FREEZE), and the read-only audit of its results. These are copies of the files in the run folder on the training machine, published so that every number on the page can be traced.

| File | What it is |
|---|---|
| AUDIT.md, audit_analysis.json | The audit of the results (report, and every number in it as JSON): event dependence, excess over the rules with no model with intervals, placebos, the leak check, data checks, why two steps crashed, and a verdict per row. |
| pre_vs_heldout.md | Every row's build-year result beside its held-out result. |
| v2_results_heldout.md, lab_results_heldout.md | The two scorers' held-out tables (version 2 and version 1), as the run wrote them. |
| registered_verdicts_heldout.json | The registered verdict run: ideas A to D against the habit rule, with the Holm correction (objectives 5 and 6). |
| DONE_audit.md, heldout_summary_rebuilt.py | The run's summary step crashed after scoring; this is the summary rebuilt outside the frozen code, with the script that wrote it (a two-line guard and new paths). |
| analysis_2025_and_correlation.md | Why the pair rows earned so much in 2025, how the candidate parts of a book co-move, and the two books. |
| a_2025.py, a_2025.txt, a_v1cdeep.txt | The 2025 analysis: script and outputs. |
| b_corr.py, b_corr.txt | Correlations of daily profit in three periods, and yearly sums. |
| b_books.py, b_books.csv, b_books.txt | The books sized to my drawdown limits. |

Wording edits made in these copies, nothing else: the row names ending "[lead]" read "[second build]", as on the page; in AUDIT.md the run's driver script is called "the run script" (twice); in analysis_2025_and_correlation.md the drawdown limit is described as mine; in b_books.py one docstring word about who asked for the change is replaced by "revised limits"; and the sign of the city-minus-Capital gap is corrected to match a_2025.txt (the working copy had written "CAPITL minus N.Y.C." for numbers that are N.Y.C. minus CAPITL).
