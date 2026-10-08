# Archived thesis notebooks

These are the notebooks that produced the thesis results, **kept as a record with their original
outputs**. The only edit is replacing the lab PC's Windows username in printed paths with `USER`. For
new work use the scripts in `scripts/`. They implement the same pipeline with the same
settings and can be resumed.

| Notebook | Thesis section | Ran on | Replaced by |
|---|---|---|---|
| `01_baseline_banglat5.ipynb` (was `Final_base.ipynb`) | 4.4.6 Baseline 2: all gold pairs, external dev set. Val BLEU 55.98, test BLEU 14.71 | 2026-03-25 | `scripts/train_base.py --config configs/base.yaml` |
| `02_baseline_indomain_80_10_10.ipynb` (was `banglat5_80_10_10.ipynb`) | 4.4.3 Experiment 1: 80/10/10 in-domain split. Test BLEU 30.26 in-domain, 14.09 out-of-domain | 2026-02-04 | `scripts/train_base.py --config configs/base_indomain.yaml` |
| `03_iterative_back_translation.ipynb` (was `updated_BT.ipynb`) | 4.4.7 and Chapter 5: two back-translation iterations, evaluation, plots, bootstrap test | 2026-04-15 to 05-19 | `scripts/run_back_translation.py`, `scripts/evaluate_model.py`, `scripts/bootstrap_significance.py` |

They won't run as-is: paths point to the lab PC layout (`../../data/...`,
`./outputs/banglat5_final_20260325_023556/final_model`).

## Naming in `03_iterative_back_translation.ipynb`

The notebook counts *steps*, not thesis iterations, so its names are shifted by one:

| Notebook name | Thesis name | What it is | Test BLEU |
|---|---|---|---|
| `synthetic_iter1.csv` | Iteration 1, synthetic Bangla | base model translates 38,032 monolingual Chakma sentences | |
| `iter1_backward_model` | **Iteration 1 backward** | Bangla → Chakma, 20 epochs | |
| `synthetic_iter2.csv` | Iteration 1, synthetic Chakma | backward model translates 50k Bangla sentences | |
| `iter2_forward_model` | **Iteration 1 forward (best, deployed)** | Chakma → Bangla, 20 epochs, best epoch 18 | **19.00** |
| `synthetic_iter3_bn.csv` | Iteration 2, synthetic Bangla | | |
| `iter3_backward_model` | **Iteration 2 backward** | 30 epochs | |
| `synthetic_iter3_ck.csv` | Iteration 2, synthetic Chakma | | |
| `iter3_forward_model` | **Iteration 2 forward** | 30 epochs | 18.01 |
| `final_bleu_iter2` / `final_bleu_iter3` in the JSON files | Iteration 1 / Iteration 2 test scores | | |

The `scripts/` pipeline and `results/` use the thesis names.

## Other things to know about notebook 03

- The `CONFIG` cell shows `epochs_iter1 = 30`, but the iteration-1 models were trained with a
  20-epoch ceiling (their logs say so). The value was changed to 30 later for iteration 2. The
  config files use `epochs_per_iteration: [20, 30]`.
- Cells were run out of order across several sessions (some show no execution count, and cell 13
  reloads a saved model). Read it top to bottom as a log, not as something to re-run.
- `quality_filter` (2.5 length ratio) is the filter that was used. `filter_synthetic` and
  `oversample_quality_synthetic` are defined but never called, and the curriculum callback is disabled.
