# Results

Numbers from the thesis runs. How they were produced and how to reproduce them is explained in the
main [README](../README.md#8-results).

| File | Contents |
|---|---|
| `test_metrics.json` | Out-of-domain test scores (BLEU, chrF, METEOR) for the baseline and both back-translation iterations, the bootstrap test, and the Chapter 4 baseline figures |
| `training_curves/base.csv` | Baseline, 30 epochs: validation loss / BLEU / chrF++ / METEOR / WER |
| `training_curves/base_indomain_80_10_10.csv` | 80/10/10 in-domain baseline |
| `training_curves/iter1_backward.csv` | Iteration 1, Bangla -> Chakma (20 epochs) |
| `training_curves/iter1_forward.csv` | Iteration 1, Chakma -> Bangla, the best model (20 epochs, best epoch 18) |
| `training_curves/iter2_backward.csv` | Iteration 2, Bangla -> Chakma (30 epochs) |
| `training_curves/iter2_forward.csv` | Iteration 2, Chakma -> Bangla (30 epochs) |

The training curves were taken from the logs printed in `notebooks/archive/`. `train_loss_last_logged`
is the last training loss logged before each evaluation, not an epoch average.

## Figures (`figures/`)

| Figure | Shows |
|---|---|
| `final_results_table.png` | Test BLEU / chrF / METEOR for base, Iteration 1 and Iteration 2 |
| `base_vs_best_ibt.png` | Baseline against Iteration 1 forward |
| `ibt_overiteration.png` | Score progression base -> Iter 1 -> Iter 2 |
| `chakmanmt_comparison.png` | This work against ChakmaNMT results on the same test set |
| `bootstrap_significance.png` | Paired bootstrap distributions for both comparisons |
| `iter1_forward_all_metrics.png` | Validation BLEU / chrF / METEOR per epoch, Iteration 1 forward |
| `iter1_forward_overfitting.png` | Train against validation loss, Iteration 1 forward |
| `iter1_forward_train_loss.png` | Step-level training loss, Iteration 1 forward |
| `iter1_forward_lr_schedule.png` | Learning-rate schedule (800 warm-up steps, linear decay) |
| `data_mix_composition.png` | Gold and synthetic share of each training mix |
| `anchor_repeat_effect.png` | Effect of repeating gold pairs 2x on the mix |
| `synthetic_data_volume.png` | Synthetic pairs kept after filtering, per step |
| `length_distribution.png` | Sentence-length distributions |
| `hyperparameter_table.png` | Back-translation hyperparameters |
