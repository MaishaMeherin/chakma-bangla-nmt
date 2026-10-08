"""Paired bootstrap test: is system B better than system A on the test set? (thesis Section 5.3)

    python scripts/bootstrap_significance.py \
        --a outputs/back_translation/base_eval/test_predictions.csv \
        --b outputs/back_translation/iter1/eval/test_predictions.csv

Both files come from evaluate_model.py (columns: reference, prediction) and must have the same rows.
Score = mean sentence-BLEU (NLTK, smoothing method 1, whitespace tokens), resampled 10,000
times with seed 42, exactly as in the thesis. p = share of resamples where B - A <= 0.
"""

import argparse

import _setup  # noqa: F401
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from chakma_nmt.utils import save_json


def sentence_scores(preds, refs):
    smooth = SmoothingFunction().method1
    return np.array([sentence_bleu([r.split()], str(p).split(), smoothing_function=smooth) for p, r in zip(preds, refs)])


def bootstrap(scores_a, scores_b, n=10_000, seed=42):
    rng = np.random.RandomState(seed)
    k = len(scores_a)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.choice(k, size=k, replace=True)
        diffs[i] = scores_b[idx].mean() - scores_a[idx].mean()
    return {
        "observed_diff": float(scores_b.mean() - scores_a.mean()),
        "p_value": float((diffs <= 0).mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "n_bootstrap": n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="baseline predictions CSV")
    p.add_argument("--b", required=True, help="improved system predictions CSV")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--out", default=None, help="optional JSON output path")
    a = p.parse_args()

    da, db = pd.read_csv(a.a).fillna(""), pd.read_csv(a.b).fillna("")
    if len(da) != len(db) or (da["reference"] != db["reference"]).any():
        raise SystemExit("The two files do not contain the same test sentences in the same order.")
    refs = da["reference"].tolist()
    res = bootstrap(sentence_scores(da["prediction"], refs), sentence_scores(db["prediction"], refs), a.n)
    print(f"mean sentence-BLEU difference (B - A): {res['observed_diff']:+.4f}")
    print(f"95% CI: [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]   p = {res['p_value']:.4f}"
          f"   -> {'significant' if res['p_value'] < 0.05 else 'not significant'} at 0.05")
    if a.out:
        save_json(res, a.out)


if __name__ == "__main__":
    main()
