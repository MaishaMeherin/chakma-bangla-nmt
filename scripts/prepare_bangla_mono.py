"""Build the 50,000-sentence monolingual Bangla sample used for back-translation (thesis Table 4.6).

The source is the Bangla side of the csebuetnlp/BanglaNMT training set (2,379,749 sentences,
~456 MB, too big for GitHub). The finished sample is already committed at
data/monolingual/bangla_mono_50k.csv, so you only need this script to rebuild it or change it.

    python scripts/prepare_bangla_mono.py --input /path/to/BanglaNMT_train.csv

`--input` is a CSV with a `bn` column (one sentence per row), in the original corpus order.
Steps: BanglaT5 normalisation, keep sentences with >= 2 tokens, de-duplicate (order kept),
drop sentences that are also Bangla targets in the gold training data, keep the first 50,000.
"""

import argparse

import _setup  # noqa: F401
import pandas as pd

from chakma_nmt.data import TGT, load_parallel, normalize_pairs, select_bangla_monolingual
from chakma_nmt.utils import get_logger, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="BanglaNMT_train.csv (column: bn)")
    p.add_argument("--train", default="data/parallel/train.csv")
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--out", default="data/monolingual/bangla_mono_50k.csv")
    a = p.parse_args()

    logger = get_logger()
    raw = pd.read_csv(a.input, encoding="utf-8")[TGT].dropna().tolist()
    gold = normalize_pairs(load_parallel(a.train, keep_nan_rows=False, name="train"))
    sample, stats = select_bangla_monolingual(raw, gold[TGT].tolist(), a.n)
    pd.DataFrame({TGT: sample}).to_csv(a.out, index=False)
    save_json(stats, a.out.replace(".csv", "_stats.json"))
    for k, v in stats.items():
        logger.info(f"  {k:32s} {v:>10,}")
    logger.info(f"Wrote {len(sample):,} sentences to {a.out}")


if __name__ == "__main__":
    main()
