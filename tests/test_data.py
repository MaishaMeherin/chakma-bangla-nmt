"""Check that data loading reproduces the counts reported in the thesis (no GPU, ~30 s)."""

from pathlib import Path

import pandas as pd
import pytest

from chakma_nmt.data import SRC, TGT, load_chakma_monolingual, load_parallel, load_splits

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "train": ROOT / "data/parallel/train.csv",
    "dev": ROOT / "data/parallel/dev.csv",
    "test": ROOT / "data/parallel/test.csv",
    "split": "external_dev",
}


def test_parallel_counts_back_translation():
    train, dev, test = load_splits(dict(DATA, keep_nan_rows=False))
    assert (len(train), len(dev), len(test)) == (15_954, 2_994, 600)   # thesis Table 4.4


def test_parallel_counts_baseline():
    # The baseline notebook kept 6 rows with a missing Chakma cell -> 15,960 (thesis Table 4.3)
    assert len(load_parallel(DATA["train"], keep_nan_rows=True)) == 15_960


def test_indomain_split_sizes():
    train, dev, test = load_splits(dict(DATA, split="indomain_80_10_10", keep_nan_rows=True))
    assert (len(train), len(dev), len(test)) == (12_768, 1_596, 1_596)


@pytest.fixture(scope="module")
def chakma_mono():
    train, dev, test = load_splits(dict(DATA, keep_nan_rows=False))
    sources = set(train[SRC]) | set(dev[SRC]) | set(test[SRC])
    return load_chakma_monolingual(ROOT / "data/monolingual/chakma_mono.csv", sources)


def test_chakma_monolingual_count(chakma_mono):
    assert len(chakma_mono) == 38_032   # thesis Table 4.5


def test_committed_synthetic_data_matches_pipeline(chakma_mono):
    """The Iteration-2 synthetic Chakma->Bangla file keeps 37,980 of the 38,032 sentences, in order."""
    syn = pd.read_csv(ROOT / "data/synthetic/iter2_ccp2bn.csv")[SRC].tolist()
    assert len(syn) == 37_980
    it = iter(chakma_mono)
    assert all(any(s == m for m in it) for s in syn)


def test_bangla_monolingual_sample():
    bn = pd.read_csv(ROOT / "data/monolingual/bangla_mono_50k.csv")[TGT]
    assert len(bn) == 50_000 and bn.is_unique
