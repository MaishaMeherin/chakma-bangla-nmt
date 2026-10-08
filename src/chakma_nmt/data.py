"""Loading, cleaning and normalising the parallel and monolingual data.

Column names used throughout:
    chakma_transliterated  Chakma written in Bangla script (model source side)
    bn                     Bangla (model target side)
    ccp                    Chakma in its own script (kept for reference, not used for training)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SRC = "chakma_transliterated"
TGT = "bn"

logger = logging.getLogger("chakma_nmt")

try:
    from normalizer import normalize as _bt5_normalize

    NORMALIZER_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    NORMALIZER_AVAILABLE = False
    _bt5_normalize = None


def normalize(text):
    """BanglaT5 normaliser (csebuetnlp/normalizer, NFKC). Every thesis run used it."""
    if not NORMALIZER_AVAILABLE:
        raise ImportError(
            "The BanglaT5 normaliser is not installed. Run:\n"
            "  pip install git+https://github.com/csebuetnlp/normalizer"
        )
    return _bt5_normalize(text, unicode_norm="NFKC")


def _ws(text):
    return " ".join(str(text).split()).strip()


def clean_parallel(df, keep_nan_rows=False, name="data"):
    """Whitespace-normalise, drop empty pairs and drop exact duplicate pairs.

    keep_nan_rows=True copies the baseline notebook, which converted missing cells to the
    string "nan" *before* dropping empties. That kept 6 training rows whose Chakma side is
    the literal word "nan" (15,960 pairs). The back-translation pipeline dropped them
    (15,954 pairs). Use True only to reproduce the thesis baseline exactly.
    """
    missing = {SRC, TGT} - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; found {list(df.columns)}")
    df = df.copy()
    if not keep_nan_rows:
        df = df.dropna(subset=[SRC, TGT])
    for col in (SRC, TGT):
        df[col] = df[col].astype(str).map(_ws)
    df = df.replace({SRC: {"": np.nan}, TGT: {"": np.nan}}).dropna(subset=[SRC, TGT])
    before = len(df)
    df = df.drop_duplicates(subset=[SRC, TGT]).reset_index(drop=True)
    logger.info(f"  {name}: {len(df):,} pairs ({before - len(df):,} duplicates removed)")
    return df


def normalize_pairs(df):
    df = df.copy()
    for col in (SRC, TGT):
        df[col] = df[col].map(normalize)
    return df


def load_parallel(path, keep_nan_rows=False, name=None):
    df = pd.read_csv(path, encoding="utf-8")
    return clean_parallel(df, keep_nan_rows=keep_nan_rows, name=name or str(path))


def load_splits(data_cfg):
    """Return normalised (train, dev, test) DataFrames as described by the `data:` config block.

    split = "external_dev"       all of train.csv for training, dev.csv for validation (thesis default)
    split = "indomain_80_10_10"  80/10/10 split of train.csv stratified by Chakma length
                                 (thesis Experiment 1); `test` is then the in-domain 10%.
    The out-of-domain benchmark (test.csv) is always loaded separately with load_ood_test().
    """
    keep_nan = data_cfg.get("keep_nan_rows", False)
    logger.info("Loading parallel data")
    train = load_parallel(data_cfg["train"], keep_nan, "train")
    if data_cfg.get("split", "external_dev") == "external_dev":
        dev = load_parallel(data_cfg["dev"], keep_nan, "dev")
        test = load_ood_test(data_cfg, normalised=False)
    elif data_cfg["split"] == "indomain_80_10_10":
        train, dev, test = _split_80_10_10(train, data_cfg.get("split_seed", 42))
    else:
        raise ValueError(f"Unknown data.split: {data_cfg['split']}")
    logger.info(f"  train {len(train):,} | dev {len(dev):,} | test {len(test):,}")
    return normalize_pairs(train), normalize_pairs(dev), normalize_pairs(test)


def load_ood_test(data_cfg, normalised=True):
    test = load_parallel(data_cfg["test"], data_cfg.get("keep_nan_rows", False), "test (out-of-domain)")
    return normalize_pairs(test) if normalised else test


def _split_80_10_10(df, seed):
    """Same procedure as notebooks/archive/banglat5_80_10_10.ipynb."""
    df = df.copy()
    try:
        df["length_bucket"] = pd.qcut(df[SRC].str.len(), q=5, labels=False, duplicates="drop")
        train, holdout = train_test_split(
            df, test_size=0.20, random_state=seed, stratify=df["length_bucket"], shuffle=True
        )
    except ValueError:
        train, holdout = train_test_split(df, test_size=0.20, random_state=seed, shuffle=True)
    dev, test = train_test_split(holdout, test_size=0.5, random_state=seed, shuffle=True)
    drop = lambda d: d.drop(columns=["length_bucket"], errors="ignore").reset_index(drop=True)
    return drop(train), drop(dev), drop(test)


def load_chakma_monolingual(path, exclude_sources):
    """Normalise, de-duplicate (order kept) and drop sentences that appear in any parallel split.

    Thesis Table 4.5: 42,783 -> 38,219 after de-duplication -> 38,032 after overlap removal.
    """
    raw = pd.read_csv(path, encoding="utf-8")[SRC].dropna().tolist()
    normed = [normalize(s) if s else s for s in raw]
    dedup = list(dict.fromkeys(normed))
    exclude = set(exclude_sources)
    clean = [s for s in dedup if s not in exclude]
    logger.info(
        f"Monolingual Chakma: {len(raw):,} raw -> {len(dedup):,} unique -> {len(clean):,} "
        f"after removing {len(dedup) - len(clean):,} parallel overlaps"
    )
    return clean


def load_bangla_monolingual(path):
    """The 50,000-sentence Bangla sample built by scripts/prepare_bangla_mono.py."""
    sents = pd.read_csv(path, encoding="utf-8")[TGT].astype(str).tolist()
    logger.info(f"Monolingual Bangla: {len(sents):,} sentences from {path}")
    return sents


def select_bangla_monolingual(raw_sentences, parallel_bn, n=50_000):
    """Thesis Table 4.6: normalise, keep >= 2 tokens, de-duplicate, drop parallel overlap, take first n."""
    normed = []
    for s in raw_sentences:
        if s and str(s).strip():
            t = normalize(str(s).strip())
            if t and t.strip() and len(t.split()) >= 2:
                normed.append(t)
    dedup = list(dict.fromkeys(normed))
    exclude = set(parallel_bn)
    clean = [s for s in dedup if s not in exclude]
    stats = {
        "raw": len(raw_sentences),
        "after_normalise_and_min_length": len(normed),
        "after_dedup": len(dedup),
        "after_parallel_overlap_removal": len(clean),
        "selected": min(n, len(clean)),
    }
    return clean[:n], stats
