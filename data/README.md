# Data

Everything the pipeline needs is in this folder (about 50 MB), so a fresh clone can train straight away.

```
data/
├── parallel/
│   ├── train.csv          17,037 rows -> 15,954 clean Chakma–Bangla pairs (gold training data)
│   ├── dev.csv             3,005 rows ->  2,994 pairs (validation, picks the best epoch)
│   └── test.csv              600 pairs  out-of-domain news benchmark (RisingNewsChakma)
├── monolingual/
│   ├── chakma_mono.csv    42,783 Chakma sentences -> 38,032 after cleaning (back-translation input)
│   ├── bangla_mono_50k.csv 50,000 Bangla sentences (back-translation input)
│   └── bangla_mono_50k_stats.json   how the 50k sample was selected
└── synthetic/             back-translated pairs from thesis Iteration 2 (reference only)
    ├── iter2_ccp2bn.csv   37,980 rows: real monolingual Chakma + generated Bangla
    └── iter2_bn2ccp.csv   49,974 rows: generated Chakma + real monolingual Bangla
```

## Columns

| Column | Meaning |
|---|---|
| `chakma_transliterated` | Chakma written in **Bangla script**. This is the model's input. |
| `bn` | Bangla. This is the model's output (the reference translation). |
| `ccp` | Chakma in its own script (𑄌𑄋𑄴𑄟𑄳𑄦). The pipeline doesn't use it. |
| `en` | English translation, where available. The pipeline doesn't use it. |

`dev.csv` has no `en` column. `chakma_mono.csv` keeps only `ccp` and `chakma_transliterated`. The
original file also had `bn`/`en` columns, but they belonged to a different corpus and did not line up
with the Chakma rows, so they were removed. The row order is unchanged.

## Sources and attribution

| Data | Source |
|---|---|
| Parallel pairs, Chakma monolingual corpus, 600-sentence test set | A. Chakma, A. Chakma, S. Khisa, C. Tripura, M. Hasan, R. Shahriyar, *ChakmaNMT: Machine Translation for a Low-Resource and Endangered Language via Transliteration*, [arXiv:2410.10219](https://arxiv.org/abs/2410.10219). Obtained from the authors on request. |
| Additional parallel pairs | Chakma "Language POS Tagging Dataset" (Mendeley Data) and MELD, the Multilingual Ethnic Language Dataset |
| Bangla monolingual sample | Bangla side of the [csebuetnlp/BanglaNMT](https://github.com/csebuetnlp/banglanmt) training set (Hasan et al., 2020) |
| Bangla-script transliteration of Chakma | [Aksharamukha](https://github.com/virtualvinodh/aksharamukha-python) (`scripts/transliterate.py`) |

If you use this data, please cite the original sources above.

## How each file was made

- **Parallel data.** The three sources were merged and the ChakmaNMT Chakma-script text was
  transliterated to Bangla script with Aksharamukha. Cleaning (whitespace, empty rows, exact duplicates)
  happens at load time in `src/chakma_nmt/data.py`, so the CSVs here are uncleaned:
  17,037 -> 15,954 train pairs (thesis Table 4.2).
- **`chakma_mono.csv`.** Cleaned at load time: BanglaT5 normalisation, de-duplication
  (42,783 -> 38,219), then removal of any sentence that also appears as a source in train/dev/test
  (-> 38,032; thesis Table 4.5).
- **`bangla_mono_50k.csv`.** Built by `scripts/prepare_bangla_mono.py` from the 2,379,749-sentence
  BanglaNMT file (456 MB, not committed): normalise, keep sentences with ≥ 2 tokens (2,019,406),
  de-duplicate (1,874,919), remove gold Bangla targets (1,869,450), take the first 50,000 (thesis
  Table 4.6). Verified: every Bangla sentence in `synthetic/iter2_bn2ccp.csv` appears in this file in
  the same order, so it is the exact sample the thesis used.
- **`synthetic/`.** Outputs of the thesis Iteration 2 generation steps, after prefix stripping and the
  2.5 length-ratio filter. The Iteration 1 synthetic files were not kept. The pipeline regenerates
  synthetic data itself; these files are for inspection and comparison only.

## Known issue: dev overlaps with train

1,890 of the 2,994 dev source sentences also occur in `train.csv`, and 1,549 dev pairs are exact
copies of training pairs. Validation BLEU is therefore inflated (55.98 for the baseline against 14.57 on
the out-of-domain test set). It is fine for choosing the best epoch, but don't read it as
generalisation. The test set has **no** overlap with train. For future work, consider de-duplicating
dev against train (see "Ideas for future work" in the main README).
