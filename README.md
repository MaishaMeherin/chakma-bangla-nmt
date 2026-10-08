# Chakma → Bangla Neural Machine Translation

Neural machine translation from **Chakma**, an endangered Indo-Aryan language of Bangladesh, India and
Myanmar, into **Bangla**. The model is a fine-tuned [BanglaT5](https://huggingface.co/csebuetnlp/banglat5),
improved with **iterative back-translation**. Chakma is written in Bangla script (transliterated), so
the model can reuse what BanglaT5 already knows about Bangla.

This repository holds the code, data and results of the B.Sc. thesis *"Neural Machine Translation for
Chakma–Bangla: An Exploration of Low-Resource Techniques for Model Optimization"*, BRAC University,
June 2026. It is set up so the work can be reproduced and extended.

**Headline result** (600-sentence out-of-domain news test set):

| Model | BLEU | chrF | METEOR |
|---|---:|---:|---:|
| Fine-tuned BanglaT5 (baseline) | 14.57 | 43.41 | 33.13 |
| **+ back-translation, Iteration 1 (best)** | **19.00** | **51.71** | **40.25** |
| + back-translation, Iteration 2 | 18.01 | 51.59 | 39.74 |
| ChakmaNMT, BanglaT5 + 2 iterations of back-translation (previous best fine-tuned system) | 13.9 | 46.3 | – |
| ChakmaNMT, GPT-4.1 with 400 in-context examples (previous best overall) | 17.8 | 49.6 | – |

---

## Contents

1. [What is in this package](#1-what-is-in-this-package)
2. [Quick start (5 steps)](#2-quick-start-5-steps)
3. [Detailed setup and what to expect](#3-detailed-setup-and-what-to-expect)
4. [How to check the result](#4-how-to-check-the-result)
5. [Quick code check without a GPU](#5-quick-code-check-without-a-gpu-a-few-minutes)
6. [How the pipeline works](#6-how-the-pipeline-works)
7. [Models and hyperparameters](#7-models-and-hyperparameters)
8. [Results](#8-results)
9. [Data and rules compliance](#9-data-and-rules-compliance)
10. [Reproducibility notes](#10-reproducibility-notes)
11. [Troubleshooting](#11-troubleshooting)
12. [Ideas for future work](#12-ideas-for-future-work)

---

## 1. What is in this package

```
chakma-bangla-nmt/
├── README.md                     you are here
├── requirements.txt              pinned to the versions used for the thesis
├── pyproject.toml                makes `src/chakma_nmt` installable (pip install -e .)
├── configs/
│   ├── base.yaml                 baseline fine-tuning (the starting model)
│   ├── base_indomain.yaml        baseline on an 80/10/10 in-domain split (thesis Experiment 1)
│   └── back_translation.yaml     two-iteration back-translation (the main method)
├── data/                         all data needed to train, about 50 MB (see data/README.md)
│   ├── parallel/                 train / dev / test Chakma–Bangla pairs
│   ├── monolingual/              Chakma and Bangla monolingual text for back-translation
│   └── synthetic/                back-translated pairs from the thesis run (reference)
├── src/chakma_nmt/               the library
│   ├── data.py                   loading, cleaning, normalisation, monolingual selection
│   ├── filters.py                prefix stripping + 2.5 length-ratio filter for synthetic pairs
│   ├── metrics.py                BLEU / chrF / chrF++ / METEOR / WER
│   ├── modeling.py               model loading, Trainer setup, batched translation
│   ├── smoke.py                  tiny-model settings for the CPU check
│   └── utils.py                  seeding, logging, YAML config loading
├── scripts/                      command-line entry points
│   ├── train_base.py             step 1: fine-tune BanglaT5 on gold pairs
│   ├── run_back_translation.py   step 2: back-translation iterations (resumable)
│   ├── evaluate_model.py         score any saved model on the test set
│   ├── bootstrap_significance.py paired bootstrap test between two systems
│   ├── translate.py              translate your own sentences
│   ├── prepare_bangla_mono.py    rebuild the 50k Bangla sample from BanglaNMT
│   └── transliterate.py          Chakma script -> Bangla script, for new data
├── tests/                        pytest checks of data counts and filters (no GPU)
├── results/
│   ├── test_metrics.json         final test scores + bootstrap results
│   ├── training_curves/          per-epoch validation scores of every thesis run (CSV)
│   └── figures/                  the thesis figures
└── notebooks/
    ├── README.md                 notebook -> thesis naming map
    └── archive/                  the original notebooks with their outputs (record only)
```

**Not in the repo:** the trained model weights (about 1 GB each; see
[Trained models](#trained-models)) and the 456 MB raw BanglaNMT corpus (its 50k sample is included).

---

## 2. Quick start (5 steps)

On a Linux or Windows machine with an NVIDIA GPU (16 GB is enough):

```bash
# 1. Get the code
git clone https://github.com/MaishaMeherin/chakma-bangla-nmt.git
cd chakma-bangla-nmt

# 2. Create an environment (Python 3.10–3.12)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install PyTorch for your CUDA (https://pytorch.org/get-started/locally/), then the rest
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 4. Train the baseline                                   (~5 h on an RTX 4080 SUPER)
python scripts/train_base.py --config configs/base.yaml

# 5. Run back-translation on top of it                    (~2–2.5 days for both iterations)
python scripts/run_back_translation.py --config configs/back_translation.yaml
```

The final scores print at the end and are saved to `outputs/back_translation/summary.json`.
The best model is `outputs/back_translation/iter1/forward/`.

Before starting a long GPU run on a new machine, run the 3-minute check in [Section 5](#5-quick-code-check-without-a-gpu-a-few-minutes).

---

## 3. Detailed setup and what to expect

### Hardware

The thesis ran on one **NVIDIA RTX 4080 SUPER (16 GB)** under Windows with Python 3.12,
PyTorch 2.10 + CUDA 12.6 and transformers 4.36.2. Batch size 4 with 16 gradient-accumulation steps
(effective batch 64) fits in 16 GB, even in fp32. On a smaller GPU, lower `train.batch_size` and raise
`train.grad_accum` by the same factor so the effective batch stays 64.

### Installation details

- `requirements.txt` pins `transformers==4.36.2`, the version behind the thesis numbers. The code
  also runs on newer 4.x releases (checked with 4.57.6), but only 4.36.2 was used for the reported scores.
- The **BanglaT5 normaliser** is installed from GitHub (`pip install git+https://github.com/csebuetnlp/normalizer`).
  It is required, and the code stops with an install hint if it's missing.
- The first run downloads `csebuetnlp/banglat5` (about 1 GB) from Hugging Face, plus the METEOR
  WordNet data (NLTK), into your cache.

### What each command does and how long it takes

| Step | Command | Output | Time (RTX 4080 SUPER) |
|---|---|---|---|
| Baseline | `python scripts/train_base.py --config configs/base.yaml` | `outputs/base/final_model/`, `results.json` | 5 h (30 epochs) |
| Iter 1, generate Bangla | `run_back_translation.py` (automatic) | `iter1/synthetic_ccp2bn.csv` | ~20 min |
| Iter 1, backward model | ″ | `iter1/backward/` | 8 h (20 epochs) |
| Iter 1, generate Chakma | ″ | `iter1/synthetic_bn2ccp.csv` | ~25 min |
| Iter 1, forward model | ″ | `iter1/forward/` ← **best model** | 9 h (20 epochs) |
| Iter 2, all four steps | ″ | `iter2/...` | ~33 h (30-epoch ceiling) |
| Test evaluation | automatic after each forward model | `iterK/eval/test_metrics.json` | ~10 min each |

To run only the iteration that matters (Iteration 1), use `--iterations 1`.

**Resuming.** Every step writes its output and is skipped when that output already exists. If the
machine restarts, run the same command again and it continues from the last finished step. To redo
a step: `--force train_forward` (choices: `gen_bn train_backward gen_ccp train_forward evaluate`).

**Changing settings** without editing YAML: `--set train.epochs=10 train.batch_size=2`.

### Trained models

Each model is about 1 GB, too big for GitHub. The thesis checkpoints are kept on the lab PC/drive.
The local backup has only the two backward models (`itr1_backward_model`, `itr2_backward_model`); the
best model (Iteration 1 forward) has `config.json` but **no weights** there. To share a model, upload
its folder to the Hugging Face Hub:

```bash
huggingface-cli login
huggingface-cli upload <your-username>/chakma-bangla-banglat5-bt outputs/back_translation/iter1/forward .
```

Then anyone can use it with `python scripts/translate.py --model <your-username>/chakma-bangla-banglat5-bt "..."`.

---

## 4. How to check the result

1. **Look at the summary** printed at the end of `run_back_translation.py`:
   ```
   Test results (BLEU / chrF / METEOR):
     base             14.57  43.41  33.13
     iter1_forward    19.00  51.71  40.25
     iter2_forward    18.01  51.59  39.74
   ```
   Expect to land within about ±0.5 BLEU of these numbers. GPU kernels are not bit-exact across
   machines and drivers (see [Section 10](#10-reproducibility-notes)). What should hold is the shape:
   Iteration 1 clearly beats the baseline, and Iteration 2 does not beat Iteration 1.
2. **Compare with the thesis numbers** in `results/test_metrics.json`, and compare per-epoch
   validation BLEU with `results/training_curves/*.csv`. For example, Iteration 1 forward should peak
   near 69.8 validation BLEU around epoch 18.
3. **Test significance** of the improvement:
   ```bash
   python scripts/bootstrap_significance.py \
       --a outputs/back_translation/base_eval/test_predictions.csv \
       --b outputs/back_translation/iter1/eval/test_predictions.csv
   ```
   Expect p < 0.001 and a 95% CI that does not include 0.
4. **Read the translations.** `outputs/.../eval/test_predictions.csv` has source, reference and
   prediction side by side. To score any saved model:
   `python scripts/evaluate_model.py --model <model folder>`.
5. **Translate your own sentence:**
   `python scripts/translate.py --model outputs/back_translation/iter1/forward "মুই গম আগং"`

---

## 5. Quick code check without a GPU (a few minutes)

Use this to confirm a machine is set up correctly before committing a GPU for days. It runs the
**whole** pipeline on CPU with a tiny randomly initialised T5 (same vocabulary), a few dozen
sentences and one epoch. Everything goes to `outputs/smoke/`. The scores are meaningless (around 0
BLEU); the check only shows that every stage runs.

```bash
pytest -q                                                    # 9 tests: data counts + filters (~5 s)
python scripts/train_base.py --smoke                          # baseline pipeline (~1 min)
python scripts/run_back_translation.py --smoke                # both back-translation iterations (~1 min)
python scripts/train_base.py --config configs/base_indomain.yaml --smoke   # 80/10/10 variant
```

The tests check that loading the data reproduces the thesis counts: 15,954 / 2,994 / 600 parallel
pairs, 15,960 for the baseline variant, a 12,768 / 1,596 / 1,596 in-domain split, 38,032 monolingual
Chakma sentences and the 50,000-sentence Bangla sample. Verified on macOS (CPU) with transformers
4.36.2 and 4.57.6.

---

## 6. How the pipeline works

```
                  gold parallel data (15,954 Chakma–Bangla pairs)
                                   │
                   ┌───────────────▼────────────────┐
                   │  Baseline: fine-tune BanglaT5  │  Chakma -> Bangla, 30 epochs
                   └───────────────┬────────────────┘
                                   │  forward model F0
   ┌───────────────────────────────▼──────────────────────────────── one iteration ──┐
   │ 1. F(k-1) translates 38,032 monolingual Chakma  -> synthetic Bangla              │
   │    (strip copied prefixes, drop pairs whose output is > 2.5x the input)          │
   │ 2. Backward model B(k): fresh BanglaT5, Bangla -> Chakma,                        │
   │    trained on gold x2 + step-1 pairs (synthetic Bangla in, real Chakma out)      │
   │ 3. B(k) translates 50,000 monolingual Bangla -> synthetic Chakma (+ filter)      │
   │ 4. Forward model F(k): restart from the baseline, Chakma -> Bangla,              │
   │    trained on gold x2 + step-3 pairs (synthetic Chakma in, real Bangla out)      │
   └───────────────────────────────┬──────────────────────────────────────────────────┘
                                   │  F(1) = best model; F(2) repeats with F(1) as generator
                                   ▼
                     evaluate on the 600-sentence news test set
```

What sets it apart from the ChakmaNMT back-translation setup (thesis Table 4.16):

| Design choice | ChakmaNMT | This work | Why |
|---|---|---|---|
| Training length | fixed 5 epochs | up to 20 (iter 1) / 30 (iter 2) epochs, early stopping (patience 3) | the baseline was still improving at epoch 22–24 |
| Anchor repetition | gold pairs 1x | gold pairs **2x** | keeps real data at about 39% of the training mix instead of 24% |
| Synthetic-data filter | none reported | drop if output length > **2.5x** input | 2.5 is 5.75 SD above the gold mean ratio; rejects only 9 of 15,954 gold pairs |
| Prefix stripping | not reported | strip copied "Translate from Chakma to Bangla:" style prefixes | prefix-contaminated pairs lowered test BLEU |

Text handling: whitespace normalisation, empty and duplicate pairs dropped, and the BanglaT5
normaliser (NFKC) applied to both sides. Chakma is fed in its Bangla-script transliteration; the
original Chakma-script column (`ccp`) is kept but not used.

---

## 7. Models and hyperparameters

All models are `csebuetnlp/banglat5` (T5 encoder–decoder, 12+12 layers, d_model 768, 248M
parameters, 32,100-token SentencePiece vocabulary). The baseline and back-translation runs share the
same hyperparameters, so any difference comes from the data (thesis Tables 4.1 and 4.15):

| Setting | Baseline (`base.yaml`) | Back-translation (`back_translation.yaml`) |
|---|---|---|
| Learning rate / schedule | 3e-4, linear decay, 800 warm-up steps | same |
| Batch | 4 per device x 16 accumulation = 64 | same |
| Epochs | 30 | max 20 (iter 1), 30 (iter 2); early stopping patience 3 on val BLEU |
| Label smoothing / weight decay | 0.3 / 0.01 | same |
| Optimiser | AdamW | AdamW |
| Precision | bf16 (fp16 if bf16 unsupported) | fp32 (fp16 gave NaN losses) |
| Max length | 128 tokens (source and target) | same |
| Checkpoint selection | best validation BLEU (epoch-wise eval) | same |
| Validation decoding | beam 5, max_length 128 | greedy, max_new_tokens 128 (faster) |
| Synthetic generation | – | beam 5, max_new_tokens 128, batch 16 |
| Anchor repetition / length filter | – | 2x / 2.5 |
| Backward model init | – | pretrained BanglaT5 (fresh each iteration) |
| Forward model init | pretrained BanglaT5 | fine-tuned baseline (fresh each iteration) |
| Test decoding | beam 5, max_new_tokens 128, one sentence at a time | same |
| Seed | 42 | 42 |

---

## 8. Results

### Out-of-domain test set (600 news sentences, Chakma -> Bangla)

| Model | BLEU | chrF | METEOR | Epochs run (best) | Training data |
|---|---:|---:|---:|---|---|
| Baseline BanglaT5 | 14.57 | 43.41 | 33.13 | 30 (29) | 15,960 gold |
| **Iteration 1 forward** | **19.00** | **51.71** | **40.25** | 20 (18) | 31,908 gold (2x) + 49,890 synthetic |
| Iteration 2 forward | 18.01 | 51.59 | 39.74 | 30 (29) | 31,908 gold (2x) + 49,974 synthetic |

Iteration 1 improves on the baseline by **+4.43 BLEU (+30% relative), +8.30 chrF and +7.12 METEOR**.
Iteration 2 trains on better synthetic data but scores 0.99 BLEU lower, so Iteration 1 is the final model.

**Statistical significance** (paired bootstrap on sentence-BLEU, 10,000 resamples, recorded in
`notebooks/archive/03_iterative_back_translation.ipynb` and `results/figures/bootstrap_significance.png`):

| Comparison | Mean difference | 95% CI | Result |
|---|---:|---|---|
| Baseline -> Iteration 1 | +0.0391 | [+0.0290, +0.0492] | significant, p < 0.0001 |
| Iteration 1 -> Iteration 2 | −0.0052 | [−0.0123, +0.0017] | **not** significant (CI includes 0) |

So Iteration 2 is not better than Iteration 1, and its drop is within noise.

### Comparison with the literature (same test set)

| System | BLEU | chrF |
|---|---:|---:|
| ChakmaNMT: BanglaT5 baseline | 11.7 | 37.9 |
| ChakmaNMT: BanglaT5 + 2 iterations of back-translation | 13.9 | 46.3 |
| ChakmaNMT: GPT-4.1 + 400-shot in-context learning | 17.8 | 49.6 |
| **This work: Iteration 1 forward** | **19.00** | **51.71** |

That is **+5.10 BLEU / +5.41 chrF** over the best fine-tuned system and **+1.20 BLEU / +2.11 chrF**
over GPT-4.1 with in-context learning.

### Baseline variants (thesis Chapter 4, scored with `Trainer.evaluate`)

| Setup | Validation BLEU | In-domain test BLEU | Out-of-domain test BLEU |
|---|---:|---:|---:|
| 80/10/10 split (`base_indomain.yaml`) | 32.10 | 30.26 | 14.09 |
| All gold + external dev (`base.yaml`) | 55.98 | – | 14.71 |

The gap between validation (about 56) and out-of-domain test (about 15) has two causes: the test
set is formal news while the training data is conversational, and the dev set overlaps with the
training data (see [Section 10](#10-reproducibility-notes)).

### Files

- `results/test_metrics.json`: all numbers above, machine-readable.
- `results/training_curves/*.csv`: per-epoch validation loss, BLEU, chrF++ and METEOR for each of the
  six thesis training runs, taken from the notebook logs.
- `results/figures/`: thesis figures (score progression, training curves, data mix, synthetic data
  volumes, length-ratio analysis, bootstrap test, ChakmaNMT comparison).

---

## 9. Data and rules compliance

- The data comes from **ChakmaNMT** (Chakma et al., [arXiv:2410.10219](https://arxiv.org/abs/2410.10219);
  parallel pairs, monolingual Chakma, the 600-sentence test set), the **Chakma POS Tagging Dataset**
  (Mendeley Data), **MELD**, and **csebuetnlp/BanglaNMT**. Full attribution and processing details are
  in [`data/README.md`](data/README.md). Please cite those sources if you use the data.
- The ChakmaNMT data was obtained from its authors on request. Check with them before using it for
  anything beyond research on this task.
- **No test leakage.** The 600 test sentences never appear in training. Monolingual Chakma sentences
  that appear in any parallel split (train, dev or test) are removed before back-translation (187
  sentences), and monolingual Bangla sentences that are gold training targets are removed too.
- The test set is used only for final evaluation. Checkpoints are chosen on the dev set.
- No personal data is included.

---

## 10. Reproducibility notes

- **Seeds.** Seed 42 everywhere: Python, NumPy, PyTorch, the data shuffle and the 80/10/10 split. The
  baseline also enables cuDNN deterministic mode. GPU training is still not bit-exact across GPUs,
  driver versions or library versions, so expect small differences.
- **Test protocol.** All Chapter 5 numbers use beam search 5, `max_new_tokens=128`, one sentence at a
  time (`test_generation.batch_size: 1`). The baseline also has a Chapter 4 figure of 14.71 BLEU from
  `Trainer.evaluate` (batched, `max_length=128`). The 0.14 BLEU gap to 14.57 comes only from the
  evaluation method. `train_base.py` reports both (`test_out_of_domain_trainer` and `test_out_of_domain`).
- **chrF vs chrF++.** Table 5.1 reports chrF (`word_order=0`). Validation logs and Chapter 4 report
  chrF++ (`word_order=2`). `test_metrics.json` written by `evaluate_model.py` contains both.
- **Baseline 15,960 vs 15,954 pairs.** The baseline notebook turned 6 missing Chakma cells into the
  string "nan" and kept them. `base.yaml` reproduces that (`keep_nan_rows: true`); the back-translation
  pipeline drops them.
- **Dev overlaps train.** 1,549 of the 3,005 dev pairs are exact copies of training pairs, so
  validation BLEU is optimistic. It is fine for picking epochs, but don't read it as generalisation.
- **In-domain split.** The 80/10/10 notebook read an older copy of the training CSV (capitalised
  column names), so the exact rows in each split may differ from the thesis run. Split sizes match.
- **Synthetic generation** in the notebook went through `Trainer.predict` (batch 16, beam 5). The
  script calls `model.generate` with the same settings, so synthetic sentences can differ in rare
  cases, for example where the base checkpoint's saved `max_length` cuts a long output.
- **Single run.** Each configuration was trained once (one seed). Multi-seed runs would show the
  variance.
- **Original notebooks** are in `notebooks/archive/`. The notebook names models one step off from the
  thesis (its `iter2_forward_model` is the thesis "Iteration 1 forward"); the mapping is in
  [`notebooks/README.md`](notebooks/README.md).

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError: The BanglaT5 normaliser is not installed` | `pip install git+https://github.com/csebuetnlp/normalizer` |
| `CUDA out of memory` | `--set train.batch_size=2 train.grad_accum=32` (same effective batch of 64) |
| Loss becomes `nan` in back-translation runs | keep `precision: fp32`; fp16 caused this in the thesis |
| Errors inside `transformers` (unexpected keyword, missing attribute) | you probably have a different major version; install the tested one: `pip install transformers==4.36.2` (4.57.6 also works; 5.x is untested) |
| METEOR fails with `LookupError: wordnet` | `python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"` |
| NLTK/HF downloads fail with SSL errors (macOS) | `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` |
| `Baseline model not found at outputs/base/final_model` | train the baseline first, or point to your copy: `--set base_model=/path/to/final_model` |
| A run stopped half-way | run the same command again; finished steps are skipped |
| Want to redo one step | `--force gen_bn` (or `train_backward`, `gen_ccp`, `train_forward`, `evaluate`) |
| Bangla text shows as `?` in the Windows console | run `chcp 65001` first, or read the CSV/JSON outputs |
| `aksharamukha` fails to import | it needs Python ≤ 3.13; it is only used by `scripts/transliterate.py` |

---

## 12. Ideas for future work

Taken from thesis Section 6.3 and from what this clean-up turned up:

- **Multi-seed runs** of the baseline and Iteration 1 to measure variance (`--set seed=...`).
- **Ablations** of each design choice: `back_translation.anchor_repeat` (1, 1.5, 3),
  `length_ratio_threshold` (none, 2.0, 3.0), `strip_prefixes: false`, shorter training.
- **Longer training.** Three of the four back-translation models were still improving at their epoch
  ceiling (`results/training_curves/`).
- **A clean dev set.** Remove the 1,549 dev pairs that duplicate training pairs, so validation BLEU
  tracks generalisation.
- **Bangla -> Chakma** as a target direction in its own right (the backward models reach about 31–45 validation BLEU).
- **Other domains** (government, dialogue, literature) and **human evaluation** by Chakma speakers.
- **Larger or multilingual models** (mT5, NLLB, IndicTrans2) with the same pipeline: change `model.name`.

---

## Citation

If you use this code, please cite the thesis (BRAC University, Department of Computer Science and
Engineering, 2026) and the data sources listed in [`data/README.md`](data/README.md).
