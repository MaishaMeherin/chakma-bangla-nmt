"""Score a trained Chakma -> Bangla model on the out-of-domain test set (thesis protocol).

    python scripts/evaluate_model.py --model outputs/base/final_model
    python scripts/evaluate_model.py --model outputs/back_translation/iter1/forward --out results/iter1_forward

Writes <out>/test_metrics.json and <out>/test_predictions.csv (source, reference, prediction).
"""

import argparse
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd

from chakma_nmt.data import SRC, TGT, load_ood_test
from chakma_nmt.metrics import score_corpus
from chakma_nmt.modeling import load_model, load_tokenizer, translate
from chakma_nmt.utils import device, get_logger, save_json


def evaluate_model(model, tokenizer, test_df, out_dir, num_beams=5, max_new_tokens=128, batch_size=1,
                   max_source_length=128, logger=None):
    preds = translate(model, tokenizer, test_df[SRC].tolist(), num_beams=num_beams, max_new_tokens=max_new_tokens,
                      max_source_length=max_source_length, batch_size=batch_size, desc="Test set")
    metrics = score_corpus(preds, test_df[TGT].tolist())
    metrics.update(num_beams=num_beams, max_new_tokens=max_new_tokens, batch_size=batch_size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source": test_df[SRC], "reference": test_df[TGT], "prediction": preds}).to_csv(
        out_dir / "test_predictions.csv", index=False)
    save_json(metrics, out_dir / "test_metrics.json")
    if logger:
        logger.info(f"Test BLEU {metrics['bleu']:.2f} | chrF {metrics['chrf']:.2f} | METEOR {metrics['meteor']:.2f} "
                    f"-> {out_dir / 'test_metrics.json'}")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="folder with a saved model + tokenizer")
    p.add_argument("--test", default="data/parallel/test.csv")
    p.add_argument("--out", default=None, help="output folder (default: <model>/eval)")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1, help="1 = thesis protocol; larger is faster")
    p.add_argument("--limit", type=int, default=None, help="only the first N sentences (quick check)")
    a = p.parse_args()

    logger = get_logger()
    test_df = load_ood_test({"test": a.test})
    if a.limit:
        test_df = test_df.head(a.limit)
    tokenizer = load_tokenizer(a.model)
    model = load_model(a.model, tokenizer).to(device())
    evaluate_model(model, tokenizer, test_df, a.out or Path(a.model) / "eval", a.num_beams, a.max_new_tokens,
                   a.batch_size, logger=logger)


if __name__ == "__main__":
    main()
