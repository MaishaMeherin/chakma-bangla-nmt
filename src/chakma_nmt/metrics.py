"""Translation metrics.

Two places compute metrics:
  * During training (validation every epoch): BLEU, chrF++ (word_order=2), METEOR (0-1), and WER
    for the baseline. This is what picks the best checkpoint (by BLEU).
  * Final test evaluation (score_corpus): the protocol behind the thesis results table:
    sacreBLEU, chrF (word_order=0) and METEOR x 100. chrF++ and WER are reported as extras.
"""

import logging

import evaluate
import numpy as np

logger = logging.getLogger("chakma_nmt")

_METRICS = {}


def _metric(name):
    if name not in _METRICS:
        _METRICS[name] = evaluate.load(name)
    return _METRICS[name]


def build_compute_metrics(tokenizer, include_wer=False):
    """compute_metrics callback for Seq2SeqTrainer(predict_with_generate=True)."""

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        vocab_size = len(tokenizer)
        if isinstance(preds, np.ndarray) and preds.ndim == 3:
            preds = preds.argmax(-1)
        preds = np.clip(preds, 0, vocab_size - 1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = np.clip(labels, 0, vocab_size - 1)

        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_refs = [r.strip() for r in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        empty = {"bleu": 0.0, "chrf": 0.0, "meteor": 0.0}
        if include_wer:
            empty["wer"] = 100.0
        if not any(decoded_preds) or not any(decoded_refs):
            return empty

        refs_nested = [[r] for r in decoded_refs]
        try:
            result = {
                "bleu": float(_metric("sacrebleu").compute(predictions=decoded_preds, references=refs_nested)["score"]),
                "chrf": float(_metric("chrf").compute(predictions=decoded_preds, references=refs_nested, word_order=2)["score"]),
                "meteor": float(_metric("meteor").compute(predictions=decoded_preds, references=refs_nested)["meteor"]),
            }
            if include_wer:
                result["wer"] = float(_metric("wer").compute(predictions=decoded_preds, references=decoded_refs) * 100)
            return result
        except Exception as e:  # keep training alive if a metric fails on odd output
            logger.warning(f"Metric computation failed: {e}")
            return empty

    return compute_metrics


def score_corpus(predictions, references):
    """Thesis test protocol (Table 5.1): BLEU, chrF and METEOR x 100, plus chrF++ and WER."""
    refs_nested = [[r] for r in references]
    preds = [p if p else "" for p in predictions]
    return {
        "bleu": float(_metric("sacrebleu").compute(predictions=preds, references=refs_nested)["score"]),
        "chrf": float(_metric("chrf").compute(predictions=preds, references=refs_nested)["score"]),
        "meteor": float(_metric("meteor").compute(predictions=preds, references=references)["meteor"] * 100),
        "chrf_plus_plus": float(_metric("chrf").compute(predictions=preds, references=refs_nested, word_order=2)["score"]),
        "wer": float(_metric("wer").compute(predictions=preds, references=references) * 100),
        "n_sentences": len(preds),
    }
