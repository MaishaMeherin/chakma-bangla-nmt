"""Fine-tune BanglaT5 for Chakma -> Bangla on the gold parallel data (the thesis baseline).

    python scripts/train_base.py --config configs/base.yaml
    python scripts/train_base.py --config configs/base.yaml --smoke          # CPU check, ~2 min
    python scripts/train_base.py --config configs/base.yaml --set train.epochs=5

Outputs in <output_dir>/:
    final_model/        best checkpoint by validation BLEU (+ tokenizer)
    results.json        validation and test scores
    eval/               out-of-domain test predictions and metrics (thesis protocol)
    train.log
"""

import argparse
from pathlib import Path

import _setup  # noqa: F401

from chakma_nmt import smoke
from chakma_nmt.data import SRC, TGT, load_ood_test, load_splits
from chakma_nmt.modeling import build_trainer, load_model, load_tokenizer, save_model, tokenize_pairs
from chakma_nmt.utils import get_logger, load_config, save_json, set_seed
from evaluate_model import evaluate_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    p.add_argument("--smoke", action="store_true", help="tiny model + tiny data on CPU")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    if a.smoke:
        cfg = smoke.apply(cfg)
    out = Path(cfg["output_dir"])
    logger = get_logger(out / "train.log")
    logger.info(f"Config: {a.config}{' (smoke)' if a.smoke else ''} -> {out}")
    set_seed(cfg["seed"], deterministic=cfg.get("deterministic", False))
    cfg["train"]["seed"] = cfg["seed"]

    train_df, dev_df, test_df = load_splits(cfg["data"])
    # With split=external_dev the test split already is the out-of-domain benchmark.
    ood_df = test_df if cfg["data"].get("split") == "external_dev" else load_ood_test(cfg["data"])
    if a.smoke:
        train_df, dev_df, test_df, ood_df = (smoke.head(train_df, "train"), smoke.head(dev_df, "dev"),
                                             smoke.head(test_df, "test"), smoke.head(ood_df, "test"))

    m = cfg["model"]
    tokenizer = load_tokenizer(m["name"])
    model = load_model(m["name"], tokenizer, set_special_tokens=True, smoke=a.smoke)
    tok = lambda df: tokenize_pairs(df, tokenizer, SRC, TGT, m["max_source_length"], m["max_target_length"])
    train_ds, dev_ds, test_ds = tok(train_df), tok(dev_df), tok(test_df)

    trainer = build_trainer(model, tokenizer, train_ds, dev_ds, cfg["train"], out / "checkpoints",
                            cfg["val_generation"], include_wer=True)
    train_result = trainer.train()
    save_model(trainer, tokenizer, out / "final_model")

    results = {"config": a.config, "smoke": a.smoke, "train_pairs": len(train_df), "dev_pairs": len(dev_df),
               "train_loss": train_result.metrics.get("train_loss")}
    # Validation (and in-domain test) through the Trainer, as in thesis Chapter 4.
    results["validation"] = trainer.evaluate(eval_dataset=dev_ds, metric_key_prefix="val")
    if cfg["data"].get("split") == "indomain_80_10_10":
        results["test_in_domain"] = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    results["test_out_of_domain_trainer"] = trainer.evaluate(eval_dataset=tok(ood_df), metric_key_prefix="test")

    # Out-of-domain test with the per-sentence protocol used for every Chapter 5 number.
    g = cfg["test_generation"]
    results["test_out_of_domain"] = evaluate_model(
        trainer.model, tokenizer, ood_df, out / "eval", g["num_beams"], g["max_new_tokens"], g["batch_size"],
        m["max_source_length"], logger)
    save_json(results, out / "results.json")
    logger.info(f"Done. Results in {out / 'results.json'}")


if __name__ == "__main__":
    main()
