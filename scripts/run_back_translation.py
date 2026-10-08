"""Iterative back-translation for Chakma -> Bangla (thesis Chapter 4.4.7 / Chapter 5).

    python scripts/run_back_translation.py --config configs/back_translation.yaml
    python scripts/run_back_translation.py --config configs/back_translation.yaml --iterations 1
    python scripts/run_back_translation.py --config configs/back_translation.yaml --smoke   # CPU check

Every step writes its result to <output_dir>/iter<k>/ and is skipped when that result already
exists, so re-running the same command resumes an interrupted run. Use --force STEP to redo a step.

Output layout (thesis names; the archived notebook used shifted names, see notebooks/README.md):
    base_eval/                      baseline scored with the same test protocol
    iter1/synthetic_ccp2bn.csv      step 1: monolingual Chakma + generated Bangla
    iter1/backward/                 step 2: Bangla -> Chakma model
    iter1/synthetic_bn2ccp.csv      step 3: monolingual Bangla + generated Chakma
    iter1/forward/                  step 4: Chakma -> Bangla model (the deployable one)
    iter1/eval/                     step 5: test predictions + metrics
    iter2/...                       same, starting from iter1/forward
    summary.json                    test scores of base and every iteration
"""

import argparse
import gc
import json
from pathlib import Path

import _setup  # noqa: F401
import pandas as pd
import torch

from chakma_nmt import smoke
from chakma_nmt.data import SRC, TGT, load_bangla_monolingual, load_chakma_monolingual, load_splits
from chakma_nmt.filters import length_ratio_filter, strip_prefixes
from chakma_nmt.modeling import (PRETRAINED, build_trainer, load_model, load_tokenizer, save_model,
                                 tokenize_pairs, translate)
from chakma_nmt.utils import device, get_logger, load_config, save_json, set_seed
from evaluate_model import evaluate_model

STEPS = ("gen_bn", "train_backward", "gen_ccp", "train_forward", "evaluate")


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Pipeline:
    def __init__(self, cfg, is_smoke, force):
        self.cfg, self.smoke, self.force = cfg, is_smoke, set(force)
        self.out = Path(cfg["output_dir"])
        self.log = get_logger(self.out / "run.log")
        self.bt = cfg["back_translation"]
        self.m = cfg["model"]
        set_seed(cfg["seed"])
        cfg["train"]["seed"] = cfg["seed"]

        self.train_df, self.dev_df, self.test_df = load_splits(cfg["data"])
        all_sources = set(self.train_df[SRC]) | set(self.dev_df[SRC]) | set(self.test_df[SRC])
        self.chakma_mono = load_chakma_monolingual(cfg["data"]["chakma_mono"], all_sources)
        self.bangla_mono = load_bangla_monolingual(cfg["data"]["bangla_mono"])
        if is_smoke:
            self.train_df, self.dev_df, self.test_df = (smoke.head(self.train_df, "train"),
                                                        smoke.head(self.dev_df, "dev"), smoke.head(self.test_df, "test"))
            self.chakma_mono = smoke.head(self.chakma_mono, "chakma_mono")
            self.bangla_mono = smoke.head(self.bangla_mono, "bangla_mono")
        self.tokenizer = load_tokenizer(cfg["base_model"])

    def done(self, step, path):
        if step in self.force or not Path(path).exists():
            return False
        self.log.info(f"[skip] {step}: {path} already exists")
        return True

    # ---- generation steps -------------------------------------------------------------
    def generate(self, model_path, sources, direction, out_csv):
        model = load_model(model_path, self.tokenizer).to(device())
        gen = translate(model, self.tokenizer, sources, num_beams=self.bt["synthetic_num_beams"],
                        max_new_tokens=self.bt["synthetic_max_new_tokens"], max_source_length=self.m["max_source_length"],
                        batch_size=self.bt["synthetic_batch_size"], desc=f"Generating {direction}")
        del model
        free_gpu()
        n_stripped = 0
        if direction == "ccp2bn" and self.bt.get("strip_prefixes", True):
            gen, n_stripped = strip_prefixes(gen)
        src, gen, n_removed = length_ratio_filter(sources, gen, self.bt["length_ratio_threshold"])
        cols = {SRC: src, TGT: gen} if direction == "ccp2bn" else {SRC: gen, TGT: src}
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cols).to_csv(out_csv, index=False)
        self.log.info(f"{out_csv}: kept {len(src):,} of {len(sources):,} "
                      f"({n_removed:,} filtered, {n_stripped:,} prefixes stripped)")

    # ---- training steps ---------------------------------------------------------------
    def train(self, init_from, synthetic_csv, direction, epochs, out_model, set_special_tokens):
        """direction 'bn2ccp' = backward model, 'ccp2bn' = forward model. Gold pairs are repeated anchor_repeat times."""
        synthetic = pd.read_csv(synthetic_csv)[[SRC, TGT]]
        gold = pd.concat([self.train_df[[SRC, TGT]]] * self.bt["anchor_repeat"], ignore_index=True)
        mix = pd.concat([synthetic, gold], ignore_index=True).sample(frac=1, random_state=self.cfg["seed"])
        mix = mix.reset_index(drop=True)
        self.log.info(f"Training {direction} on {len(mix):,} pairs = {len(gold):,} gold "
                      f"({self.bt['anchor_repeat']}x) + {len(synthetic):,} synthetic; max {epochs} epochs")
        src_col, tgt_col = (SRC, TGT) if direction == "ccp2bn" else (TGT, SRC)
        tok = lambda df: tokenize_pairs(df, self.tokenizer, src_col, tgt_col, self.m["max_source_length"],
                                        self.m["max_target_length"])
        model = load_model(init_from, self.tokenizer, set_special_tokens=set_special_tokens, smoke=self.smoke)
        train_cfg = dict(self.cfg["train"], epochs=epochs)
        trainer = build_trainer(model, self.tokenizer, tok(mix), tok(self.dev_df), train_cfg,
                                Path(out_model).parent / f"{Path(out_model).name}_checkpoints",
                                self.cfg["val_generation"])
        trainer.train()
        save_model(trainer, self.tokenizer, out_model)
        del trainer, model
        free_gpu()

    def evaluate(self, model_path, out_dir):
        g = self.cfg["test_generation"]
        model = load_model(model_path, self.tokenizer).to(device())
        metrics = evaluate_model(model, self.tokenizer, self.test_df, out_dir, g["num_beams"], g["max_new_tokens"],
                                 g["batch_size"], self.m["max_source_length"], self.log)
        del model
        free_gpu()
        return metrics

    # ---- driver -------------------------------------------------------------------------
    def run(self, iterations):
        summary = {}
        base_eval = self.out / "base_eval"
        if not self.done("evaluate", base_eval / "test_metrics.json"):
            self.evaluate(self.cfg["base_model"], base_eval)
        summary["base"] = json.loads((base_eval / "test_metrics.json").read_text())

        forward = self.cfg["base_model"]
        epochs = self.cfg["train"]["epochs_per_iteration"]
        for k in range(1, iterations + 1):
            d = self.out / f"iter{k}"
            ep = epochs[min(k, len(epochs)) - 1]
            self.log.info(f"===== Iteration {k} (generator: {forward}) =====")
            if not self.done("gen_bn", d / "synthetic_ccp2bn.csv"):
                self.generate(forward, self.chakma_mono, "ccp2bn", d / "synthetic_ccp2bn.csv")
            if not self.done("train_backward", d / "backward"):
                # Backward model: fresh pretrained BanglaT5 every iteration.
                self.train(PRETRAINED, d / "synthetic_ccp2bn.csv", "bn2ccp", ep, d / "backward", False)
            if not self.done("gen_ccp", d / "synthetic_bn2ccp.csv"):
                self.generate(d / "backward", self.bangla_mono, "bn2ccp", d / "synthetic_bn2ccp.csv")
            if not self.done("train_forward", d / "forward"):
                # Forward model: restarts from the fine-tuned baseline every iteration.
                self.train(self.cfg["base_model"], d / "synthetic_bn2ccp.csv", "ccp2bn", ep, d / "forward", False)
            if not self.done("evaluate", d / "eval" / "test_metrics.json"):
                self.evaluate(d / "forward", d / "eval")
            summary[f"iter{k}_forward"] = json.loads((d / "eval" / "test_metrics.json").read_text())
            forward = str(d / "forward")

        save_json(summary, self.out / "summary.json")
        self.log.info("Test results (BLEU / chrF / METEOR):")
        for name, s in summary.items():
            self.log.info(f"  {name:15s} {s['bleu']:6.2f} {s['chrf']:6.2f} {s['meteor']:6.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/back_translation.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    p.add_argument("--iterations", type=int, default=None, help="default: num_iterations from the config")
    p.add_argument("--force", nargs="*", default=[], choices=STEPS, help="re-run these steps even if outputs exist")
    p.add_argument("--smoke", action="store_true", help="tiny model + tiny data on CPU (run train_base.py --smoke first)")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    if a.smoke:
        cfg = smoke.apply(cfg)
    if not Path(cfg["base_model"]).exists():
        raise SystemExit(f"Baseline model not found at {cfg['base_model']}. Train it first:\n"
                         f"  python scripts/train_base.py --config configs/base.yaml{' --smoke' if a.smoke else ''}")
    Pipeline(cfg, a.smoke, a.force).run(a.iterations or cfg["num_iterations"])


if __name__ == "__main__":
    main()
