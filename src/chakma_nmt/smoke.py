"""--smoke mode: run any pipeline end to end on CPU in a few minutes.

Uses a tiny randomly initialised T5 (BanglaT5 vocabulary), a few dozen sentences and one
epoch, and writes everything under outputs/smoke/. Scores are meaningless; the point is to
check that data loading, training, generation, filtering and evaluation all run.
"""

from pathlib import Path

LIMITS = {"train": 64, "dev": 16, "test": 16, "chakma_mono": 32, "bangla_mono": 32}


def apply(cfg):
    cfg = dict(cfg)
    cfg["output_dir"] = str(Path("outputs/smoke") / Path(cfg["output_dir"]).name)
    if "base_model" in cfg:
        cfg["base_model"] = "outputs/smoke/base/final_model"
    train = dict(cfg["train"])
    train.update(epochs=1, batch_size=8, grad_accum=1, warmup_steps=0, precision="fp32", logging_steps=5)
    if "epochs_per_iteration" in train:
        train["epochs_per_iteration"] = [1] * len(train["epochs_per_iteration"])
    cfg["train"] = train
    cfg["deterministic"] = False
    for key in ("val_generation", "test_generation"):
        gen = dict(cfg[key])
        gen["num_beams"] = 2
        gen.pop("max_length", None)
        gen["max_new_tokens"] = 16
        if key == "val_generation":
            gen.pop("early_stopping", None)
        cfg[key] = gen
    if "back_translation" in cfg:
        bt = dict(cfg["back_translation"])
        bt.update(synthetic_num_beams=2, synthetic_max_new_tokens=16)
        cfg["back_translation"] = bt
    return cfg


def head(df_or_list, key):
    return df_or_list[: LIMITS[key]] if isinstance(df_or_list, list) else df_or_list.head(LIMITS[key]).reset_index(drop=True)
