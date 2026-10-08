"""Model/tokenizer loading, dataset tokenisation, Trainer construction and batched translation."""

import inspect
import logging
from pathlib import Path

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

from .data import SRC, TGT
from .metrics import build_compute_metrics

logger = logging.getLogger("chakma_nmt")

PRETRAINED = "csebuetnlp/banglat5"


def load_tokenizer(name_or_path=PRETRAINED):
    # The thesis used the slow SentencePiece tokenizer (use_fast=False) everywhere.
    return AutoTokenizer.from_pretrained(name_or_path, use_fast=False)


def tiny_model(tokenizer):
    """A 2-layer, randomly initialised T5 with the BanglaT5 vocabulary, for CPU smoke tests."""
    cfg = AutoConfig.from_pretrained(PRETRAINED)
    cfg.update(dict(d_model=64, d_ff=128, d_kv=16, num_layers=2, num_decoder_layers=2, num_heads=4))
    model = T5ForConditionalGeneration(cfg)
    model.resize_token_embeddings(len(tokenizer))
    return model


def load_model(name_or_path, tokenizer, set_special_tokens=False, smoke=False):
    """Load a seq2seq model.

    set_special_tokens=True reproduces the baseline notebook: pad/bos/decoder-start = pad id,
    eos = eos id, and the embeddings resized from 32,128 to the tokenizer's 32,100 rows.
    The back-translation notebook loaded models without this step, so it stays off there.
    """
    if smoke and name_or_path == PRETRAINED:
        model = tiny_model(tokenizer)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path)
    if set_special_tokens:
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
        model.config.pad_token_id = pad_id
        model.config.decoder_start_token_id = pad_id
        model.config.bos_token_id = pad_id
        model.config.eos_token_id = eos_id
        if model.config.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
            model.config.vocab_size = len(tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model from {name_or_path} ({n_params:,} parameters)")
    return model


def tokenize_pairs(df, tokenizer, source_col, target_col, max_source_length=128, max_target_length=128):
    """DataFrame -> tokenised HF Dataset. No padding here; the collator pads each batch."""
    ds = Dataset.from_dict({"src": df[source_col].tolist(), "tgt": df[target_col].tolist()})

    def _tok(batch):
        enc = tokenizer(batch["src"], max_length=max_source_length, truncation=True, padding=False)
        enc["labels"] = tokenizer(text_target=batch["tgt"], max_length=max_target_length, truncation=True, padding=False)["input_ids"]
        return enc

    return ds.map(_tok, batched=True, remove_columns=["src", "tgt"], desc="Tokenising")


def _precision_flags(precision):
    if precision == "fp32" or not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    if precision == "bf16":
        return {"fp16": False, "bf16": True}
    if precision == "fp16":
        return {"fp16": True, "bf16": False}
    bf16 = torch.cuda.is_bf16_supported()  # "auto": what the baseline notebook did
    return {"fp16": not bf16, "bf16": bf16}


def _compat_kwargs(cls, kwargs):
    """Keep working on newer transformers: evaluation_strategy -> eval_strategy, tokenizer -> processing_class."""
    params = inspect.signature(cls.__init__).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "tokenizer" in kwargs and "tokenizer" not in params:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    return kwargs


def build_trainer(model, tokenizer, train_ds, eval_ds, train_cfg, output_dir, val_generation, include_wer=False):
    """Seq2SeqTrainer with the thesis settings: epoch-wise eval/save, best checkpoint by validation BLEU."""
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
        **val_generation,
    )
    args = Seq2SeqTrainingArguments(**_compat_kwargs(Seq2SeqTrainingArguments, dict(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        learning_rate=train_cfg["lr"],
        lr_scheduler_type=train_cfg.get("scheduler", "linear"),
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        label_smoothing_factor=train_cfg["label_smoothing"],
        optim="adamw_torch",
        predict_with_generate=True,
        generation_config=generation_config,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=train_cfg.get("logging_steps", 50),
        logging_first_step=True,
        report_to=[],
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        **_precision_flags(train_cfg.get("precision", "fp32")),
    )))
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
        pad_to_multiple_of=8 if torch.cuda.is_available() and train_cfg.get("pad_to_multiple_of_8") else None,
    )
    callbacks = []
    if train_cfg.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"]))
    return Seq2SeqTrainer(**_compat_kwargs(Seq2SeqTrainer, dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer, include_wer=include_wer),
        callbacks=callbacks,
    )))


def save_model(trainer, tokenizer, path):
    path = Path(path)
    trainer.save_model(str(path))
    tokenizer.save_pretrained(str(path))
    logger.info(f"Saved model to {path}")


@torch.no_grad()
def translate(model, tokenizer, texts, num_beams=5, max_new_tokens=128, max_source_length=128,
              batch_size=16, early_stopping=True, desc="Translating"):
    """Beam-search translation. batch_size=1 is the per-sentence protocol used for the thesis test scores."""
    model.eval()
    dev = next(model.parameters()).device
    outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_source_length).to(dev)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping if num_beams > 1 else False,
        )
        outputs.extend(t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def gold_columns(direction):
    """(source column, target column) for a translation direction."""
    return (SRC, TGT) if direction == "ccp2bn" else (TGT, SRC)
