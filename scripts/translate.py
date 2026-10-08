"""Translate Chakma (written in Bangla script) into Bangla with a trained model.

    python scripts/translate.py --model outputs/back_translation/iter1/forward "মুই গম আগং"
    python scripts/translate.py --model outputs/base/final_model --input sentences.txt --output out.txt

Input sentences must be Chakma transliterated into Bangla script (see scripts/transliterate.py
if you have Chakma-script text).
"""

import argparse

import _setup  # noqa: F401

from chakma_nmt.data import normalize
from chakma_nmt.modeling import load_model, load_tokenizer, translate
from chakma_nmt.utils import device


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("sentences", nargs="*")
    p.add_argument("--input", help="text file, one sentence per line")
    p.add_argument("--output", help="write translations here (default: print)")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()

    sents = list(a.sentences)
    if a.input:
        with open(a.input, encoding="utf-8") as f:
            sents += [line.strip() for line in f if line.strip()]
    if not sents:
        raise SystemExit("Give sentences as arguments or with --input.")

    tokenizer = load_tokenizer(a.model)
    model = load_model(a.model, tokenizer).to(device())
    outs = translate(model, tokenizer, [normalize(s) for s in sents], num_beams=a.num_beams, batch_size=a.batch_size)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write("\n".join(outs) + "\n")
    else:
        for s, t in zip(sents, outs):
            print(f"{s}\n  -> {t}")


if __name__ == "__main__":
    main()
