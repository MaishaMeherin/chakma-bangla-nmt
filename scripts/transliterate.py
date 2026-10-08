"""Transliterate Chakma script (𑄌𑄋𑄴𑄟𑄳𑄦) into Bangla script, for adding new data.

    pip install aksharamukha
    python scripts/transliterate.py --input new_pairs.csv --output new_pairs_translit.csv

Reads the `ccp` column and writes a `chakma_transliterated` column, using Aksharamukha
(thesis Section 4.3.1). Checked against the committed data with aksharamukha 2.3: the dev and
test sets are reproduced exactly; ~8% of training rows differ, mostly because য় is stored as
two code points (য + ়) in the data and a few rows were corrected by hand. So keep the
committed CSVs as the reference and use this only for new text.
Aksharamukha needs Python <= 3.13.
"""

import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--column", default="ccp")
    a = p.parse_args()

    from aksharamukha import transliterate

    df = pd.read_csv(a.input, encoding="utf-8")
    df["chakma_transliterated"] = [
        transliterate.process("Chakma", "Bengali", s) if isinstance(s, str) else s for s in df[a.column]
    ]
    df.to_csv(a.output, index=False)
    print(f"Wrote {len(df):,} rows to {a.output}")


if __name__ == "__main__":
    main()
