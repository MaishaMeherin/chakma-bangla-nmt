"""Clean-up applied to synthetic (back-translated) pairs before they are used for training."""

# Task-prefix strings the model sometimes copied into its output (thesis Section 4.3.2,
# "Prefix Contamination Handling"). Removed from Chakma -> Bangla generations.
PREFIX_VARIANTS = (
    "Translate from Chakma to Bangla:",
    "অনুবাদ: চাকমা থেকে বাংলা:",
    "অনুবাদ করেছেন চাকমা থেকে বাংলা:",
    "অনুবাদ থেকে চাকমা থেকে বাংলা:",
    "অনুগ্রহপূর্বক চাকমা থেকে বাংলায়:",
)

# Gold data: mean bn/chakma length ratio 0.983 (sd 0.264), 99.9th percentile 2.33.
# Only 9 of 15,954 gold pairs exceed 2.5 (thesis Figure 4.2).
LENGTH_RATIO_THRESHOLD = 2.5


def strip_prefix(text):
    text = text.strip()
    for prefix in PREFIX_VARIANTS:
        if text.startswith(prefix):
            return text[len(prefix):].strip(), True
    return text, False


def strip_prefixes(texts):
    out, n_stripped = [], 0
    for t in texts:
        t, stripped = strip_prefix(t)
        out.append(t)
        n_stripped += stripped
    return out, n_stripped


def length_ratio_filter(sources, generated, max_ratio=LENGTH_RATIO_THRESHOLD):
    """Drop empty generations and generations longer than `max_ratio` x the source (whitespace tokens).

    `sources` are the real monolingual sentences, `generated` the model's translations.
    Returns (kept_sources, kept_generated, n_removed).
    """
    kept_src, kept_gen, removed = [], [], 0
    for src, gen in zip(sources, generated):
        src_len, gen_len = len(src.split()), len(gen.split())
        if gen_len < 1 or (max_ratio is not None and src_len > 0 and gen_len > max_ratio * src_len):
            removed += 1
            continue
        kept_src.append(src)
        kept_gen.append(gen)
    return kept_src, kept_gen, removed
