from chakma_nmt.filters import length_ratio_filter, strip_prefix, strip_prefixes


def test_strip_prefix_removes_known_variants():
    assert strip_prefix("অনুবাদ: চাকমা থেকে বাংলা: আমি ভালো আছি") == ("আমি ভালো আছি", True)
    assert strip_prefix("Translate from Chakma to Bangla: hello") == ("hello", True)
    assert strip_prefix("  আমি ভালো আছি ") == ("আমি ভালো আছি", False)


def test_strip_prefixes_counts():
    out, n = strip_prefixes(["অনুবাদ করেছেন চাকমা থেকে বাংলা: ক", "খ"])
    assert out == ["ক", "খ"] and n == 1


def test_length_ratio_filter():
    src = ["a b", "a b", "a b", "a"]
    gen = ["x y z w y", "x y z w y q", "", "x"]   # ratios 2.5 (kept), 3.0, empty, 1.0
    kept_src, kept_gen, removed = length_ratio_filter(src, gen, 2.5)
    assert kept_gen == ["x y z w y", "x"]
    assert removed == 2
