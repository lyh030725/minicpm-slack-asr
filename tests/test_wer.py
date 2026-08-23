from minicpm_slack_asr.wer import aggregate_counts, compute_wer, normalize_for_wer


def test_normalize_for_wer_openbmb_contraction():
    assert normalize_for_wer("Hello, WORLD! Don't stop.") == "hello world do not stop"


def test_openbmb_number_normalization():
    assert normalize_for_wer("twenty") == normalize_for_wer("20")
    assert compute_wer("TWENTY", "20").errors == 0


def test_openbmb_spelling_normalization():
    assert compute_wer("COLOUR", "color").errors == 0


def test_openbmb_ignored_fillers():
    assert compute_wer("HELLO", "uh hello").errors == 0


def test_compute_wer_exact():
    counts = compute_wer("HELLO WORLD", "hello world")
    assert counts.errors == 0
    assert counts.wer == 0.0


def test_compute_wer_operations():
    counts = compute_wer("A B C", "A X C D")
    assert counts.reference_words == 3
    assert counts.errors == 2
    assert counts.substitutions == 1
    assert counts.insertions == 1


def test_aggregate_counts():
    a = compute_wer("A B", "A")
    b = compute_wer("C", "D")
    total = aggregate_counts([a, b])
    assert total.reference_words == 3
    assert total.errors == 2
