from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class WERCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.reference_words


def normalize_for_wer(text: str) -> str:
    """Light LibriSpeech-style normalization for model-output WER.

    LibriSpeech references are uppercase and mostly punctuation free. We lowercase,
    Unicode-normalize, keep apostrophes inside words, replace other punctuation with
    spaces, and collapse whitespace.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> WERCounts:
    ref = normalize_for_wer(reference).split()
    hyp = normalize_for_wer(hypothesis).split()

    # dp[i][j] = (total errors, substitutions, deletions, insertions)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(len(hyp) + 1)] for _ in range(len(ref) + 1)
    ]
    for i in range(1, len(ref) + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue

            sub_prev = dp[i - 1][j - 1]
            del_prev = dp[i - 1][j]
            ins_prev = dp[i][j - 1]
            candidates = [
                (sub_prev[0] + 1, sub_prev[1] + 1, sub_prev[2], sub_prev[3]),
                (del_prev[0] + 1, del_prev[1], del_prev[2] + 1, del_prev[3]),
                (ins_prev[0] + 1, ins_prev[1], ins_prev[2], ins_prev[3] + 1),
            ]
            dp[i][j] = min(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))

    _, substitutions, deletions, insertions = dp[-1][-1]
    return WERCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(ref),
    )


def aggregate_counts(items: list[WERCounts]) -> WERCounts:
    return WERCounts(
        substitutions=sum(x.substitutions for x in items),
        deletions=sum(x.deletions for x in items),
        insertions=sum(x.insertions for x in items),
        reference_words=sum(x.reference_words for x in items),
    )
