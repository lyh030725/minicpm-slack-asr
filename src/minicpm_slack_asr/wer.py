from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


OPENBMB_ULTRAEVAL_COMMIT = "bbc07b1effc03a85006c36dc765b8f2b8eae8d36"


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


@lru_cache(maxsize=1)
def _openbmb_normalizers() -> tuple[Any, Any]:
    """Load the pinned OpenBMB UltraEval-Audio English ASR scorer components.

    The source package itself is installed by ``scripts/setup_runpod.sh`` from
    ``OpenBMB/UltraEval-Audio`` at ``OPENBMB_ULTRAEVAL_COMMIT`` with ``--no-deps``.
    Its small runtime dependencies are pinned in ``requirements.txt``.
    """
    try:
        from audio_evals.lib.evaluate_tokenizer import EvaluationTokenizer
        from audio_evals.lib.text_normalization.en import EnglishTextNormalizer
    except ImportError as exc:  # pragma: no cover - setup error path
        raise RuntimeError(
            "OpenBMB UltraEval-Audio scorer is not installed. Run "
            "`bash scripts/setup_runpod.sh` or install the pinned scorer source "
            f"at commit {OPENBMB_ULTRAEVAL_COMMIT}."
        ) from exc

    normalizer = EnglishTextNormalizer()
    tokenizer = EvaluationTokenizer(
        tokenizer_type="13a",
        lowercase=True,
        punctuation_removal=False,
        character_tokenization=False,
    )
    return normalizer, tokenizer


def normalize_for_wer(text: str) -> str:
    """Normalize/tokenize exactly like OpenBMB UltraEval-Audio ``PracticeWER``.

    Pipeline pinned for this project:
      lowercase -> EnglishTextNormalizer -> sacreBLEU 13a EvaluationTokenizer

    The returned string is already whitespace-tokenized for WER scoring.
    """
    normalizer, tokenizer = _openbmb_normalizers()
    normalized = normalizer(str(text).lower())
    return tokenizer.tokenize(normalized).strip()


def compute_wer(reference: str, hypothesis: str) -> WERCounts:
    """Compute word error counts using the OpenBMB-normalized token sequences.

    OpenBMB UltraEval-Audio computes corpus WER from edit distance after its
    English normalization and 13a tokenization. We use the same token sequences
    and a Levenshtein DP that additionally exposes S/D/I counts. The sum of
    S+D+I is the same minimum edit distance used for WER.
    """
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
