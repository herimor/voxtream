"""
Phoneme vocabulary for the five-stream phoneme input.

Every phoneme of the alignment is split into five tokens that the model embeds
separately and sums:

    root    the phoneme itself, stress and length stripped ("ˈɔː" -> "ɔ")
    prefix  the stress mark that was stripped ("ˈ", "ˌ")
    suffix  the length mark that was stripped ("ː")
    punct   the punctuation that follows the word this phoneme ends (". , ? !")
    eow     set on the last phoneme of every word

Streams that have nothing to say carry their own PAD token, so a plain
word-internal phoneme is `(root, PAD, PAD, PAD, PAD)`. Punctuation is no longer
inserted into the sequence and deleted from the transformer output: it rides on
the phoneme that precedes it, which keeps the phoneme indices of the alignment
and the token sequence one to one.

The root vocabulary is read from a frequency table of already normalized roots
(`valid_phoneme_vocab.json`, `{root: count}`); `normalize()` maps a raw espeak
phoneme onto it:

  1. a leading or trailing "." / "," is a tokenizer artifact and is dropped,
  2. a run of length marks collapses to one ("ːː" -> "ː"),
  3. a leading "ˈ" / "ˌ" becomes the prefix token,
  4. a trailing "ː" becomes the suffix token,
  5. a root that is its own repetition collapses ("ææ" -> "æ"), as long as the
     collapsed form is the one in the vocabulary,
  6. anything left out of the vocabulary becomes UNK with PAD prefix/suffix.

Note that a length mark inside the root is part of the root ("ɔːɹ" stays whole),
only a trailing one is a suffix.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Stream layout
# ---------------------------------------------------------------------------

NUM_STREAMS = 5
ROOT, PREFIX, SUFFIX, PUNCT, EOW = range(NUM_STREAMS)
STREAM_NAMES = ("root", "prefix", "suffix", "punct", "eow")

# root stream: specials first, then the phonemes of the vocabulary
ROOT_PAD = 0
ROOT_BOS = 1
ROOT_EOS = 2
ROOT_UNK = 3
ROOT_PROMPT = 4  # a frame of the acoustic prompt, whose text is masked
NUM_ROOT_SPECIAL = 5

# prefix stream
PREFIX_PAD = 0
STRESS_TOKENS = {"ˈ": 1, "ˌ": 2}  # primary, secondary
PREFIX_VOCAB_SIZE = 1 + len(STRESS_TOKENS)

# suffix stream
SUFFIX_PAD = 0
SUFFIX_LONG = 1
SUFFIX_VOCAB_SIZE = 2

# punctuation stream
PUNCT_PAD = 0
PUNCT_TOKENS = {".": 1, ",": 2, "?": 3, "!": 4}
PUNCT_VOCAB_SIZE = 1 + len(PUNCT_TOKENS)
# a word may carry several symbols ("?!", '."'); the strongest one wins
PUNCT_PRIORITY = {"?": 4, "!": 3, ".": 2, ",": 1}
TERMINAL_PUNCT = frozenset({".", "?", "!"})
TERMINAL_PUNCT_TOKENS = frozenset(PUNCT_TOKENS[s] for s in TERMINAL_PUNCT)

# end-of-word stream
EOW_PAD = 0
EOW_SET = 1
EOW_VOCAB_SIZE = 2

LENGTH_MARK = "ː"
_EDGE_ARTIFACTS = ".,"


@dataclass
class PhonemeVocab:
    """Root vocabulary plus the fixed attribute streams."""

    roots: List[str]
    root_to_id: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.root_to_id = {
            root: NUM_ROOT_SPECIAL + i for i, root in enumerate(self.roots)
        }
        # the stored streams are uint8, and so are the attribute offsets added to them
        if self.root_vocab_size > 255:
            raise ValueError(
                f"{self.root_vocab_size} roots do not fit the uint8 phoneme streams"
            )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_counts(cls, path: Path) -> "PhonemeVocab":
        """Build from `{root: count}`, most frequent root first."""
        with open(path) as f:
            counts = json.load(f)
        roots = [r for r, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
        return cls(roots=roots)

    @classmethod
    def load(cls, path: Path) -> "PhonemeVocab":
        with open(path) as f:
            data = json.load(f)
        return cls(roots=list(data["roots"]))

    def save(self, path: Path) -> None:
        """Write the vocabulary the dataset was built with, for inference."""
        payload = {
            "roots": self.roots,
            "vocab_sizes": list(self.vocab_sizes),
            "specials": {
                "pad": ROOT_PAD,
                "bos": ROOT_BOS,
                "eos": ROOT_EOS,
                "unk": ROOT_UNK,
                "prompt": ROOT_PROMPT,
            },
            "stress": STRESS_TOKENS,
            "punct": PUNCT_TOKENS,
        }
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # -- sizes --------------------------------------------------------------

    @property
    def root_vocab_size(self) -> int:
        return NUM_ROOT_SPECIAL + len(self.roots)

    @property
    def vocab_sizes(self) -> Tuple[int, int, int, int, int]:
        return (
            self.root_vocab_size,
            PREFIX_VOCAB_SIZE,
            SUFFIX_VOCAB_SIZE,
            PUNCT_VOCAB_SIZE,
            EOW_VOCAB_SIZE,
        )

    # -- encoding -----------------------------------------------------------

    def normalize(self, phoneme: str) -> Tuple[int, int, int]:
        """Map a raw espeak phoneme onto `(root, prefix, suffix)` token ids."""
        ph = phoneme.strip().strip(_EDGE_ARTIFACTS)

        # "ːː" and longer runs mean the same as a single mark
        while LENGTH_MARK * 2 in ph:
            ph = ph.replace(LENGTH_MARK * 2, LENGTH_MARK)

        prefix = PREFIX_PAD
        if ph[:1] in STRESS_TOKENS:
            prefix = STRESS_TOKENS[ph[0]]
            ph = ph[1:]

        suffix = SUFFIX_PAD
        if ph.endswith(LENGTH_MARK):
            suffix = SUFFIX_LONG
            ph = ph[: -len(LENGTH_MARK)]

        # "ææ" is a doubled "æ"; prefer whichever form the vocabulary holds
        for candidate in (_collapse_repetition(ph), ph):
            root = self.root_to_id.get(candidate)
            if root is not None:
                return root, prefix, suffix

        return ROOT_UNK, PREFIX_PAD, SUFFIX_PAD

    def special(self, root: int) -> Tuple[int, int, int, int, int]:
        """A five-stream token for BOS/EOS/PROMPT, PAD everywhere else."""
        return (root, PREFIX_PAD, SUFFIX_PAD, PUNCT_PAD, EOW_PAD)

    def prompt_tokens(self, num_rows: int) -> np.ndarray:
        """
        `[num_rows, 5]` tokens of the masked prompt: `[BOS]`, then one `[PROMPT]` per
        prompt frame, the sequence the crop of a training sample starts with.
        """
        rows = np.zeros((num_rows, NUM_STREAMS), dtype=np.int64)
        rows[:, ROOT] = ROOT_PROMPT
        rows[0, ROOT] = ROOT_BOS
        return rows


def _collapse_repetition(root: str) -> str:
    """ "aa" -> "a", "aaaa" -> "a"; anything that is not a repetition is kept."""
    while len(root) >= 2 and len(root) % 2 == 0:
        half = len(root) // 2
        if root[:half] != root[half:]:
            break
        root = root[:half]
    return root


def punct_token(symbols: str) -> int:
    """The strongest punctuation token of a run of symbols, PAD if there is none."""
    best, best_rank = PUNCT_PAD, 0
    for symbol in symbols:
        rank = PUNCT_PRIORITY.get(symbol, 0)
        if rank > best_rank:
            best, best_rank = PUNCT_TOKENS[symbol], rank
    return best
