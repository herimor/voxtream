# The following code is adapted from: https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/text/cleaners.py
import re

import regex

from voxtream.utils.text.number_norm import normalize_numbers
from voxtream.utils.text.time_norm import AM_PM, expand_time_english

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# (abbreviation, expansion) pairs; a trailing dot is optional
pairs = [
    ("mrs", "misess"),
    ("mr", "mister"),
    ("dr", "doctor"),
    ("st", "saint"),
    # ("co", "company"),
    ("jr", "junior"),
    ("maj", "major"),
    ("gen", "general"),
    ("drs", "doctors"),
    ("rev", "reverend"),
    ("lt", "lieutenant"),
    ("hon", "honorable"),
    ("sgt", "sergeant"),
    ("capt", "captain"),
    ("esq", "esquire"),
    ("ltd", "limited"),
    ("col", "colonel"),
    ("ft", "fort"),
    ("i.e.", "that is"),
    ("e.g.", "for example"),
    ("hrs", "hours"),
    ("mins", "minutes"),
    ("ms", "milliseconds"),
]

_abbreviations = [
    (
        re.compile(r"\b" + re.escape(key.rstrip(".")) + r"\.?(?=\W|$)", re.IGNORECASE),
        replacement,
    )
    for key, replacement in pairs
]

# Target punctuation alphabet (by training-data frequency): . , ? ! ' -
# Everything else is folded onto it, expanded, or stripped.

# Single-character folds: applied in one pass via str.translate, so they are
# order-independent (unlike the old single |-joined regex, where " ." could
# fire before "...").
_char_map = str.maketrans(
    {
        # CJK / fullwidth / Arabic sentence punctuation -> ASCII
        "。": ".",
        "？": "?",
        "！": "!",
        "、": ",",
        "，": ",",
        "،": ",",
        "：": ":",
        "；": ";",
        # ellipsis characters -> ASCII dots (unified with the dot-run rule)
        "…": "...",
        "‥": "..",
        # apostrophe lookalikes -> ASCII
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        # dash variants fold to the canonical hyphen-minus; the dash itself is
        # a kept token. "一" (U+4E00) is the CJK word "one", NOT a dash.
        "—": "-",
        "–": "-",
        "―": "-",
        "‒": "-",
        "‑": "-",
        "−": "-",
        "־": "-",
        # primes (feet/inches vs arc-minutes/seconds is ambiguous, so don't
        # expand; a space keeps 5′10″ from collapsing into "510")
        "′": " ",
        "″": " ",
        # backslash is a separator, like underscore
        "\\": " ",
        # katakana prolonged sound mark: silent in an English normalizer
        "ー": None,
        # underscore is a word separator (snake_case -> snake case)
        "_": " ",
    }
)

# Symbols removed entirely (silent in speech). Single consolidated pass,
# subsuming the old punct_removal list and remove_aux_symbols().
_remove_re = re.compile(r'["“”„«»《》⟨⟩()\[\]{}<>*#•†‡‖§·・`¡¿|~]')

# A clock time, "4" or "3:30"
_CLOCK = r"\d{1,2}(?::\d{2})?"

# Contextual rules, applied as ordered, documented regex passes.
_punct_rules = [
    # Any run of 2+ dots is a pause: "Wait ..." / "So.....yeah" -> ", "
    (re.compile(r"\.{2,}"), ", "),
    # Colon -> comma, EXCEPT between digits: "3:30" must survive for
    # expand_time_english (leftovers are swept after time expansion).
    (re.compile(r"(?<!\d):(?!\d)"), ","),
    # Semicolon is a pause
    (re.compile(r";"), ","),
    # A run of dashes is always a pause: "a--b" -> "a - b"
    (re.compile(r"\s*-{2,}\s*"), " - "),
    # A dash in a time range is "to": "3-4pm" / "9am - 5pm" / "3:30-4:00"
    (
        re.compile(
            rf"(?<![\d:])({_CLOCK}(?:\s*{AM_PM}\.?)?)\s*-\s*(?={_CLOCK}\s*{AM_PM})"
        ),
        r"\1 to ",
    ),
    (
        re.compile(r"(?<![\d:])(\d{1,2}:\d{2})\s*-\s*(?=\d{1,2}:\d{2}(?!\d))"),
        r"\1 to ",
    ),
    # A dash right before a stop is dropped, so the stop survives: "Wait -."
    (re.compile(r"\s*-\s*(?=[.!?](?!\d))"), ""),
]

# Text edges, applied only to a whole text (see english_normalizer): a closing
# dash is dropped before the pause rules, an opening pause after them.
_closing_dash_re = re.compile(r"\s*-+\s*$")
_opening_pause_re = re.compile(r"^\s*,\s*")

_pause_rules = [
    # A dash not joining word characters is a pause, read as a comma (the model
    # has no dash token); converted here so the rules below tidy it. Intra-word
    # hyphens ("well-known", "3-4") and the sign of a number ("-4", but not a
    # time like "-4am" / "-3:30") are kept.
    (
        re.compile(
            r"(?<!\w)-(?!\.?\d)|(?<!\w)-(?=\d{1,2}(?::\d|\s*[aApP]\.?[mM](?!\w)))|-(?!\w|\.\d)"
        ),
        ", ",
    ),
    # Collapse repeated terminals: "!!!" -> "!", "?!" -> "?"
    (re.compile(r"([!?])[!?]+"), r"\1"),
    # No whitespace before pause/stop punctuation (a decimal point isn't one:
    # "costs .5")
    (re.compile(r"\s+([,!?]|\.(?!\d))"), r"\1"),
    # Dedupe stacked pause punctuation, even across spaces: first mark wins
    # ("Hello. , Hi" -> "Hello. Hi")
    (re.compile(r"([.,!?])(?:\s*(?:,|\.(?!\d)))+"), r"\1"),
]


def clean_punct(text: str, strip_edges: bool = True) -> str:
    text = text.translate(_char_map)
    text = _remove_re.sub("", text)
    for pattern, repl in _punct_rules:
        text = pattern.sub(repl, text)
    if strip_edges:
        # Text can't close with a dash: "Wait -" / "-Hello-"
        text = _closing_dash_re.sub("", text)
    for pattern, repl in _pause_rules:
        text = pattern.sub(repl, text)
    if strip_edges:
        # Text can't open with a pause: "-4am" / "- Hello" / "...Hello"
        text = _opening_pause_re.sub("", text)
    return text


def expand_abbreviations(text: str) -> str:
    for pattern, replacement in _abbreviations:
        text = re.sub(pattern, replacement, text)
    return text


def lowercase(text: str) -> str:
    return text.lower()


def remove_emojis(text: str) -> str:
    # Remove Emoji presentation chars
    text = regex.sub(r"\p{Emoji_Presentation}", "", text)
    # Remove extra variation selectors & modifiers (skin tone, gender, etc.)
    text = regex.sub(r"[\uFE0F\u200D]", "", text)
    return text


def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, " ", text).strip()


def _normalize_internal_caps_word(match: re.Match[str]) -> str:
    """Lowercase internal capitals in a word while keeping all-uppercase words unchanged."""
    word = match.group(0)
    if any(ch.isalpha() for ch in word) and word.isupper():
        return word
    if len(word) <= 1:
        return word
    first = word[0]
    rest = "".join(ch.lower() if ch.isalpha() else ch for ch in word[1:])
    return f"{first}{rest}"


def lower_internal_capitals(text: str) -> str:
    """Normalize words with mixed case (e.g., VoXtream2 -> Voxtream2)."""
    return re.sub(r"\b[A-Za-z][A-Za-z0-9]*\b", _normalize_internal_caps_word, text)


def punct_norm(text: str, strip_edges: bool = True) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    text = clean_punct(text, strip_edges=strip_edges)

    # Expand single letter abbreviations like "A. B. C." to "A B C"
    text = re.sub(r"(?:(?<=\s)|^)([A-Z])\.\s*", r"\1 ", text)

    # remove apostrophes that are not part of a contraction
    text = re.sub(r"(?<![A-Za-z])[']|['](?![A-Za-z])", "", text)

    def dot_replacer(match):
        left = match.group(1)
        right = match.group(2)
        return f"{left} {right}"  # replace the dot with a space

    # Remove dots between multiple letters (e.g., sci.math -> sci math, np.array -> np array)
    pattern = r"([A-Za-z]{2,})\.([A-Za-z]{2,})"
    text = re.sub(pattern, dot_replacer, text)

    return text


def replace_symbols(text: str) -> str:
    """Read symbols out as English words: "&" -> "and", "%" -> "percent"."""
    text = text.replace("&", " and ")
    text = text.replace("%", " percent ")
    text = text.replace("+", " plus ")
    text = text.replace("=", " equals ")
    text = text.replace("$", " dollar ")
    text = text.replace("/", " slash ")
    text = text.replace("@", " at ")
    return text


def english_normalizer(text: str, strip_edges: bool = True) -> str:
    """Pipeline for English text, including number and abbreviation expansion.

    `strip_edges` drops a pause opening the text and a dash closing it. Pass
    False for a chunk of a text stream, where a pause at the chunk edge is a
    real pause between chunks ("Wait" / "-" / "what").
    """
    text = remove_emojis(text)
    text = punct_norm(text, strip_edges=strip_edges)
    # text = lowercase(text)
    text = expand_time_english(text)

    # Colons between digits were kept for time expansion; anything left
    # (scores like "2:1") is a pause. The space keeps "2, 1" from being
    # read as a comma-grouped number by normalize_numbers.
    text = text.replace(":", ", ")

    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = lower_internal_capitals(text)
    text = replace_symbols(text)
    text = collapse_whitespace(text)

    return text
