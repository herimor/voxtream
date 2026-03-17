# The following code is adapted from: https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/text/cleaners.py
import re

import regex

from voxtream.utils.text.number_norm import normalize_numbers
from voxtream.utils.text.time_norm import expand_time_english

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations in english:

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

abbreviations_en = [
    (
        re.compile(r"\b" + re.escape(key.rstrip(".")) + r"\.?(?=\W|$)", re.IGNORECASE),
        replacement,
    )
    for key, replacement in pairs
]

# Symbols to REPLACE (with specific substitutes)
punct_replacements = {
    # comma-like
    "...": ", ",
    "…": ", ",
    ":": ",",
    " - ": ", ",
    ";": ", ",
    " ,": ",",
    "、": ",",
    "，": ",",
    # full stop-like
    "..": ". ",
    " .": ".",
    "？": "?",
    "！": "!",
    "。": ".",
    # dashes to space
    "—": " ",
    "-": " ",
    "ー": " ",
    "一": " ",
    "–": " ",
    "‘": "'",
    "’": "'",
}

# Symbols to REMOVE entirely
punct_removal = ["“", "”", "«", "»", '"', "#", "*", "_", "{", "}", "¡", "¿", "|", "~"]
# TODO: keep dashes -> hyphen, replace underscores with space, keep quotation marks

# Build a single regex for replacements
replace_pattern = re.compile("|".join(map(re.escape, punct_replacements.keys())))
remove_pattern = re.compile("[" + re.escape("".join(punct_removal)) + "]")


def clean_punct(text: str) -> str:
    # Replace mapped symbols
    text = replace_pattern.sub(lambda m: punct_replacements[m.group(0)], text)
    # Remove unwanted symbols
    text = remove_pattern.sub("", text)

    # Remove a comma if it follows .,!? or another comma
    text = re.sub(r"([.,!?]),+", r"\1", text)
    return text


def expand_abbreviations(text: str, lang: str = "en") -> str:
    if lang == "en":
        _abbreviations = abbreviations_en
    else:
        raise NotImplementedError(
            f"Abbreviation expansion not implemented for language: {lang}"
        )
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


def remove_aux_symbols(text: str) -> str:
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    return text


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


def punct_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    text = clean_punct(text)

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


def replace_symbols(text: str, lang: str = "en") -> str:
    """Replace symbols based on the lenguage tag.

    Args:
      text:
       Input text.
      lang:
        Lenguage identifier. ex: "en", "fr", "pt", "ca".

    Returns:
      The modified text
      example:
        input args:
            text: "si l'avi cau, diguem-ho"
            lang: "ca"
        Output:
            text: "si lavi cau, diguemho"
    """
    if lang == "en":
        text = text.replace("&", " and ")
        text = text.replace("%", " percent ")
        text = text.replace("+", " plus ")
        text = text.replace("=", " equals ")
        text = text.replace("$", " dollar ")
        text = text.replace("/", " slash ")
        text = text.replace("@", " at ")
    elif lang == "fr":
        text = text.replace("&", " et ")
    elif lang == "pt":
        text = text.replace("&", " e ")
    elif lang == "ca":
        text = text.replace("&", " i ")
        text = text.replace("'", "")
        text = text.replace("-", " ")
    return text


def english_normalizer(text: str) -> str:
    """Pipeline for English text, including number and abbreviation expansion."""
    text = remove_emojis(text)
    text = punct_norm(text)
    # text = lowercase(text)
    text = expand_time_english(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = lower_internal_capitals(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)

    return text
