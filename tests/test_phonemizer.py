import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxtream.utils.dataset.phoneme_vocab import PUNCT_TOKENS
from voxtream.utils.text.normalizer import english_normalizer
from voxtream.utils.text.phonemizer import ESpeak

pytestmark = pytest.mark.skipif(
    not ESpeak.is_available(), reason="espeak / espeak-ng not installed"
)

# Punctuation the model has tokens for; anything else must not reach it.
PUNCT_MAP = PUNCT_TOKENS
PUNCT_LIKE = re.compile(r"[.,!?;:\-—–…\"'()\[\]]")

# Raw text -> normalizer -> espeak. Phonemes differ across espeak versions, so
# only the structure is checked: one phoneme word per text word, punctuation
# glued to the end of the previous word, and only model punctuation.
SENTENCES = [
    "Well-known - said George.",
    "Well-known -said George.",
    "He said -hello- to me.",
    "He came — finally — at noon.",
    "Hello. -Hi there-. Bye",
    "Hello.-Hi",
    "Hi, - there.",
    "Wait - .",
    "Really -?",
    "So - um - yes",
    "-Hello",
    "- Hello there.",
    "-4 degrees.",
    "It was -4 outside.",
    "3 -4",
    "-3:30 pm",
    "COVID-19 cases",
    "x-ray, a well-known fact",
    "a 3-year-old",
    "3-4 pages",
    "Wait... what?",
    "Hi!!! Really?!",
    'She said: "yes" (maybe).',
    "a well-known fact: it works; mostly.",
    "Meet at 10:15 pm, ok?",
    "Count 1, 2, 3 and go.",
    "9am-5pm",
    "At 3:00 p.m. sharp",
    "At 4 P.M. Then we left.",
    "May 5th, 2021",
    "costs .5 now",
    "Wait -",
]

# Stream chunks keep the pauses at their edges (strip_edges=False).
STREAM_CHUNKS = [
    "-",
    "...",
    ",",
    "- what",
    ", world, ok",
    "—finally",
    "-Hello-",
    "Wait -",
]


BACKENDS = ["cli", "lib"]


def _espeak(backend: str) -> ESpeak:
    if backend == "lib" and not ESpeak.library_available():
        pytest.skip("libespeak-ng not found")
    return ESpeak("en-us", library="espeak-lib" if backend == "lib" else "espeak")


# every structural check runs on the command-line tool and on the shared library
@pytest.fixture(scope="module", params=BACKENDS)
def espeak(request):
    return _espeak(request.param)


@pytest.fixture(scope="module")
def espeak_cli():
    return _espeak("cli")


@pytest.fixture(scope="module")
def espeak_lib():
    return _espeak("lib")


# word chunks of a text stream, as the generator phonemizes them (`text_context`)
STREAM_WORDS = [
    "Get context",
    "early. context",
    "the context",
    "campfires context",
    "extinguished, context",
    "hypnotic context",
]

# The library translates the text without the synthesis pass the CLI runs, which on
# rare clauses moves a stress mark; the phonemes themselves agree.
_STRESS = str.maketrans("", "", "ˈˌ")


@pytest.mark.parametrize("text", SENTENCES + STREAM_CHUNKS + STREAM_WORDS)
def test_library_backend_matches_cli(espeak_cli, espeak_lib, text):
    norm = english_normalizer(text, strip_edges=False)
    cli = espeak_cli.phonemize(norm, separator="|")
    lib = espeak_lib.phonemize(norm, separator="|")
    assert cli.translate(_STRESS) == lib.translate(_STRESS), (cli, lib)


def test_library_backend_is_flagged(espeak_cli, espeak_lib):
    assert not espeak_cli.library
    assert espeak_lib.library
    assert espeak_lib._lib.version == espeak_cli.backend_version


def _phonemize(espeak, text, strip_edges=True):
    norm = english_normalizer(text, strip_edges=strip_edges)
    return norm, espeak.phonemize(norm, separator="|")


@pytest.mark.parametrize("text", SENTENCES)
def test_one_phoneme_word_per_word(espeak, text):
    norm, phones = _phonemize(espeak, text)
    assert len(phones.split()) == len(norm.split()), (norm, phones)


@pytest.mark.parametrize("text", SENTENCES)
def test_punctuation_follows_its_word(espeak, text):
    norm, phones = _phonemize(espeak, text)
    for word, phone_word in zip(norm.split(), phones.split()):
        mark = word[-1] if word[-1] in PUNCT_MAP else ""
        tail = phone_word[-1] if phone_word[-1] in PUNCT_MAP else ""
        assert mark == tail, (word, phone_word)


@pytest.mark.parametrize("text", SENTENCES)
def test_only_model_punctuation(espeak, text):
    _, phones = _phonemize(espeak, text)
    assert set(PUNCT_LIKE.findall(phones)) <= set(PUNCT_MAP), phones


@pytest.mark.parametrize("text", ["", "   ", "-", "—", "...", ","])
def test_empty_after_normalization(espeak, text):
    assert _phonemize(espeak, text)[1] == ""


@pytest.mark.parametrize("text", STREAM_CHUNKS)
def test_stream_chunk_keeps_edge_pauses(espeak, text):
    norm, phones = _phonemize(espeak, text, strip_edges=False)
    # every word survives, and the edge pauses are model punctuation
    assert len(phones.split()) == len(norm.split()), (norm, phones)
    assert phones.startswith(",") == norm.startswith(",")
    assert phones.endswith(",") == norm.endswith(",")
    assert set(PUNCT_LIKE.findall(phones)) <= set(PUNCT_MAP), phones


@pytest.mark.parametrize(
    "text, chunks",
    [
        (", what", ["what"]),
        (", what, ok", ["what", "ok"]),
        ("Hello, world.", ["Hello", "world"]),
        ("Hello", ["Hello"]),
    ],
)
def test_punctuation_split_keeps_text(espeak, text, chunks):
    # a lone opening mark used to drop the text after it (", what" -> ", ")
    assert espeak._punctuator.strip_to_restore(text)[0] == chunks


def test_hyphenated_word_is_one_word(espeak):
    phones = espeak.phonemize("well-known", separator="|")
    assert len(phones.split()) == 1
    assert "-" not in phones


def test_raw_spaced_dash_is_dropped(espeak):
    # Why the normalizer turns a spaced dash into a comma: "-" isn't in the
    # phonemizer's punctuation set, so espeak silently drops it.
    phones = espeak.phonemize("Well-known - said George.", separator="|")
    assert "-" not in phones
    assert "," not in phones
    assert len(phones.split()) == 3


def test_punctuation_split_keeps_hyphenated_words(espeak):
    chunks, puncs = espeak._punctuator.strip_to_restore("Well-known, said George.")
    assert chunks == ["Well-known", "said George"]
    assert [p.punc.strip() for p in puncs] == [",", "."]


def test_deterministic(espeak):
    text = english_normalizer("Well-known - said George.")
    assert espeak.phonemize(text, separator="|") == espeak.phonemize(
        text, separator="|"
    )
