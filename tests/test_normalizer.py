import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voxtream.utils.text.normalizer import english_normalizer
from voxtream.utils.text.number_norm import normalize_numbers

# Colons and semicolons are pauses, read as commas; a colon between digits is
# kept for time expansion.
COLON_SEMICOLON = [
    ("a well-known fact: it works; mostly.", "a well-known fact, it works, mostly."),
    ("Note: this", "Note, this"),
    ("one; two", "one, two"),
    ("2:1", "two, one"),
    ("Score 2:1 - nice.", "Score two, one, nice."),
    ("The score was 2:1 at 9:05.", "The score was two, one at nine oh five a m."),
    ("Meet at 10:15 pm, ok?", "Meet at ten fifteen p m, ok?"),
    ("15:45", "three forty-five p m"),
    ("12:00", "twelve p m"),
]

# Times: am/pm is spelled out, a dash between times is "to".
TIMES = [
    ("At 3:30 today.", "At three thirty a m today."),
    ("At 3:00 p.m. sharp", "At three p m sharp"),
    ("See you at 4 p.m.", "See you at four p m."),
    ("At 4 P.M. Then we left.", "At four p m. Then we left."),
    ("4pm", "four p m"),
    ("12am", "twelve a m"),
    ("3:30-4:00", "three thirty a m to four a m"),
    ("3:30-4:00 pm.", "three thirty a m to four p m."),
    ("9am-5pm", "nine a m to five p m"),
    ("9 a.m. - 5 p.m.", "nine a m to five p m."),
    ("9 AM-5 PM", "nine a m to five p m"),
    ("3-4pm", "three to four p m"),
    ("3 - 4 pm", "three to four p m"),
    ("10-11 pm", "ten to eleven p m"),
    # "am" / "a m" not after an hour is left alone
    ("I am happy", "I am happy"),
    ("I am 5 now", "I am five now"),
    ("Room 5 a meeting", "Room five a meeting"),
]

# Ordinals and decimals.
ORDINALS_DECIMALS = [
    ("1st 2nd 3rd 4th", "first second third fourth"),
    ("May 5th, 2021", "May fifth, twenty twenty-one"),
    ("the 21st century", "the twenty-first century"),
    ("2nd floor", "second floor"),
    ("4th.", "fourth."),
    ("1,000th", "one thousandth"),
    ("5thing", "five thing"),
    ("1990s", "nineteen ninety s"),
    ("3.14", "three point fourteen"),
    (".5", "point five"),
    ("costs .5 now", "costs point five now"),
    ("Hello, .5", "Hello, point five"),
    ("It is -.5 now", "It is minus point five now"),
    ("3 ,5", "thirty-five"),
]

# A spaced (or one-side spaced) dash is a pause, read as a comma; em/en dashes
# fold to it. Runs of dashes are one dash.
DASH_PAUSE = [
    ("Well-known - said George.", "Well-known, said George."),
    ("Well-known -said George.", "Well-known, said George."),
    ("Well-known- said George.", "Well-known, said George."),
    ("He said -hello- to me.", "He said, hello, to me."),
    ("He came — finally — at noon.", "He came, finally, at noon."),
    ("x — y", "x, y"),
    # an unspaced em/en dash is a hyphen
    ("He came—finally.", "He came-finally."),
    ("came–went", "came-went"),
    ("So - um - yes", "So, um, yes"),
    ("a--b", "a, b"),
    ("a - - b", "a, b"),
    ("Well -- I mean", "Well, I mean"),
    ("a -- 4", "a, four"),
    ("3 - 4", "three, four"),
    ("3- 4", "three, four"),
]

# A pause never opens the text or a sentence, a dash never closes the text,
# and a dash before a stop is dropped so the stop survives.
DASH_SENTENCE_EDGES = [
    ("Wait -", "Wait"),
    ("-Hello-", "Hello"),
    ("word -", "word"),
    ("x -", "x"),
    ("Hello -  ", "Hello"),
    ("-Hello", "Hello"),
    ("- Hello there.", "Hello there."),
    ("- word", "word"),
    ("--Hello", "Hello"),
    ("...Hello", "Hello"),
    (", Hello", "Hello"),
    ("- 4", "four"),
    ("Hello. -Hi there.", "Hello. Hi there."),
    ("Hello. -Hi there-. Bye", "Hello. Hi there. Bye"),
    ("Hi! - Yes.", "Hi! Yes."),
    ("Hello.-Hi", "Hello. Hi"),
    ("Wait - .", "Wait."),
    ("Wait -!", "Wait!"),
    ("Really -?", "Really?"),
    ("No -!", "No!"),
    ("Hi, - there.", "Hi, there."),
    ("Hello, - there", "Hello, there"),
]

# A dash is a sign only when it doesn't join a word or number.
NUMBER_SIGN = [
    ("-4", "minus four"),
    ("-4 degrees.", "minus four degrees."),
    ("It was -4 outside.", "It was minus four outside."),
    ("Temp: -4.", "Temp, minus four."),
    ("(-4)", "minus four"),
    ('"-4"', "minus four"),
    ("Value −4", "Value minus four"),
    ("3 -4", "three minus four"),
    ("-3.5 degrees", "minus three point five degrees"),
    ("It costs -3.5; ok", "It costs minus three point five, ok"),
    ("-1,000 dollars", "minus one thousand dollars"),
    # a dash before a time is not a sign
    ("-4am", "four a m"),
    ("-3:30", "three thirty a m"),
    ("-3:30 pm", "three thirty p m"),
]

# A dash joining numbers or words is a hyphen: never glued "minus".
HYPHEN = [
    ("3-4", "three-four"),
    ("4-5", "four-five"),
    ("3-4 pages", "three-four pages"),
    ("10-11 times a year", "ten-eleven times a year"),
    ("In 1999-2000.", "In nineteen ninety-nine-two thousand."),
    ("pages 10–20", "pages ten-twenty"),
    ("2021-05-03", "twenty twenty-one-five-three"),
    ("COVID-19 cases", "COVID-nineteen cases"),
    ("x-4", "x-four"),
    ("B-52 bomber", "B-fifty-two bomber"),
    ("F-16s", "F-sixteen s"),
    ("a 3-year-old", "a three-year-old"),
    ("x-ray, a well-known fact", "x-ray, a well-known fact"),
    ("twenty-five", "twenty-five"),
]

# Ellipses, repeated and stacked punctuation.
TERMINALS = [
    ("Wait... what?", "Wait, what?"),
    ("So.....yeah", "So, yeah"),
    ("Wait ...", "Wait,"),
    ("Hi!!! Really?!", "Hi! Really?"),
    ("What?? Why!!", "What? Why!"),
    ("He said, , ok", "He said, ok"),
    ("Hello,.", "Hello,"),
    ("etc., and so on", "etc. and so on"),
    ("A . B", "A B"),
]

# The rest of the pipeline: symbols, numbers, abbreviations, casing, scripts.
GENERAL = [
    ("", ""),
    ("Hello", "Hello"),
    ("  lots   of   space  ", "lots of space"),
    ("-", ""),
    ("—", ""),
    ("...", ""),
    (",", ""),
    ("e.g., this", "for example this"),
    ("i.e. that", "that is that"),
    (
        "Dr. Smith lives at 221 St. James.",
        "doctor Smith lives at two hundred twenty-one saint James.",
    ),
    ("Mr. and Mrs. Jones", "mister and misess Jones"),
    ("e.g. 50% of 1,000 people", "for example fifty percent of one thousand people"),
    ('She said: "yes" (maybe).', "She said, yes maybe."),
    ("VoXtream2 is fast", "Voxtreamtwo is fast"),
    ("iPhone and NASA", "iphone and NASA"),
    ("HELLO world", "HELLO world"),
    ("A. B. C. test", "A B C test"),
    ("Count 1, 2, 3 and go.", "Count one, two, three and go."),
    ("rock 'n' roll", "rock n roll"),
    ("‘quoted’ text", "quoted text"),
    ("don't", "don't"),
    ("snake_case and back\\slash", "snake case and back slash"),
    ("5′10″", "five ten"),
    ("Hello 👋 world 😀", "Hello world"),
    ("a & b", "a and b"),
    ("1 + 1 = 2", "one plus one equals two"),
    ("a/b", "a slash b"),
    ("me@x", "me at x"),
    ("100%", "one hundred percent"),
    ("$5", "five dollars"),
    ("£3", "three pounds sterling"),
    ("1,000,000", "one million"),
    ("2000", "two thousand"),
    ("2005", "two thousand five"),
    ("1066", "ten sixty-six"),
    ("0", "zero"),
    ("中文。", "中文."),
    ("你好，世界！", "你好,世界!"),
    ("Привет", "Привет"),
]

ALL_CASES = (
    COLON_SEMICOLON
    + TIMES
    + ORDINALS_DECIMALS
    + DASH_PAUSE
    + DASH_SENTENCE_EDGES
    + NUMBER_SIGN
    + HYPHEN
    + TERMINALS
    + GENERAL
)

# A stream chunk (strip_edges=False) keeps the pauses at its edges: they are
# pauses between chunks. Everything inside the chunk is normalized as usual.
STREAM_CHUNKS = [
    ("-", ","),
    ("—", ","),
    ("...", ","),
    (",", ","),
    ("Wait -", "Wait,"),
    ("Wait...", "Wait,"),
    ("- what", ", what"),
    (", world", ", world"),
    ("-Hello-", ", Hello,"),
    ("-4am", ", four a m"),
    ("-4", "minus four"),
    ("Hello", "Hello"),
    ("", ""),
    ("Well-known - said", "Well-known, said"),
    ("Wait - .", "Wait."),
]

# Inputs for the invariant checks: every case plus some free text.
CORPUS = [text for text, _ in ALL_CASES] + [
    "I paid $3.50 for 2 apples on May 5th, 2021.",
    "The 1990s were great!!! Weren't they?",
    "555-1234",
    "- - -",
    "-- --",
    "Hello - - - world",
    "Wait... - what?!",
    "Hi -- , - there.",
    "A: - B; - C",
    "1-2-3 go",
    "-1 -2 -3",
    "Go -- now -- please -- ok.",
]


def _ids(cases):
    return [text or "<empty>" for text, *_ in cases]


@pytest.mark.parametrize("text, expected", ALL_CASES, ids=_ids(ALL_CASES))
def test_english_normalizer(text, expected):
    assert english_normalizer(text) == expected


@pytest.mark.parametrize("text, expected", STREAM_CHUNKS, ids=_ids(STREAM_CHUNKS))
def test_english_normalizer_stream_chunk(text, expected):
    assert english_normalizer(text, strip_edges=False) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3-4", "three-four"),
        ("-4", "minus four"),
        ("x -4", "x minus four"),
        ("x-4", "x-four"),
        ("COVID-19", "COVID-nineteen"),
        ("4pm", "four pm"),
        ("1,000", "one thousand"),
        ("1st", "first"),
        ("22nd", "twenty-second"),
        ("3.5", "three point five"),
        (".5", "point five"),
        ("-.5", "minus point five"),
        ("x-.5", "x-point five"),
    ],
)
def test_normalize_numbers(text, expected):
    assert normalize_numbers(text) == expected


@pytest.mark.parametrize("text", CORPUS, ids=_ids([(t,) for t in CORPUS]))
def test_invariants(text):
    out = english_normalizer(text)
    assert out == out.strip(), "surrounding whitespace"
    assert "  " not in out, "double space"
    assert not out.startswith((",", ".", "!", "?")), "opens with punctuation"
    assert not re.search(r"\s[.,!?]", out), "space before punctuation"
    assert not re.search(r"[.,!?]\s*,", out), "stacked pause"
    assert " - " not in out and not re.search(r"(^|\s)-|-(\s|$)", out), "dash left"
    assert not re.search(r"\wminus", out), "glued minus"
    assert not re.search(r"[:;…—–]", out), "unfolded punctuation"


@pytest.mark.parametrize("text", CORPUS, ids=_ids([(t,) for t in CORPUS]))
def test_idempotent(text):
    out = english_normalizer(text)
    assert english_normalizer(out) == out
