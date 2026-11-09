# The following code is adapted from: https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/text/english/time_norm.py
import re

import inflect

_inflect = inflect.engine()

# am / pm / a.m. / P.M. / "P M" (punct_norm splits "P.M." into letters);
# captures the letter. The abbreviation's dot is consumed only when the
# sentence goes on ("4 p.m. sharp"), so a sentence-final "p.m." keeps its stop.
AM_PM = r"([aApP])\.?\s?[mM](?!\w)(?:\.(?=\s+[a-z0-9]))?"

_time_re = re.compile(
    r"""\b
                          ((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3]))  # hours
                          :
                          ([0-5][0-9])                            # minutes
                          (?:\s*"""
    + AM_PM
    + r""")?                                                      # am/pm
                          (?!\w)""",
    re.X,
)

# A bare hour with am/pm: "4pm" / "9 a.m."
_hour_re = re.compile(r"(?<![\d:.])\b(1[0-2]|0?[1-9])\s*" + AM_PM)


def _expand_num(n: int) -> str:
    return _inflect.number_to_words(n)


def _expand_time_english(match: "re.Match") -> str:
    hour = int(match.group(1))
    past_noon = hour >= 12
    time = []
    if hour > 12:
        hour -= 12
    elif hour == 0:
        hour = 12
        past_noon = True
    time.append(_expand_num(hour))

    minute = int(match.group(6))
    if minute > 0:
        if minute < 10:
            time.append("oh")
        time.append(_expand_num(minute))
    am_pm = match.group(7)
    if am_pm is None:
        time.append("p m" if past_noon else "a m")
    else:
        time.append(f"{am_pm.lower()} m")
    return " ".join(time)


def _expand_hour_english(match: "re.Match") -> str:
    return f"{_expand_num(int(match.group(1)))} {match.group(2).lower()} m"


def expand_time_english(text: str) -> str:
    text = re.sub(_time_re, _expand_time_english, text)
    return re.sub(_hour_re, _expand_hour_english, text)
