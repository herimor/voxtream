import logging
import re
import subprocess
from collections import OrderedDict
from typing import Dict, List, Tuple

from packaging.version import Version

from voxtream.utils.text.base_phonemizer import BasePhonemizer
from voxtream.utils.text.punctuation import Punctuation

DEFAULT_PUNCTUATIONS = Punctuation.default_puncs()


def is_tool(name):
    from shutil import which

    return which(name) is not None


# Use a regex pattern to match the espeak version, because it may be
# symlinked to espeak-ng, which moves the version bits to another spot.
espeak_version_pattern = re.compile(r"text-to-speech:\s(?P<version>\d+\.\d+(\.\d+)?)")


def get_espeak_version():
    output = subprocess.getoutput("espeak --version")
    match = espeak_version_pattern.search(output)

    return match.group("version")


def get_espeakng_version():
    output = subprocess.getoutput("espeak-ng --version")
    return output.split()[3]


# priority: espeakng > espeak
if is_tool("espeak-ng"):
    _DEF_ESPEAK_LIB = "espeak-ng"
    _DEF_ESPEAK_VER = get_espeakng_version()
elif is_tool("espeak"):
    _DEF_ESPEAK_LIB = "espeak"
    _DEF_ESPEAK_VER = get_espeak_version()
else:
    _DEF_ESPEAK_LIB = None
    _DEF_ESPEAK_VER = None


def _espeak_exe(espeak_lib: str, args: List, sync=False) -> List[str]:
    """Run espeak with the given arguments."""
    cmd = [
        espeak_lib,
        "-q",
        "-b",
        "1",  # UTF8 text encoding
    ]
    cmd.extend(args)
    logging.debug("espeakng: executing %s", repr(cmd))

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as p:
        res = iter(p.stdout.readline, b"")
        if not sync:
            p.stdout.close()
            if p.stderr:
                p.stderr.close()
            if p.stdin:
                p.stdin.close()
            return res
        res2 = []
        for line in res:
            res2.append(line)
        p.stdout.close()
        if p.stderr:
            p.stderr.close()
        if p.stdin:
            p.stdin.close()
        p.wait()
    return res2


class ESpeak(BasePhonemizer):
    """ESpeak wrapper calling `espeak` or `espeak-ng` from the command-line the perform G2P

    Args:
        language (str):
            Valid language code for the used backend.

        backend (str):
            Name of the backend library to use. `espeak` or `espeak-ng`. If None, set automatically
            prefering `espeak-ng` over `espeak`. Defaults to None.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to Punctuation.default_puncs().

        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to True.

    Example:

        >>> from TTS.tts.utils.text.phonemizers import ESpeak
        >>> phonemizer = ESpeak("tr")
        >>> phonemizer.phonemize("Bu Türkçe, bir örnektir.", separator="|")
        'b|ʊ t|ˈø|r|k|tʃ|ɛ, b|ɪ|r œ|r|n|ˈɛ|c|t|ɪ|r.'

    """

    _ESPEAK_LIB = _DEF_ESPEAK_LIB
    _ESPEAK_VER = _DEF_ESPEAK_VER
    _SUPPORTED_LANGS = None
    _SUPPORTED_LANGS_BACKEND = None

    def __init__(
        self,
        language: str = "en-us",
        backend=None,
        punctuations=None,
        keep_puncs=True,
        cache_size: int = 8192,
    ):
        if self._ESPEAK_LIB is None:
            raise Exception(
                " [!] No espeak backend found. Install espeak-ng or espeak to your system."
            )
        self.backend = self._ESPEAK_LIB

        # band-aid for backwards compatibility
        if language == "en":
            language = "en-us"
        if language == "zh-cn":
            language = "cmn"

        punctuations = DEFAULT_PUNCTUATIONS if punctuations is None else punctuations
        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        if backend is not None:
            self.backend = backend
        self._cache_size = max(0, int(cache_size))
        self._phoneme_cache: OrderedDict[Tuple[str, str, bool, str], str] = (
            OrderedDict()
        )

    @property
    def backend(self):
        return self._ESPEAK_LIB

    @property
    def backend_version(self):
        return self._ESPEAK_VER

    @backend.setter
    def backend(self, backend):
        if backend not in ["espeak", "espeak-ng"]:
            raise Exception("Unknown backend: %s" % backend)
        self._ESPEAK_LIB = backend
        self._ESPEAK_VER = (
            get_espeakng_version() if backend == "espeak-ng" else get_espeak_version()
        )
        # Backend-dependent cache; invalidate on backend switch.
        if hasattr(self, "_phoneme_cache"):
            self._phoneme_cache.clear()
        self.__class__._SUPPORTED_LANGS = None
        self.__class__._SUPPORTED_LANGS_BACKEND = None

    def auto_set_espeak_lib(self) -> None:
        if is_tool("espeak-ng"):
            self._ESPEAK_LIB = "espeak-ng"
            self._ESPEAK_VER = get_espeakng_version()
        elif is_tool("espeak"):
            self._ESPEAK_LIB = "espeak"
            self._ESPEAK_VER = get_espeak_version()
        else:
            raise Exception(
                "Cannot set backend automatically. espeak-ng or espeak not found"
            )

    @staticmethod
    def name():
        return "espeak"

    def phonemize_espeak(
        self, text: str, separator: str = "|", tie=False, language: str = None
    ) -> str:
        """Convert input text to phonemes.

        Args:
            text (str):
                Text to be converted to phonemes.

            tie (bool, optional) : When True use a '͡' character between
                consecutive characters of a single phoneme. Else separate phoneme
                with '_'. This option requires espeak>=1.49. Default to False.
        """
        # set arguments
        language = language if language is not None else self._language
        self._init_language(language)
        cache_key = (language, separator, tie, text)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        args = ["-v", f"{language}"]
        # espeak and espeak-ng parses `ipa` differently
        if tie:
            # use '͡' between phonemes
            if self.backend == "espeak":
                args.append("--ipa=1")
            else:
                args.append("--ipa=3")
        else:
            # split with '_'
            if self.backend == "espeak":
                if Version(self.backend_version) >= Version("1.48.15"):
                    args.append("--ipa=1")
                else:
                    args.append("--ipa=3")
            else:
                args.append("--ipa=1")
        if tie:
            args.append("--tie=%s" % tie)

        args.append(text)
        # compute phonemes
        phonemes = ""
        for line in _espeak_exe(self._ESPEAK_LIB, args, sync=True):
            logging.debug("line: %s", repr(line))
            ph_decoded = line.decode("utf8").strip()
            ph_decoded = re.sub(r"\(.+?\)", "", ph_decoded)
            phonemes += ph_decoded.strip()

        phonemes = phonemes.replace("_", separator)
        self._cache_set(cache_key, phonemes)
        return phonemes

    def _cache_get(self, key: Tuple[str, str, bool, str]) -> str | None:
        if self._cache_size <= 0:
            return None
        if key not in self._phoneme_cache:
            return None
        value = self._phoneme_cache.pop(key)
        self._phoneme_cache[key] = value
        return value

    def _cache_set(self, key: Tuple[str, str, bool, str], value: str) -> None:
        if self._cache_size <= 0:
            return
        if key in self._phoneme_cache:
            self._phoneme_cache.pop(key)
        self._phoneme_cache[key] = value
        while len(self._phoneme_cache) > self._cache_size:
            self._phoneme_cache.popitem(last=False)

    def _phonemize(self, text, separator: str = None, language: str = None) -> str:
        return self.phonemize_espeak(text, separator, tie=False, language=language)

    @classmethod
    def supported_languages(cls) -> Dict:
        """Get a dictionary of supported languages.

        Returns:
            Dict: Dictionary of language codes.
        """
        if cls._ESPEAK_LIB is None:
            return {}
        if (
            cls._SUPPORTED_LANGS is not None
            and cls._SUPPORTED_LANGS_BACKEND == cls._ESPEAK_LIB
        ):
            return cls._SUPPORTED_LANGS
        args = ["--voices"]
        langs = {}
        count = 0
        for line in _espeak_exe(cls._ESPEAK_LIB, args, sync=True):
            line = line.decode("utf8").strip()
            if count > 0:
                cols = line.split()
                lang_code = cols[1]
                lang_name = cols[3]
                langs[lang_code] = lang_name
            logging.debug("line: %s", repr(line))
            count += 1
        cls._SUPPORTED_LANGS = langs
        cls._SUPPORTED_LANGS_BACKEND = cls._ESPEAK_LIB
        return langs

    def version(self) -> str:
        """Get the version of the used backend.

        Returns:
            str: Version of the used backend.
        """
        args = ["--version"]
        for line in _espeak_exe(self.backend, args, sync=True):
            version = line.decode("utf8").strip().split()[2]
            logging.debug("line: %s", repr(line))
            return version
        return ""

    @classmethod
    def is_available(cls):
        """Return true if ESpeak is available else false"""
        return is_tool("espeak") or is_tool("espeak-ng")
