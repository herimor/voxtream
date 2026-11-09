import ctypes
import ctypes.util
import logging
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

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


class _LibESpeak:
    """
    `espeak_TextToPhonemes` of the espeak-ng shared library, bound through ctypes.

    The command-line backend spawns an `espeak-ng` process per call, which costs about
    10 ms of process start-up and dictionary loading for a word of text; the library
    call takes well under a millisecond with the same voice and phoneme inventory. It
    translates the text without the synthesis pass the CLI runs, which on rare clauses
    moves a stress mark (primary for secondary) and changes nothing else.

    One binding serves the whole process: the library keeps its state, the voice among
    it, in globals, so it is initialized once and the voice is set whenever a call asks
    for another one.
    """

    _shared: Optional["_LibESpeak"] = None

    def __init__(self, path: str):
        lib = ctypes.cdll.LoadLibrary(path)
        lib.espeak_Initialize.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.espeak_Initialize.restype = ctypes.c_int
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_SetVoiceByName.restype = ctypes.c_int
        lib.espeak_TextToPhonemes.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        lib.espeak_Info.argtypes = [ctypes.c_void_p]
        lib.espeak_Info.restype = ctypes.c_char_p
        # 0x02 is AUDIO_OUTPUT_SYNCHRONOUS: no audio device is opened
        if lib.espeak_Initialize(0x02, 0, None, 0) <= 0:
            raise RuntimeError(f"failed to initialize the espeak library {path}")
        self.lib = lib
        self.path = path
        self.version = lib.espeak_Info(None).decode("utf8").split()[0]
        self.voice: Optional[str] = None

    @staticmethod
    def find() -> Optional[str]:
        """
        Path of the shared library: `ESPEAK_LIBRARY`, else the loader's search, else
        the `lib` directory next to the `espeak-ng` executable (macOS dyld does not
        search Homebrew's /opt/homebrew/lib).
        """
        path = os.environ.get("ESPEAK_LIBRARY")
        if path:
            return path
        for name in ("espeak-ng", "espeak"):
            path = ctypes.util.find_library(name)
            if path:
                return path
        for name in ("espeak-ng", "espeak"):
            exe = shutil.which(name)
            if exe is None:
                continue
            prefix = os.path.dirname(os.path.dirname(os.path.realpath(exe)))
            lib_dir = os.path.join(prefix, "lib")
            for ext in ("dylib", "so", "dll"):
                path = os.path.join(lib_dir, f"lib{name}.{ext}")
                if os.path.exists(path):
                    return path
        return None

    @classmethod
    def shared(cls) -> "_LibESpeak":
        if cls._shared is None:
            path = cls.find()
            if path is None:
                raise RuntimeError(
                    "libespeak-ng not found: install espeak-ng with its shared library "
                    "or point ESPEAK_LIBRARY at it"
                )
            cls._shared = cls(path)
        return cls._shared

    def text_to_phonemes(self, text: str, voice: str, tie: bool = False) -> str:
        """
        The IPA phonemes of `text`, `_` between phonemes and a space between words, as
        `espeak-ng --ipa=1 -v {voice}` prints them; with `tie` a tie bar joins the
        characters of a phoneme instead (`--ipa=3`).
        """
        if voice != self.voice:
            # the same call the CLI makes for `-v`
            if self.lib.espeak_SetVoiceByName(voice.encode("utf8")) != 0:
                raise RuntimeError(f"espeak voice {voice!r} not found")
            self.voice = voice
        # bit 1 asks for IPA, bits 8-23 hold the separator; bit 7 makes it a tie bar
        mode = 0x02 | ((0x80 | ord("͡") << 8) if tie else ord("_") << 8)
        ptr = ctypes.pointer(ctypes.c_char_p(text.encode("utf8")))
        clauses = []
        # one clause per call; the pointer moves past it and ends at NULL
        while ptr.contents.value is not None:
            out = self.lib.espeak_TextToPhonemes(ptr, 1, mode)  # 1: UTF-8 text
            if out:
                clauses.append(out.decode("utf8"))
        return " ".join(clauses)


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

        library (str):
            `espeak-lib` calls the espeak-ng shared library in place of the command-line
            tool (`_LibESpeak`): the same phonemes without a process per call. Any other
            value (e.g. `espeak`) runs the tool, which is still needed to list the
            supported languages. Defaults to `espeak-lib`.

    Example:

        >>> phonemizer = ESpeak("en-us")
        >>> phonemizer.phonemize("Hello world.", separator="|")

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
        library: str = "espeak-lib",
    ):
        if self._ESPEAK_LIB is None:
            raise Exception(
                " [!] No espeak backend found. Install espeak-ng or espeak to your system."
            )
        self.backend = self._ESPEAK_LIB
        self._lib = _LibESpeak.shared() if library == "espeak-lib" else None

        if language == "en":
            language = "en-us"

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

    @staticmethod
    def library_available() -> bool:
        """Whether the espeak-ng shared library can be found (`library="espeak-lib"`)."""
        return _LibESpeak.find() is not None

    @property
    def library(self) -> bool:
        return self._lib is not None

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
        if self._lib is not None:
            lines = self._lib.text_to_phonemes(text, voice=language, tie=tie).split(
                "\n"
            )
        else:
            lines = [
                line.decode("utf8")
                for line in _espeak_exe(
                    self._ESPEAK_LIB, self._cli_args(language, tie, text), sync=True
                )
            ]
        # compute phonemes
        phonemes = ""
        for line in lines:
            logging.debug("line: %s", repr(line))
            ph_decoded = re.sub(r"\(.+?\)", "", line.strip())
            phonemes += ph_decoded.strip()

        phonemes = phonemes.replace("_", separator)
        self._cache_set(cache_key, phonemes)
        return phonemes

    def _cli_args(self, language: str, tie: bool, text: str) -> List[str]:
        """Arguments of the command-line call for `text`."""
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
        return args

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
