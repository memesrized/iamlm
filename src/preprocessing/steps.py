import re

from IPython.display import clear_output
from polyglot.detect import Detector
from regex import UNICODE, VERBOSE, compile

# TODO: make more consistent API for functions
# input as list or as single text


def replace_url(string, substitute=""):
    sub = r"""
    (?<=^|[\s<"'(\[{])            # visual border
    (                             # RFC3986-like URIs:
        [A-z]+                    # required scheme
        ://                       # required hier-part
        (?:[^@]+@)?               # optional user
        (?:[\w-]+\.)+\w+          # required host
        (?::\d+)?                 # optional port
        (?:\/[^?\#\s'">)\]}]*)?   # optional path
        (?:\?[^\#\s'">)\]}]+)?    # optional query
        (?:\#[^\s'">)\]}]+)?      # optional fragment
    |                             # simplified e-Mail addresses:
        [\w.#$%&'*+/=!?^`{|}~-]+  # local part
        @                         # klammeraffe
        (?:[\w-]+\.)+             # (sub-)domain(s)
        \w+                       # TLD
    )(?=[\s>"')\]}]|$)            # visual border
    """
    sub = compile(sub, UNICODE | VERBOSE)
    return sub.sub(substitute, string)


def replace_html_tags(text, substitute=" "):
    """Replace html tags with space.

    May cause texts removal such as <sample text>.
    """
    return re.sub(r"<[^<>]*>", substitute, text)


def replace_punct(text):
    return re.sub("\W+", " ", text).strip()


def remove_non_russian(sentences, threshold=0.7):
    detector = Detector("", quiet=True)

    def is_rus(text):
        # find better way to use detector
        try:
            lang = detector.detect(text)
            clear_output()
        except Exception:
            return False

        if lang.code == "ru" and lang.confidence >= threshold:
            return True
        else:
            return False

    return [x for x in sentences if is_rus(x)]
