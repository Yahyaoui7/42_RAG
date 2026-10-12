import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase then extract alphanumeric tokens.

    Keeps logic words (not/none/and/or/is/in) for code sensitivity.
    Example: 'API_server v2.0' -> ['api', 'server', 'v2', '0']
    Example: 'if not user.is_active: return None' -> ['if', 'not', 'user', 'is', 'active', 'return', 'none']
    """
    return _TOKEN_RE.findall(text.lower())


if __name__ == "__main__":
    print(tokenize("Python makes building search engines easier"))
    print(tokenize("if not user.is_active: return None"))
    print(tokenize("API_server v2.0"))



