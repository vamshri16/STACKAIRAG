import re

# Each pattern maps a PII category name to its compiled regex.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    ),
    "email": re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
    ),
    "phone": re.compile(
        r"(?:\+\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b"
    ),
}


def contains_pii(text: str) -> bool:
    """Return True if *text* contains any recognised PII pattern."""
    if not text:
        return False
    for pattern in _PII_PATTERNS.values():
        if pattern.search(text):
            return True
    return False


def redact_pii(text: str) -> str:
    """Replace all recognised PII patterns in *text* with ``[REDACTED]``."""
    if not text:
        return text
    result = text
    for pattern in _PII_PATTERNS.values():
        result = pattern.sub("[REDACTED]", result)
    return result


def get_pii_types(text: str) -> list[str]:
    """Return the list of PII category names found in *text*.

    Example return value: ``["ssn", "email"]``
    """
    if not text:
        return []
    return [name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)]
