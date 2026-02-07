import re

# Common English stopwords — kept minimal and inline to avoid external deps.
STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "above", "after", "again", "all", "also", "am",
    "any", "are", "aren", "because", "before", "below", "between", "both",
    "each", "few", "further", "get", "got", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "into", "its",
    "itself", "let", "me", "more", "most", "my", "myself", "off", "once",
    "only", "other", "our", "ours", "ourselves", "out", "over", "own",
    "re", "same", "she", "some", "such", "that", "their", "theirs",
    "them", "themselves", "these", "they", "this", "those", "through",
    "under", "until", "up", "we", "what", "when", "where", "which",
    "while", "who", "whom", "why", "you", "your", "yours", "yourself",
    "yourselves",
}

# Ligature replacements for common PDF extraction artifacts.
_LIGATURES: dict[str, str] = {
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# Regex: sentence-ending punctuation followed by whitespace or end-of-string.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Regex: hyphen at end of line (word broken across lines in PDF).
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

# Regex: non-alphanumeric characters (for tokenization).
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Regex: control characters except newline and tab.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalize raw PDF-extracted text.

    1. Remove control characters and null bytes.
    2. Fix ligatures (ﬁ → fi, ﬂ → fl).
    3. Re-join hyphenated line breaks (compli-\\ncated → complicated).
    4. Collapse multiple whitespace into single spaces.
    5. Strip leading/trailing whitespace.
    """
    if not text:
        return ""

    # Remove control characters.
    text = _CONTROL_CHARS_RE.sub("", text)

    # Replace ligatures.
    for lig, replacement in _LIGATURES.items():
        text = text.replace(lig, replacement)

    # Re-join hyphenated line breaks.
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def count_tokens(text: str) -> int:
    """Approximate token count via whitespace split.

    This is intentionally simple — avoids adding a tokenizer dependency.
    Close enough for chunking decisions.
    """
    if not text or not text.strip():
        return 0
    return len(text.split())


def split_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split *text* into overlapping chunks of approximately *chunk_size* tokens.

    Respects sentence boundaries when possible.  If a single sentence exceeds
    *chunk_size*, it is force-split on whitespace.

    Returns an empty list if *text* is empty.
    """
    if not text or not text.strip():
        return []

    sentences = _SENTENCE_SPLIT_RE.split(text)
    # Filter out empty sentences produced by the split.
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # Edge case: a single sentence longer than chunk_size.
        if sentence_tokens > chunk_size:
            # Flush whatever we've accumulated so far.
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_token_count = 0

            # Force-split the long sentence on whitespace.
            words = sentence.split()
            for i in range(0, len(words), chunk_size):
                piece = " ".join(words[i : i + chunk_size])
                if piece:
                    chunks.append(piece)
            continue

        # Would adding this sentence exceed the limit?
        if current_token_count + sentence_tokens > chunk_size and current_sentences:
            # Save the current chunk.
            chunks.append(" ".join(current_sentences))

            # Build overlap: walk backwards through current_sentences until we
            # reach approximately chunk_overlap tokens.
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for prev in reversed(current_sentences):
                prev_tokens = count_tokens(prev)
                if overlap_tokens + prev_tokens > chunk_overlap:
                    break
                overlap_sentences.insert(0, prev)
                overlap_tokens += prev_tokens

            current_sentences = overlap_sentences
            current_token_count = overlap_tokens

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Don't forget the last chunk.
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def tokenize(text: str) -> list[str]:
    """Break *text* into lowercase tokens for keyword scoring.

    1. Lowercase.
    2. Split on non-alphanumeric characters.
    3. Remove stopwords.
    4. Remove empty strings.
    """
    if not text:
        return []

    lowered = text.lower()
    tokens = _NON_ALNUM_RE.split(lowered)
    return [t for t in tokens if t and t not in STOPWORDS]
