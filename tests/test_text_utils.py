"""Unit tests for app/utils/text_utils.py."""

from app.utils.text_utils import clean_text, count_tokens, split_into_chunks, tokenize


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("  hello   world  \n\t ") == "hello world"

    def test_replaces_ligatures(self):
        assert clean_text("The \ufb01rst \ufb02oor") == "The first floor"

    def test_replaces_ffi_ffl_ligatures(self):
        assert clean_text("o\ufb03ce wa\ufb04e") == "office waffle"

    def test_fixes_hyphenated_linebreaks(self):
        assert clean_text("compli-\ncated") == "complicated"

    def test_removes_control_characters(self):
        result = clean_text("a\x00b\x01c")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_falsy_input(self):
        assert clean_text("") == ""

    def test_already_clean(self):
        assert clean_text("hello world") == "hello world"

    def test_preserves_sentence_punctuation(self):
        result = clean_text("Hello. World! How?")
        assert result == "Hello. World! How?"


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_normal_text(self):
        assert count_tokens("hello world foo bar") == 4

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_whitespace_only(self):
        assert count_tokens("   ") == 0

    def test_single_word(self):
        assert count_tokens("hello") == 1


# ---------------------------------------------------------------------------
# split_into_chunks
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    def test_empty_string(self):
        assert split_into_chunks("", chunk_size=10, chunk_overlap=2) == []

    def test_whitespace_only(self):
        assert split_into_chunks("   ", chunk_size=10, chunk_overlap=2) == []

    def test_short_text_single_chunk(self):
        result = split_into_chunks("Short text here.", chunk_size=100, chunk_overlap=10)
        assert len(result) == 1
        assert "Short text here." in result[0]

    def test_multiple_chunks_created(self):
        # Create text with many sentences.
        sentences = [f"Sentence number {i} has some words." for i in range(30)]
        text = " ".join(sentences)
        result = split_into_chunks(text, chunk_size=20, chunk_overlap=5)
        assert len(result) > 1

    def test_chunks_respect_size_limit(self):
        sentences = [f"Sentence number {i} has some extra words here." for i in range(50)]
        text = " ".join(sentences)
        result = split_into_chunks(text, chunk_size=20, chunk_overlap=5)
        for chunk in result:
            # Allow some tolerance — sentence-aware splitting can slightly exceed.
            assert count_tokens(chunk) <= 30  # 20 + tolerance

    def test_overlap_exists(self):
        sentences = [f"Unique word{i} appears in sentence {i}." for i in range(20)]
        text = " ".join(sentences)
        result = split_into_chunks(text, chunk_size=15, chunk_overlap=5)
        if len(result) >= 2:
            # Last words of chunk 0 should appear at the start of chunk 1.
            words_end_0 = set(result[0].split()[-5:])
            words_start_1 = set(result[1].split()[:10])
            assert len(words_end_0 & words_start_1) > 0

    def test_long_sentence_force_split(self):
        # One sentence with 50 words, chunk_size=10.
        long_sentence = " ".join([f"word{i}" for i in range(50)])
        result = split_into_chunks(long_sentence, chunk_size=10, chunk_overlap=2)
        assert len(result) >= 5  # 50 words / 10 per chunk


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_removes_stopwords(self):
        result = tokenize("The quick brown fox")
        assert "the" not in result
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_lowercases(self):
        result = tokenize("Hello WORLD")
        assert "hello" in result
        assert "world" in result

    def test_empty_string(self):
        assert tokenize("") == []

    def test_splits_on_punctuation(self):
        result = tokenize("hello, world! foo.")
        assert "hello" in result
        assert "world" in result
        assert "foo" in result

    def test_removes_common_stopwords(self):
        result = tokenize("a an the and or but in on at to for of with by")
        assert result == []
