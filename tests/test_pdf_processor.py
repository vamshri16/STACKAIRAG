"""Unit tests for app/core/pdf_processor.py."""

import pytest

from app.core.pdf_processor import extract_text_from_pdf, process_pdf, PDFProcessingError


class TestExtractTextFromPdf:
    def test_valid_pdf(self, test_pdf_path):
        pages = extract_text_from_pdf(test_pdf_path)
        assert len(pages) >= 1
        # Pages are 1-indexed tuples of (page_number, text).
        for page_num, text in pages:
            assert page_num >= 1
            assert isinstance(text, str)
            assert len(text.strip()) > 0

    def test_corrupt_pdf_raises(self, corrupt_pdf_path):
        with pytest.raises(PDFProcessingError):
            extract_text_from_pdf(corrupt_pdf_path)

    def test_empty_file_raises(self, empty_pdf_path):
        with pytest.raises((PDFProcessingError, Exception)):
            extract_text_from_pdf(empty_pdf_path)

    def test_nonexistent_file_raises(self):
        with pytest.raises((PDFProcessingError, FileNotFoundError)):
            extract_text_from_pdf("/nonexistent/path/fake.pdf")


class TestProcessPdf:
    def test_valid_pdf_returns_chunks(self, test_pdf_path):
        chunks = process_pdf(test_pdf_path, "test_sample.pdf")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.source == "test_sample.pdf"
            assert chunk.page >= 1
            assert chunk.text.strip()
            assert chunk.chunk_id.startswith("test_sample.pdf_p")

    def test_chunk_ids_are_unique(self, test_pdf_path):
        chunks = process_pdf(test_pdf_path, "test_sample.pdf")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_corrupt_pdf_raises(self, corrupt_pdf_path):
        with pytest.raises(PDFProcessingError):
            process_pdf(corrupt_pdf_path, "corrupt.pdf")

    def test_empty_pdf_raises(self, empty_pdf_path):
        with pytest.raises((PDFProcessingError, Exception)):
            process_pdf(empty_pdf_path, "empty.pdf")
