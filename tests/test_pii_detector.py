"""Unit tests for app/utils/pii_detector.py."""

from app.utils.pii_detector import contains_pii, get_pii_types


class TestContainsPii:
    def test_detects_ssn(self):
        assert contains_pii("My SSN is 123-45-6789") is True

    def test_detects_credit_card_dashes(self):
        assert contains_pii("Card: 4111-1111-1111-1111") is True

    def test_detects_credit_card_spaces(self):
        assert contains_pii("Card: 4111 1111 1111 1111") is True

    def test_detects_credit_card_no_separator(self):
        assert contains_pii("Card: 4111111111111111") is True

    def test_detects_email(self):
        assert contains_pii("Contact user@example.com") is True

    def test_detects_phone_with_country_code(self):
        assert contains_pii("Call +1 (555) 123-4567") is True

    def test_detects_phone_no_country_code(self):
        assert contains_pii("Call 555-123-4567") is True

    def test_no_pii_clean_query(self):
        assert contains_pii("What is machine learning?") is False

    def test_no_pii_empty(self):
        assert contains_pii("") is False

    def test_no_pii_short_numbers(self):
        assert contains_pii("The year 2023 saw growth") is False


class TestGetPiiTypes:
    def test_returns_ssn(self):
        types = get_pii_types("My SSN is 123-45-6789")
        assert "ssn" in types

    def test_returns_email(self):
        types = get_pii_types("Contact user@example.com")
        assert "email" in types

    def test_returns_multiple(self):
        types = get_pii_types("SSN 123-45-6789 email a@b.com")
        assert "ssn" in types
        assert "email" in types

    def test_returns_empty_for_clean(self):
        assert get_pii_types("What is machine learning?") == []

    def test_returns_empty_for_empty_string(self):
        assert get_pii_types("") == []
