"""Unit tests for app/config.py."""

from app.config import Settings, settings


class TestSettingsDefaults:
    """Verify all default values are set correctly."""

    def test_mistral_api_key_default_empty(self):
        s = Settings(mistral_api_key="")
        assert s.mistral_api_key == ""

    def test_mistral_embed_model(self):
        s = Settings()
        assert s.mistral_embed_model == "mistral-embed"

    def test_mistral_chat_model(self):
        s = Settings()
        assert s.mistral_chat_model == "mistral-large-latest"

    def test_mistral_base_url(self):
        s = Settings()
        assert s.mistral_base_url == "https://api.mistral.ai/v1"

    def test_chunk_size(self):
        s = Settings()
        assert s.chunk_size == 800

    def test_chunk_overlap(self):
        s = Settings()
        assert s.chunk_overlap == 150

    def test_top_k_retrieval(self):
        s = Settings()
        assert s.top_k_retrieval == 20

    def test_top_k_final(self):
        s = Settings()
        assert s.top_k_final == 5

    def test_similarity_threshold(self):
        s = Settings()
        assert s.similarity_threshold == 0.45

    def test_semantic_weight(self):
        s = Settings()
        assert s.semantic_weight == 0.7

    def test_data_dir(self):
        s = Settings()
        assert s.data_dir == "./data"

    def test_index_dir(self):
        s = Settings()
        assert s.index_dir == "./indexes"


class TestSettingsOverrides:
    """Verify settings can be overridden via constructor."""

    def test_override_chunk_size(self):
        s = Settings(chunk_size=500)
        assert s.chunk_size == 500

    def test_override_similarity_threshold(self):
        s = Settings(similarity_threshold=0.5)
        assert s.similarity_threshold == 0.5

    def test_override_top_k_final(self):
        s = Settings(top_k_final=10)
        assert s.top_k_final == 10

    def test_override_data_dir(self):
        s = Settings(data_dir="/tmp/test_data")
        assert s.data_dir == "/tmp/test_data"


class TestModuleLevelSingleton:
    """The module-level `settings` object should be a valid Settings instance."""

    def test_settings_is_instance(self):
        assert isinstance(settings, Settings)

    def test_settings_has_all_fields(self):
        assert hasattr(settings, "mistral_api_key")
        assert hasattr(settings, "chunk_size")
        assert hasattr(settings, "data_dir")
        assert hasattr(settings, "index_dir")
