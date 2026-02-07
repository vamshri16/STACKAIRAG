from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mistral_api_key: str = ""
    mistral_embed_model: str = "mistral-embed"
    mistral_chat_model: str = "mistral-large-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieval: int = 20
    top_k_final: int = 5
    similarity_threshold: float = 0.7
    semantic_weight: float = 0.7

    data_dir: str = "./data"
    index_dir: str = "./indexes"

    model_config = {"env_file": ".env"}


settings = Settings()
