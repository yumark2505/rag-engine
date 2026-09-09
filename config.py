from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Định vị đường dẫn tuyệt đối tới file .env
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Ép buộc nạp đè biến môi trường từ file .env
load_dotenv(dotenv_path=ENV_PATH, override=True)


class Settings(BaseSettings):
    # ── App & Server ─────────────────────────────────────────────
    APP_NAME: str = "Fastapi-rag"
    DEBUG: bool = False
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000, ge=1, le=65535)

    # ── Vector Store (pgvector) ──────────────────────────────────
    PGVECTOR_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/rag"
    )
    TOP_K: int = Field(default=8)

    # ── Chunking ─────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(default=1200, ge=50, le=5000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=2000)

    # ── Ollama Provider ──────────────────────────────────────────
    OLLAMA_BASE_URL: str = Field(default="http://127.0.0.1:11434")
    LLM_MODEL: str = Field(default="qwen3:8b")
    EMBEDDING_MODEL: str = Field(default="nomic-embed-text:latest")
    BATCH_SIZE: int = Field(default=64)

    # ── OpenAI Provider ──────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")

    # ── Google Provider ──────────────────────────────────────────
    GEMINI_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash")

    # ── vLLM Provider ────────────────────────────────────────────
    VLLM_BASE_URL: str = Field(default="http://127.0.0.1:8001/v1")
    VLLM_MODEL: str = Field(default="Qwen/Qwen2.5-7B-Instruct")

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()