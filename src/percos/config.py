"""Application-wide configuration loaded from environment / .env file."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="Model name")
    openai_base_url: str | None = Field(default=None, description="Optional custom base URL")

    # ── Database ─────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/percos.db",
        description="Async SQLAlchemy database URL",
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="ChromaDB persistence directory",
    )

    # ── Server ───────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: LogLevel = LogLevel.INFO

    # ── Security ─────────────────────────────────────────
    secret_key: str = "change-me-in-production"

    # ── Maintenance & Evolution ─────────────────────────
    maintenance_interval_minutes: int = Field(
        default=0,
        description="Run maintenance every N minutes (0=disabled)",
    )
    auto_evolve_reflection: bool = Field(
        default=True,
        description="Auto-run evolution pipeline on low-risk reflection proposals",
    )

    # ── Paths ────────────────────────────────────────────
    data_dir: Path = Field(default=Path("./data"), description="Root data directory")
    domain_schema_path: str = Field(
        default="./domain.yaml",
        description="Path to domain schema YAML file",
    )

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
