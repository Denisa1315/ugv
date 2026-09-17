"""Application-level settings.

Domain-specific config (risk field weights, decision-engine thresholds, etc.)
is added incrementally in its own module as each pipeline phase is built,
rather than front-loaded here.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ugv-backend"
    host: str = "0.0.0.0"
    port: int = 8000

    # Origins allowed to call the API / connect the WebSocket telemetry feed.
    # 5183 is this project's pinned frontend dev port (see frontend/vite.config.ts);
    # 5173 is Vite's own default, kept for anyone running the frontend unpinned.
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5183",
        "http://127.0.0.1:5183",
    ]

    # -- simulator / environment (phase 1b) --------------------------------
    simulator_tick_hz: float = 10.0
    env_width: float = 20.0
    env_height: float = 20.0
    env_goal: tuple[float, float] = (18.0, 18.0)

    # -- telemetry / event log (phase 1l) ----------------------------------
    db_path: str = "ugv_events.db"

    model_config = SettingsConfigDict(env_prefix="UGV_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
