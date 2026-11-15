from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    allowed_origins: str = "*"
    # Earthdata Login bearer token — only the pipeline needs it, to download granules from ASF.
    earthdata_token: str = ""

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def events_dir(self) -> Path:
        return self.data_dir / "events"


settings = Settings()
