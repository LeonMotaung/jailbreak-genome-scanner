"""Configuration management for the scanner."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Perplexity API (DEPRECATED - Use Lambda scraper instead)
    perplexity_api_key: Optional[str] = Field(default=None, alias="PERPLEXITY_API_KEY")
    
    # Twitter/X API
    twitter_bearer_token: Optional[str] = Field(default=None, alias="TWITTER_BEARER_TOKEN")
    twitter_api_key: Optional[str] = Field(default=None, alias="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(default=None, alias="TWITTER_API_SECRET")
    twitter_access_token: Optional[str] = Field(default=None, alias="TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret: Optional[str] = Field(
        default=None, alias="TWITTER_ACCESS_TOKEN_SECRET"
    )
    
    # Reddit API
    reddit_client_id: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(
        default="jailbreak-genome-scanner/1.0", alias="REDDIT_USER_AGENT"
    )
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    
    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    
    # Lambda Cloud
    lambda_api_key: Optional[str] = Field(default=None, alias="LAMBDA_API_KEY")
    lambda_default_instance_type: str = Field(
        default="gpu_1x_a10", alias="LAMBDA_DEFAULT_INSTANCE_TYPE"
    )
    lambda_default_region: str = Field(
        default="us-east-1", alias="LAMBDA_DEFAULT_REGION"
    )
    lambda_use_default_firewall: bool = Field(
        default=True, alias="LAMBDA_USE_DEFAULT_FIREWALL"
    )
    lambda_filesystem_id: Optional[str] = Field(
        default=None, alias="LAMBDA_FILESYSTEM_ID"
    )
    
    # ChromaDB Configuration
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection_name: str = Field(
        default="jailbreak_embeddings", alias="CHROMA_COLLECTION_NAME"
    )
    
    # Arena Configuration
    arena_rounds: int = Field(default=100, alias="ARENA_ROUNDS")
    min_attackers: int = Field(default=5, alias="MIN_ATTACKERS")
    scoring_threshold: float = Field(default=0.7, alias="SCORING_THRESHOLD")
    
    # Agent Configuration
    monitor_interval: int = Field(default=300, alias="MONITOR_INTERVAL")
    min_coordination_threshold: int = Field(
        default=5, alias="MIN_COORDINATION_THRESHOLD"
    )
    min_bot_similarity_score: float = Field(
        default=0.85, alias="MIN_BOT_SIMILARITY_SCORE"
    )
    coordination_time_window_minutes: int = Field(
        default=60, alias="COORDINATION_TIME_WINDOW_MINUTES"
    )
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/scanner.log", alias="LOG_FILE")
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    logs_dir: Path = project_root / "logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()

