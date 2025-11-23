"""Configuration - Simple config for starter interface."""

from pathlib import Path

class Config:
    """Simple config wrapper using existing settings."""
    
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    INPUT_DIR = DATA_DIR / "input"
    OUTPUT_DIR = DATA_DIR / "output"
    STORAGE_DIR = DATA_DIR / "storage"
    
    # JVI weights (existing formula uses fixed weights)
    DEFAULT_WEIGHTS = {
        "exploit_rate": 0.3,
        "severity": 0.3,
        "diversity": 0.25,
        "coverage": 0.15
    }
    
    # Safety settings
    SHOW_HARMFUL = False
    CLASSIFIER_GATING = True
    
    # Model settings
    MODEL_TYPE = "mock"
    MODEL_NAME = "mock-model"
    
    # Embedding settings (uses existing embedder default)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    def ensure_dirs(self):
        """Ensure all required directories exist."""
        for dir_path in [self.INPUT_DIR, self.OUTPUT_DIR, self.STORAGE_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, **kwargs):
        """Allow config to be used as factory."""
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
        return self

