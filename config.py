import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-change-this")

    # Flask Configuration
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # File Upload Configuration
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'mp3', 'wav', 'm4a', 'mp4', 'avi'}

    # Model Configuration
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")

    # Security Configuration
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "./logs/app.log")

    # YouTube Configuration
    YOUTUBE_MAX_DURATION = int(os.getenv("YOUTUBE_MAX_DURATION_SECONDS", "3600"))
    YOUTUBE_AUDIO_QUALITY = os.getenv("YOUTUBE_AUDIO_QUALITY", "192")

    @staticmethod
    def validate():
        """Validate required configuration"""
        errors = []

        if not Config.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is missing")

        if Config.SECRET_KEY == "default-secret-key-change-this":
            errors.append("SECRET_KEY must be changed from default")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")