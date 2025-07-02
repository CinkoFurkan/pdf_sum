# whisper_utils.py
import whisper
import yt_dlp
import os
import logging
from config import Config

logger = logging.getLogger(__name__)

# Load Whisper model once globally
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model...")
        _whisper_model = whisper.load_model("base")  # "base" is more accurate than "small"
    return _whisper_model


def transcribe_audio(file_path):
    """Transcribe uploaded audio file with error handling."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("Audio file too large. Please upload a file smaller than 100MB.")

        model = get_whisper_model()
        logger.info(f"Transcribing audio file: {file_path}")

        # Transcribe with language detection
        result = model.transcribe(
            file_path,
            fp16=False,  # Disable FP16 for better compatibility
            language=None,  # Auto-detect language
            task='transcribe'
        )

        # Add detected language info
        detected_language = result.get('language', 'unknown')
        transcript = result['text'].strip()

        if not transcript:
            return "No speech detected in the audio file."

        return f"[Detected language: {detected_language}]\n\n{transcript}"

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")


def download_youtube_audio(youtube_url):
    """Download audio from YouTube URL with better error handling."""
    output_path = os.path.join(Config.UPLOAD_FOLDER, "yt_audio_%(id)s.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'duration': 3600,  # Max 1 hour videos
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading audio from: {youtube_url}")

            # Extract video info first
            info = ydl.extract_info(youtube_url, download=False)
            video_duration = info.get('duration', 0)

            # Check video duration (max 1 hour)
            if video_duration > 3600:
                raise ValueError("Video is too long. Please use videos shorter than 1 hour.")

            # Download the audio
            ydl.download([youtube_url])

            # Find the downloaded file
            video_id = info.get('id', 'unknown')
            downloaded_file = os.path.join(Config.UPLOAD_FOLDER, f"yt_audio_{video_id}.mp3")

            if not os.path.exists(downloaded_file):
                # Try alternative naming patterns
                for file in os.listdir(Config.UPLOAD_FOLDER):
                    if file.startswith("yt_audio_") and file.endswith(".mp3"):
                        downloaded_file = os.path.join(Config.UPLOAD_FOLDER, file)
                        break

            if os.path.exists(downloaded_file):
                logger.info(f"Successfully downloaded audio to: {downloaded_file}")
                return downloaded_file
            else:
                raise FileNotFoundError("Downloaded audio file not found")

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"YouTube download error: {str(e)}")
        raise ValueError(f"Failed to download YouTube video: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise Exception(f"An error occurred: {str(e)}")