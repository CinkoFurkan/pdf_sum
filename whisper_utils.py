# whisper_utils.py
import whisper
import yt_dlp
import os

# Load Whisper model once globally
whisper_model = whisper.load_model("small")

def transcribe_audio(file_path):
    """Transcribe uploaded audio file."""
    result = whisper_model.transcribe(file_path)
    return result['text']

def download_youtube_audio(youtube_url):
    """Download audio from YouTube URL."""
    output_path = "./yt_audio.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    return "./yt_audio.mp3"
