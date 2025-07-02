# main.py
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from werkzeug.utils import secure_filename
import os
import uuid
from config import Config
from rag_chatbot import retrieve_answer, summarize_document, remove_temp_file, get_llm
from whisper_utils import transcribe_audio, download_youtube_audio

import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config.from_object(Config)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
Session(app)

# Create upload folder if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def save_uploaded_file(file):
    """Safely save uploaded file with unique name"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add unique identifier to prevent conflicts
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        return file_path
    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    try:
        file = request.files.get("file")
        query = request.form.get("query")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        file_path = None
        if file:
            file_path = save_uploaded_file(file)
            if not file_path:
                return jsonify({"error": "Invalid file type"}), 400

        # Initialize chat history if not exists
        if "chat_history" not in session:
            session["chat_history"] = []

        if file_path:
            answer = retrieve_answer(file_path, query)
            remove_temp_file(file_path)
        else:
            # If no file uploaded, pure chatbot mode
            llm = get_llm()
            history_text = "\n".join(session["chat_history"][-10:])  # Keep last 10 exchanges
            prompt = f"""
The following is a conversation between a user and an AI assistant.

{history_text}

User: {query}
Assistant:"""
            answer = llm(prompt)

        # Update chat history
        session["chat_history"].append(f"User: {query}")
        session["chat_history"].append(f"Assistant: {answer}")

        # Keep chat history size manageable
        if len(session["chat_history"]) > 20:
            session["chat_history"] = session["chat_history"][-20:]

        session.modified = True
        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.error(f"Error in /ask: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        file_path = save_uploaded_file(file)
        if not file_path:
            return jsonify({"error": "Invalid file type"}), 400

        summary = summarize_document(file_path)
        remove_temp_file(file_path)
        return jsonify({"summary": summary})

    except Exception as e:
        app.logger.error(f"Error in /summarize: {str(e)}")
        return jsonify({"error": "An error occurred processing your file"}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        file_path = save_uploaded_file(file)
        if not file_path:
            return jsonify({"error": "Invalid file type"}), 400

        transcript = transcribe_audio(file_path)
        remove_temp_file(file_path)
        return jsonify({"transcript": transcript})

    except Exception as e:
        app.logger.error(f"Error in /transcribe: {str(e)}")
        return jsonify({"error": "An error occurred processing your audio"}), 500


@app.route("/summarize_transcript", methods=["POST"])
def summarize_transcript():
    try:
        data = request.json
        transcript = data.get("transcript")

        if not transcript:
            return jsonify({"error": "No transcript provided"}), 400

        llm = get_llm()
        prompt = f"""
You are a professional summarizer. 
Summarize the following conversation or spoken audio text into **three key sections**:
1. Main Topics
2. Important Details
3. Conclusion

Here is the transcript:
---
{transcript[:5000]}  # Limit transcript length
---
"""
        summary = llm(prompt)
        return jsonify({"summary": summary})

    except Exception as e:
        app.logger.error(f"Error in /summarize_transcript: {str(e)}")
        return jsonify({"error": "An error occurred summarizing the transcript"}), 500


@app.route("/youtube_transcribe", methods=["POST"])
def youtube_transcribe():
    try:
        data = request.json
        youtube_url = data.get("url")

        if not youtube_url:
            return jsonify({"error": "No URL provided"}), 400

        file_path = download_youtube_audio(youtube_url)
        if not file_path:
            return jsonify({"error": "Failed to download YouTube audio"}), 500

        transcript = transcribe_audio(file_path)
        remove_temp_file(file_path)

        return jsonify({"transcript": transcript})

    except Exception as e:
        app.logger.error(f"Error in /youtube_transcribe: {str(e)}")
        return jsonify({"error": "An error occurred processing the YouTube video"}), 500


if __name__ == "__main__":
    app.run(debug=False)  # Never use debug=True in production