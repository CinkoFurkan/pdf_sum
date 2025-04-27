# server.py
from flask import Flask, request, jsonify, render_template, session
from rag_chatbot import retrieve_answer, summarize_document, remove_temp_file, get_llm
from whisper_utils import transcribe_audio, download_youtube_audio
from flask_session import Session

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        file = request.files.get("file")
        query = request.form.get("query")
        file_path = None

        if file:
            file_path = f"./temp_{file.filename}"
            file.save(file_path)

        # Initialize chat history if not exists
        if "chat_history" not in session:
            session["chat_history"] = []

        if file_path:
            answer = retrieve_answer(file_path, query)
            remove_temp_file(file_path)
        else:
            # If no file uploaded, pure chatbot mode
            llm = get_llm()
            history_text = "\n".join(session["chat_history"])
            prompt = f"""
The following is a conversation between a user and an AI assistant.

{history_text}

User: {query}
Assistant:"""
            answer = llm(prompt)

        # Update chat history
        session["chat_history"].append(f"User: {query}")
        session["chat_history"].append(f"Assistant: {answer}")

        session.modified = True
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        file = request.files["file"]
        file_path = f"./temp_{file.filename}"
        file.save(file_path)

        summary = summarize_document(file_path)
        remove_temp_file(file_path)
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        file = request.files["file"]
        file_path = f"./temp_{file.filename}"
        file.save(file_path)

        transcript = transcribe_audio(file_path)
        remove_temp_file(file_path)
        return jsonify({"transcript": transcript})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summarize_transcript", methods=["POST"])
def summarize_transcript():
    try:
        data = request.json
        transcript = data.get("transcript")

        llm = get_llm()
        prompt = f"""
You are a professional summarizer. 
Summarize the following conversation or spoken audio text into **three key sections**:
1. Main Topics
2. Important Details
3. Conclusion

Here is the transcript:
---
{transcript}
---
"""
        summary = llm(prompt)
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/youtube_transcribe", methods=["POST"])
def youtube_transcribe():
    try:
        data = request.json
        youtube_url = data.get("url")

        file_path = download_youtube_audio(youtube_url)
        transcript = transcribe_audio(file_path)
        remove_temp_file(file_path)

        return jsonify({"transcript": transcript})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
