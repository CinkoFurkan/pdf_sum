# server_fastapi.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rag_chatbot import retrieve_answer, summarize_pdf, remove_temp_file, get_llm
from whisper_utils import transcribe_audio, download_youtube_audio
import os

app = FastAPI()

# --- Enable CORS for Mobile App (Flutter) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later you can restrict this to your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Home endpoint ---
@app.get("/")
async def home():
    return {"message": "AI Assistant Backend is running 🚀"}

# --- Ask a question about PDF ---
@app.post("/ask")
async def ask(file: UploadFile = File(...), query: str = Form(...)):
    try:
        file_path = f"./temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        answer = retrieve_answer(file_path, query)
        remove_temp_file(file_path)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Summarize a PDF ---
@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    try:
        file_path = f"./temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        summary = summarize_pdf(file_path)
        remove_temp_file(file_path)
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Transcribe uploaded audio ---
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        file_path = f"./temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        transcript = transcribe_audio(file_path)
        remove_temp_file(file_path)
        return {"transcript": transcript}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Summarize a transcript ---
@app.post("/summarize_transcript")
async def summarize_transcript(data: dict):
    try:
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
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Transcribe from YouTube URL ---
@app.post("/youtube_transcribe")
async def youtube_transcribe(data: dict):
    try:
        youtube_url = data.get("url")

        file_path = download_youtube_audio(youtube_url)
        transcript = transcribe_audio(file_path)
        remove_temp_file(file_path)

        return {"transcript": transcript}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

