# 🧠 Multimodal AI Assistant with RAG

A comprehensive AI-powered assistant that combines state-of-the-art NLP and speech recognition technologies for intelligent document processing, audio transcription, and conversational AI.

<img width="1470" alt="AI Assistant Main Interface" src="https://github.com/user-attachments/assets/a3f11a0f-19fd-466a-8585-4aa76a450a7d" />

<img width="1466" alt="Document Q&A Interface" src="https://github.com/user-attachments/assets/230b958b-7a24-431f-951f-acbd5b4aaf09" />

## ✨ Features

- **📄 Document Q&A**: Upload PDF/DOCX files and ask questions using Retrieval-Augmented Generation (RAG)
- **🎙️ Audio Transcription**: Convert speech to text with OpenAI Whisper (supports 95+ languages)
- **🎥 YouTube Transcription**: Extract and transcribe audio from YouTube videos
- **📝 Document Summarization**: Generate concise summaries of lengthy documents
- **💬 Conversational AI**: Context-aware chat functionality with conversation history
- **⚡ Fast Response**: Average query response time under 3 seconds
- **🌍 Multi-language Support**: Automatic language detection and processing

## 🛠️ Tech Stack

### Backend
- **Python 3.11**
- **Flask** - Web framework
- **LangChain** - Document processing and RAG implementation
- **Groq API** - LLaMA 3 (8B) for text generation
- **OpenAI Whisper** - Speech recognition
- **ChromaDB** - Vector database for semantic search
- **PyTorch** - ML framework
- **Transformers** - NLP models

### AI Models
- **LLM**: Groq LLaMA 3-8B-8192
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Speech**: Whisper base model

### Frontend
- **HTML5/CSS3/JavaScript**
- **Responsive Design**
- **Real-time Updates**

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/CinkoFurkan/pdf_sum.git
cd pdf_sum
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
```
Edit `.env` and add your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_secret_key_here
```

5. **Create necessary directories**
```bash
python setup.py
```

6. **Run the application**
```bash
python run.py
```

Visit `http://localhost:5000` in your browser.

## 📁 Project Structure

```
pdf_sum/
├── app/
│   ├── __init__.py
│   ├── routes/
│   ├── services/
│   └── utils/
├── config/
│   └── config.py
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
├── uploads/
├── logs/
├── main.py
├── rag_chatbot.py
├── whisper_utils.py
├── requirements.txt
├── .env.example
└── README.md
```

## 💻 Usage

### Document Q&A
1. Navigate to the PDF tab
2. Upload a PDF or DOCX file
3. Type your question in the text area
4. Click "Ask Question" to get AI-powered answers

### Audio Transcription
1. Go to the Audio tab
2. Upload an MP3, WAV, or M4A file
3. Click "Transcribe Audio" to convert speech to text
4. Optionally, click "Summarize Transcript" for a summary

### YouTube Transcription
1. Switch to the YouTube tab
2. Paste a YouTube video URL
3. Click "Transcribe YouTube Video"
4. The system will download, extract audio, and transcribe it

### Chat Mode
1. Open the Chat tab
2. Type your message
3. Send to have a conversation with the AI assistant

## 🔧 Configuration

### File Limits
- Maximum file size: 50MB
- Supported formats: PDF, DOCX, MP3, WAV, M4A
- YouTube video duration: Maximum 1 hour

### Model Configuration
You can adjust model settings in the `.env` file:
```
WHISPER_MODEL_SIZE=base
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL_NAME=llama3-8b-8192
```

## 📊 Performance

- **Document Processing**: 2-3 seconds for average PDF
- **Audio Transcription**: Real-time factor of 0.1-0.3x
- **Embedding Generation**: ~100ms per chunk
- **Vector Search**: <500ms for 10k chunks
- **Total Response Time**: <3 seconds average

## 🤝 Team

This project was developed by:
- **Furkan Çinko** (210201018) - Team Lead, RAG Implementation
- **Kutay Çakır** (210201039) - Frontend Development
- **Mert Genç** (210201070) - Audio Processing
- **Samet Balaban** (220201027) - Document Processing
- **Melih Şahin** (210201044) - Backend Architecture
- **İbrahim Halil Teymur** (210201058) - Testing & Documentation

**Instructor**: Dr. Öğr. Üyesi Gamze Uslu

## 📝 License

This project is developed for educational purposes as part of the Artificial Intelligence (COME405/1) course.

## 🔮 Future Enhancements

- [ ] Add support for more document formats (RTF, TXT, Markdown)
- [ ] Implement real-time voice input
- [ ] Add multi-document comparison features
- [ ] Create API documentation with Swagger
- [ ] Add user authentication and history
- [ ] Implement batch processing for multiple files
- [ ] Add export functionality (PDF, DOCX, JSON)
- [ ] Create mobile-responsive PWA version

## 🐛 Known Issues

- Large PDF files (>40MB) may timeout on first processing
- Some scanned PDFs require OCR preprocessing
- YouTube transcription depends on video having clear audio

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the team members.

---

⭐ If you find this project helpful, please give it a star!
