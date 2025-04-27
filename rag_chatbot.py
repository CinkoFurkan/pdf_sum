# rag_chatbot.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document as LangchainDocument
from docx import Document as DocxDocument
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from unstructured.partition.pdf import partition_pdf


# --- Load Groq API LLM ---
def get_llm():
    openai.api_key = "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"
    openai.api_base = "https://api.groq.com/openai/v1"
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

    def query_llm(prompt):
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512
        )
        return response['choices'][0]['message']['content']

    return query_llm

# --- Embedding Model ---
def get_embedding_model():
    model_name = "BAAI/bge-small-en-v1.5"
    return HuggingFaceEmbeddings(model_name=model_name)

# --- Reranker Model ---
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs):
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze()
    sorted_indices = torch.argsort(scores, descending=True)
    reranked_docs = [docs[i] for i in sorted_indices]
    return reranked_docs

# --- PDF Loader and Splitter ---
def smart_load_pdf(file_path):
    elements = partition_pdf(file_path)
    docs = []
    for el in elements:
        if el.text and el.category in ["Title", "NarrativeText"]:
            docs.append(LangchainDocument(page_content=el.text))
    return docs

# --- DOCX Loader ---
def smart_load_docx(file_path):
    doc = DocxDocument(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [LangchainDocument(page_content=full_text)]


def split_documents_smartly(docs):
    if not docs:
        return []
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "heading")])
    structured_text = "\n\n".join(doc.page_content for doc in docs)
    structured_docs = md_splitter.split_text(structured_text)
    if len(structured_docs) > 5:
        return structured_docs
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100, length_function=len)
        return splitter.split_documents(docs)


# --- Load and Split Smartly ---
def load_and_split(file_path):
    if file_path.endswith(".pdf"):
        docs = smart_load_pdf(file_path)
    elif file_path.endswith(".docx"):
        docs = smart_load_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

    return split_documents_smartly(docs)

# --- Vectorstore Builder ---
def build_vector_db(chunks):
    embedding_model = get_embedding_model()
    return Chroma.from_documents(chunks, embedding_model)

# --- RAG Answer Retriever ---
def retrieve_answer(file_path, query):
    chunks = load_and_split(file_path)
    vectordb = build_vector_db(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    retrieved_docs = retriever.get_relevant_documents(query)
    reranked_docs = rerank(query, retrieved_docs)[:4]

    llm = get_llm()
    context = "\n\n".join(doc.page_content for doc in reranked_docs)
    prompt = f"""
    You are a helpful assistant. Answer the user's question based on the context below.
    Always respond in the same language the question was asked.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    return llm(prompt)

# --- Summarize Full Document ---
def summarize_document(file_path):
    chunks = load_and_split(file_path)
    llm = get_llm()
    full_text = "\n\n".join([chunk.page_content for chunk in chunks])

    prompt = f"""
You are a professional summarizer. 
Summarize the following document into **three key sections**:
1. Main Topics
2. Important Details
3. Conclusion

Use clear and simple language. 

Here is the document:
---
{full_text}
---
"""
    return llm(prompt)

# --- File Cleaner ---
def remove_temp_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
