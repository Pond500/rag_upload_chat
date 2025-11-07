# app/config.py
import os
from dotenv import load_dotenv

load_dotenv() # โหลดตัวแปรจากไฟล์ .env

# --- 1. OCR Service (Typhoon) ---
OCR_API_ENDPOINT = "http://3.113.24.61/typhoon-ocr-service/v1/chat/completions"
OCR_API_MODEL = "typhoon-ocr-preview"

# --- 2. Embedding & Reranker Models (BAAI) ---
EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# --- 3. LLM (OpenAILike) ---
# คุณต้องไปตั้งค่านี้ในไฟล์ .env
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://3.113.24.61/llm-large-inference/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "pond")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ptm-gpt-oss-120b")

# --- 4. LlamaIndex & Framework Settings ---
# เราจะใช้ Settings กลางของ LlamaIndex
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# --- 5. Qdrant Database ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "my_rag_collection" # ชื่อตารางใน Qdrant