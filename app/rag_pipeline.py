# app/rag_pipeline.py

import app.config as config
from app.ocr_service import get_text_from_pdf
import os

# --- LlamaIndex Core ---
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import SimpleSummarize
# --- Models (LLM, Embedding) ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# --- Vector DB (Qdrant) ---
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

# --- Python ---
from typing import List
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage

from app.prompts import THAI_QA_TEMPLATE

# ---------------------------------------------------------------------
# 1. GLOBAL SETTINGS LOADER
# ---------------------------------------------------------------------

def setup_global_settings():
    """
    ตั้งค่า LlamaIndex Settings (LLM, Embed Model, Chunking)
    """
    print("Setting up LlamaIndex Global Settings...")
    
    Settings.llm = OpenAILike(
        model=config.LLM_MODEL_NAME,
        api_base=config.LLM_API_BASE,
        api_key=config.LLM_API_KEY,
        is_chat_model=True,
        max_tokens=1500,
        temperature=0.5,
        timeout=120,
        context_window=8192,
        is_function_calling_model=False
    )
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL_NAME,
        trust_remote_code=True,
        embed_batch_size=10
    )
    
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    print("Global Settings setup complete.")

# ---------------------------------------------------------------------
# 2. VECTOR STORE LOADER
# ---------------------------------------------------------------------

def get_vector_store() -> QdrantVectorStore:
    """เชื่อมต่อ Qdrant Client และ LlamaIndex VectorStore"""
    client = qdrant_client.QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT
    )
    
    return QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        enable_hybrid=True
    )

# ---------------------------------------------------------------------
# 3. INDEXING PIPELINE
# ---------------------------------------------------------------------

def index_pdf(pdf_bytes: bytes, file_name: str) -> bool:
    """
    Pipeline สำหรับ Indexing:
    1. OCR (Typhoon)
    2. Create LlamaIndex Documents (with metadata)
    3. Index to Qdrant
    """
    print(f"Indexing started for: {file_name}")
    page_data_list = get_text_from_pdf(pdf_bytes)
    
    if not page_data_list:
        print(f"[Error] OCR failed or returned no text for {file_name}.")
        return False
    
    print(f"Creating LlamaIndex Documents from {len(page_data_list)} pages...")
    documents: List[Document] = []
    
    for page_data in page_data_list:
        doc = Document(
            text=page_data['text'],
            metadata={
                "file_name": file_name,
                "page_number": page_data['page_number']
            }
        )
        documents.append(doc)
    
    print("Connecting to Qdrant Vector Store...")
    try:
        vector_store = get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        print(f"Indexing {len(documents)} documents...")
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print(f"Successfully indexed: {file_name}")
        return True
    except Exception as e:
        print(f"[Error] Indexing failed for {file_name}: {e}")
        return False

# ---------------------------------------------------------------------
# 4. QUERYING PIPELINE
# ---------------------------------------------------------------------

def get_query_engine() -> BaseQueryEngine:
    """
    สร้าง RAG Query Engine:
    - Qdrant Vector Store
    - Hybrid Search (Vector + BM25)
    - BAAI Reranker
    """
    print("Building Query Engine...")
    
    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    reranker = SentenceTransformerRerank(
        model=config.RERANKER_MODEL_NAME,
        top_n=3
    )
    
    retriever = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=10
    )
    
    synthesizer = SimpleSummarize(
        llm=Settings.llm,
        text_qa_template=THAI_QA_TEMPLATE
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker],
        response_synthesizer=synthesizer
    )
    
    print("Query Engine built successfully.")
    return query_engine
