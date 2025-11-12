# app/main.py (อัปเดต 3.0 - เพิ่ม /chat)

from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

# --- LlamaIndex Imports ---
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage
from app.prompts import THAI_QA_TEMPLATE, THAI_CONDENSE_TEMPLATE
# --- Python Typing Imports ---
from typing import Optional, List, Dict 

# --- Our App Module Imports ---
import app.rag_pipeline as rag_pipeline
import app.config as config
from app.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    SourceNode,
    ChatRequest  # Import Schema ใหม่
)

# --- Global variable สำหรับ Query Engine ---
# (ใช้ Optional[...] เพื่อรองรับ Python 3.9)
query_engine: Optional[BaseQueryEngine] = None

# --- (NEW) Global variable สำหรับ Chat History ---
# นี่คือ In-Memory Store แบบง่ายๆ ครับ
# (ใน Production ควรใช้ Redis หรือ Database)
chat_histories: Dict[str, List[ChatMessage]] = {}


# --- Lifespan Manager (อัปเดต) ---
# ฟังก์ชันนี้จะรัน 1 ครั้งตอน FastAPI เริ่มทำงาน
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 1. โค้ดที่รันตอน Startup ---
    print("Application is starting up...")
    
    # ตั้งค่า LlamaIndex (LLM, Embed Model)
    rag_pipeline.setup_global_settings() 
    
    # สร้างและเก็บ Query Engine ไว้ใน Global variable
    global query_engine
    query_engine = rag_pipeline.get_query_engine()
    
    print("Application startup complete. Ready to serve requests.")
    yield
    
    # --- 2. โค้ดที่รันตอน Shutdown ---
    print("Application is shutting down...")
    query_engine = None
    chat_histories.clear() # (NEW) เคลียร์ memory เมื่อปิดแอป


# --- สร้างแอป FastAPI ---
app = FastAPI(
    title="RAG API Pipeline",
    description="API สำหรับ Indexing, Querying (Stateless), และ Chat (Stateful)",
    version="1.1.0",
    lifespan=lifespan
)


# --- Endpoint 1: Health Check (เหมือนเดิม) ---
@app.get("/health")
def health_check():
    """เช็กว่า API ทำงานอยู่หรือไม่"""
    return {"status": "ok"}


# --- Endpoint 2: Indexing (เหมือนเดิม) ---
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint สำหรับอัปโหลดไฟล์ PDF (Requirement 1)
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF allowed.")

    try:
        # อ่านไฟล์ PDF เป็น bytes
        pdf_bytes = await file.read()
        
        print(f"Received file for indexing: {file.filename}")
        
        # (NEW) รับค่า 2 ตัวจาก rag_pipeline
        success, metadata = rag_pipeline.index_pdf(pdf_bytes, file.filename)

        if success:
            return UploadResponse(
                success=True,
                filename=file.filename,
                message="File processed and indexed successfully.",
                extracted_metadata=metadata  # <-- (NEW) ส่ง metadata กลับไป
            )
        else:
            # ถ้า ocr_service หรือ indexing ล้มเหลว
            raise HTTPException(status_code=500, detail="Failed to process or index the file.")

    except Exception as e:
        print(f"[Error] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# --- Endpoint 3: Querying (Stateless) (เหมือนเดิม) ---
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Endpoint สำหรับค้นหา (RAG) แบบ Stateless (ไม่มีความจำ)
    """
    global query_engine
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query Engine is not available.")

    try:
        print(f"Received query: {request.question}")
        
        # ใช้ .query (Sync)
        response = query_engine.query(request.question) 
        
        answer = str(response)
        source_nodes = []
        for node in response.source_nodes:
            source_nodes.append(SourceNode(
                file_name=node.metadata.get("file_name", "Unknown"),
                page_number=node.metadata.get("page_number", 0),
                score=node.get_score(),
                text_content=node.get_text()
            ))

        return QueryResponse(answer=answer, source_nodes=source_nodes)

    except Exception as e:
        print(f"[Error] Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during query: {str(e)}")


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint สำหรับ Chat แบบ Stateful (มีหน่วยความจำ)
    """
    global query_engine, chat_histories
    
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query Engine is not available.")
    
    try:
        print(f"Received chat for session: {request.session_id}")
        
        # 1. ดึงประวัติแชทเก่า (หรือสร้างใหม่ถ้าไม่มี)
        session_history = chat_histories.get(request.session_id, [])
        
        # 2. สร้าง Memory Buffer สำหรับ Session นี้
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=8000,
            chat_history=session_history
        )
        
        # 3. สร้าง Chat Engine (CondenseQuestion)
        chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,  # ส่ง query_engine แทน retriever
            memory=memory,
            llm=Settings.llm,
            condense_question_prompt=THAI_CONDENSE_TEMPLATE,
            verbose=True
        )

        
        # 4. ยิงคำถาม
        response = chat_engine.chat(request.question)
        
        # 5. บันทึกประวัติแชทล่าสุดกลับเข้า Store
        chat_histories[request.session_id] = memory.get_all()
        
        # 6. คืนค่า
        answer = str(response)
        source_nodes = []
        
        for node in response.source_nodes:
            source_nodes.append(SourceNode(
                file_name=node.metadata.get("file_name", "Unknown"),
                page_number=node.metadata.get("page_number", 0),
                score=node.get_score(),
                text_content=node.get_text()
            ))
        
        return QueryResponse(answer=answer, source_nodes=source_nodes)
        
    except Exception as e:
        print(f"[Error] Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during chat: {str(e)}")
