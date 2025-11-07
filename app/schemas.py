# app/schemas.py (อัปเดต)

from pydantic import BaseModel
from typing import List, Optional

# --- สำหรับ Endpoint /upload ---
class UploadResponse(BaseModel):
    success: bool
    filename: str
    message: str

# --- สำหรับ Endpoint /query (Stateless) ---
class QueryRequest(BaseModel):
    question: str

class SourceNode(BaseModel):
    file_name: str
    page_number: int
    score: float
    text_content: str

class QueryResponse(BaseModel):
    answer: str
    source_nodes: List[SourceNode]

# --- (NEW) สำหรับ Endpoint /chat (Stateful) ---
class ChatRequest(BaseModel):
    question: str
    session_id: str  # เพิ่ม session_id