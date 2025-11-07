# app/prompts.py (สร้างไฟล์ใหม่นี้)

from llama_index.core.prompts import PromptTemplate

# --- 1. Persona Prompt หลัก (QA Prompt) ---
# เราจะใช้ Prompt นี้ทั้งใน /query และ /chat
# เราสามารถเพิ่ม "Persona" (บุคลิก) ได้ที่นี่
DEFAULT_THAI_QA_PROMPT_TMPL = (
    "เรามีข้อมูลบริบทดังต่อไปนี้: \n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "จงใช้ข้อมูลบริบทนี้เพื่อตอบคำถามเท่านั้น ห้ามใช้ความรู้เดิมที่มี"
    "จงตอบอย่างสุภาพและเป็นมิตรเสมอ \n"
    "คำถาม: {query_str}\n"
    "คำตอบ: "
)
THAI_QA_TEMPLATE = PromptTemplate(DEFAULT_THAI_QA_PROMPT_TMPL)


# --- 2. Prompt สรุปคำถาม (สำหรับ /chat) ---
DEFAULT_THAI_CONDENSE_TMPL = (
    "จากประวัติการสนทนาและคำถามล่าสุด, จงเรียบเรียงคำถามล่าสุด"
    "ให้เป็นคำถามที่สมบูรณ์และเข้าใจได้ในตัวเอง (Standalone Question)\n"
    "ประวัติการสนทนา: \n"
    "{chat_history}\n"
    "คำถามล่าสุด: {question}\n"
    "คำถามที่สมบูรณ์: "
)
THAI_CONDENSE_TEMPLATE = PromptTemplate(DEFAULT_THAI_CONDENSE_TMPL)