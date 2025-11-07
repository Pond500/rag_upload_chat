# app/ocr_service.py

import requests
import base64
from io import BytesIO
from pdf2image import convert_from_bytes
from PIL import Image
import app.config as config  # Import config ของเรา
from typing import List, Dict, Any

def _image_to_base64_url(image: Image.Image) -> str:
    """แปลง PIL Image เป็น data URL (Base64)"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def _call_typhoon_api(base64_url: str) -> str:
    """เรียก Typhoon OCR API และดึงข้อความออกมา"""
    
    payload = {
        "model": config.OCR_API_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Process this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_url}
                    }
                ]
            }
        ]
    }
    
    headers = {"Content-Type": "application/json"}

    try:
        # --- (FIX 1) ---
        # เพิ่ม Timeout เป็น 300 วินาที (5 นาที)
        response = requests.post(
            config.OCR_API_ENDPOINT, 
            headers=headers, 
            json=payload, 
            timeout=300 
        )
        # ----------------
        
        response.raise_for_status()  # แจ้งเตือนถ้า API error
        
        result_json = response.json()
        
        # ดึงข้อความจาก response (ปรับแก้ตรงนี้ถ้า response-format ไม่เหมือน OpenAI)
        page_text = result_json['choices'][0]['message']['content']
        return page_text

    except requests.exceptions.RequestException as e:
        print(f"  [Error] API Request Failed: {e}")
        return ""  # คืนค่าว่างถ้าหน้านั้นล้มเหลว
    except (KeyError, IndexError, TypeError) as e:
        print(f"  [Error] Cannot parse API response: {e}. Response: {response.text}")
        return ""

# --- ฟังก์ชันหลักที่เราจะเรียกใช้จากภายนอก ---

def get_text_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    ฟังก์ชันหลัก: แปลง PDF bytes เป็น List ของ {page_number, text}
    โดยใช้ Typhoon OCR API
    """
    
    print(f"Starting OCR process with Typhoon API...")
    pages_data = []
    
    try:
        # 1. แปลง PDF (bytes) เป็น List ของ PIL Images
        images = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as e:
        print(f"[Error] pdf2image failed: {e}")
        return [] # คืนค่า list ว่างถ้าแปลง PDF ไม่ได้

    # 2. วนลูปส่งแต่ละหน้าไป OCR
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"  Processing page {page_num}/{len(images)}...")
        
        # 3. แปลง Image เป็น Base64
        base64_url = _image_to_base64_url(image)
        
        # 4. เรียก API
        text = _call_typhoon_api(base64_url)
        
        # เราจะเพิ่มเฉพาะหน้าที่ OCR สำเร็จ
        if text:
            pages_data.append({
                "page_number": page_num,
                "text": text
            })

    print(f"OCR process finished. Succeeded on {len(pages_data)}/{len(images)} pages.")

    # --- (FIX 2) ---
    # ตรวจสอบความสมบูรณ์: ถ้าจำนวนหน้าที่ OCR ได้ ไม่เท่ากับจำนวนหน้าทั้งหมด
    # ให้ถือว่าไฟล์นี้ล้มเหลว (คืนค่า List ว่าง)
    if len(pages_data) != len(images):
        print(f"[Error] OCR Incomplete. Rejecting file due to partial success.")
        return []  
    # ----------------
    
    print(f"OCR complete. Extracted text from {len(pages_data)} pages.")
    return pages_data