import os
import json
import uuid
import random
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import fitz

app = FastAPI(title="Ethiopian Grade 12 Exam Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRICULUM_INDEX = []

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key.strip())

class ExamRequest(BaseModel):
    subject: str = "Physics"
    total_questions: int = 3
    grade_level: int = 12

@app.get("/")
def root():
    return {"status": "running", "indexed_chunks": len(CURRICULUM_INDEX)}

@app.post("/api/v1/curriculum/upload")
async def upload_textbook(file: UploadFile = File(...), subject: str = Form("Physics"), grade: int = Form(12)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks_added = 0

    max_pages = min(len(doc), 40)

    for page_idx in range(max_pages):
        page_num = page_idx + 1
        if page_num > 5:
            text = doc[page_idx].get_text("text").strip().replace("\n", " ")
            if len(text.split()) >= 25:
                CURRICULUM_INDEX.append({
                    "chunk_id": str(uuid.uuid4())[:8],
                    "subject": subject,
                    "grade_level": grade,
                    "page_number": page_num,
                    "text": text[:700]
                })
                chunks_added += 1

    doc.close()
    return {
        "status": "success",
        "subject": subject,
        "new_chunks_added": chunks_added,
        "total_corpus_chunks": len(CURRICULUM_INDEX)
    }

@app.post("/api/v1/exams/generate")
def generate_exam(req: ExamRequest):
    client = get_gemini_client()
    
    # መጽሐፉ ገና ካልተጫነ በቀጥታ ከGemini አጠቃላይ የኢትዮጵያ Grade 12 ሥርዓተ-ትምህርት ጥያቄ ያወጣል
    pool = [c for c in CURRICULUM_INDEX if c["subject"].lower() == req.subject.lower()]
    
    assembled_items = []
    
    prompt = f"""
    Create {req.total_questions} standard Ethiopian University Entrance Examination (EUEE) multiple choice questions for Grade {req.grade_level} {req.subject}.
    Return ONLY a JSON list of objects with this schema:
    [
      {{
        "question": "question text",
        "options": {{"A": "option 1", "B": "option 2", "C": "option 3", "D": "option 4"}},
        "correct_answer": "A",
        "rubric_explanation": "explanation"
      }}
    ]
    """
    
    if pool:
        selected = random.sample(pool, min(len(pool), req.total_questions))
        context_text = "\n---\n".join([f"Page {c['page_number']}: {c['text']}" for c in selected])
        prompt += f"\n\nUse the following textbook content as context:\n{context_text}"

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7
        )
    )

    try:
        raw_items = json.loads(response.text)
        if isinstance(raw_items, dict) and "questions" in raw_items:
            raw_items = raw_items["questions"]
        for idx, item in enumerate(raw_items):
            item["item_number"] = idx + 1
            item["subject"] = req.subject
            assembled_items.append(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Gemini output: {str(e)} | Raw: {response.text}")

    return {
        "exam_id": f"ETH-{req.subject[:4].upper()}-{str(uuid.uuid4())[:6].upper()}",
        "subject": req.subject,
        "total_items": len(assembled_items),
        "questions": assembled_items
    }
    @app.get("/")
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "running", "indexed_chunks": len(CURRICULUM_INDEX)}
