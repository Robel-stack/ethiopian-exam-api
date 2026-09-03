import os
import json
import uuid
import random
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    return genai.Client(api_key=api_key.strip(), http_options=types.HttpOptions(api_version="v1"))

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

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        if page_num > 7:
            text = doc[page_idx].get_text("text").strip().replace("\n", " ")
            if len(text.split()) >= 35:
                CURRICULUM_INDEX.append({
                    "chunk_id": str(uuid.uuid4())[:8],
                    "subject": subject,
                    "grade_level": grade,
                    "page_number": page_num,
                    "text": text[:900]
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
    if not CURRICULUM_INDEX:
        raise HTTPException(status_code=400, detail="No curriculum uploaded yet.")

    pool = [c for c in CURRICULUM_INDEX if c["subject"].lower() == req.subject.lower()]
    if not pool:
        raise HTTPException(status_code=404, detail=f"No textbook content found for '{req.subject}'.")

    client = get_gemini_client()
    selected_chunks = random.sample(pool, min(len(pool), req.total_questions * 2))
    assembled_items = []

    system_instruction = f"""
    You are an expert exam author for Ethiopian Secondary Education (Grade {req.grade_level} {req.subject}, EUEE standard).
    Generate challenging 4-option multiple-choice questions.
    Output MUST be valid JSON adhering strictly to:
    {{
        "question": "string",
        "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
        "correct_answer": "A | B | C | D",
        "rubric_explanation": "string",
        "bloom_level": "Knowledge | Application | Analysis"
    }}
    """

    for chunk in selected_chunks:
        if len(assembled_items) >= req.total_questions:
            break

        prompt = f"Textbook excerpt (Page {chunk['page_number']}):\n{chunk['text']}\nGenerate one EUEE multiple-choice question."
        for model_name in ["gemini-flash-lite-latest", "gemini-flash-latest"]:
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                        response_mime_type="application/json"
                    )
                )
                item = json.loads(res.text)
                item["item_number"] = len(assembled_items) + 1
                item["subject"] = req.subject
                item["page_reference"] = chunk["page_number"]
                assembled_items.append(item)
                break
            except Exception:
                continue

    return {
        "exam_id": f"ETH-{req.subject[:4].upper()}-{str(uuid.uuid4())[:6].upper()}",
        "subject": req.subject,
        "total_items": len(assembled_items),
        "questions": assembled_items
    }
