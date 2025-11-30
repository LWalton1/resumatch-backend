import os
import io
import json
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from openai import OpenAI

from docx import Document
from pdfminer.high_level import extract_text as pdf_extract_text

# ------------------ APP & MIDDLEWARE ------------------

app = FastAPI(title="ResuMatch.ai")

# Allow frontend to call backend
FRONTEND_ORIGIN = "*"  # later: replace "*" with your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, enabled=True)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ------------------ HEALTH & KEY CHECK ------------------

@app.get("/")
def health():
    return {"status": "ok", "message": "ResuMatch backend is running"}

@app.get("/check-key")
def check_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return {"status": "missing", "message": "OPENAI_API_KEY not found"}
    return {"status": "ok", "key_starts_with": key[:10] + "..."}

# ------------------ OPENAI CLIENT ------------------

api_key = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key)

# ------------------ MODELS ------------------

class TailorRequest(BaseModel):
    resume_text: str
    job_text: str
    target_title: Optional[str] = None
    tone: Optional[str] = "Professional"
    instructions: Optional[str] = ""

class TailoredSection(BaseModel):
    heading: str
    bullets: List[str]

class TailorResponse(BaseModel):
    summary: str
    improved_resume: str
    cover_letter: str
    sections: List[TailoredSection]

# ------------------ PROMPT HELPERS ------------------

SYSTEM_PROMPT = """You are ResuMatch.ai, a senior resume writer.
Rules:
- Rewrite to fit the job description precisely.
- Keep facts from the candidate; do NOT invent employers, dates, or certifications.
- Prefer action verbs, outcomes, and quantified metrics (%, $, time).
- Remove redundancies and irrelevant details.
- Mirror the job’s vocabulary when truthful.
- Output concise, scannable bullets.
- Always respond as a single JSON object with keys:
  summary, improved_resume, cover_letter, sections (array of {heading, bullets[]})
"""

def _user_prompt(req: TailorRequest) -> str:
    role = req.target_title or "the target role"
    return f"""
Target Title: {role}
Tone: {req.tone}
Extra Instructions: {req.instructions}

Job Description:
{req.job_text}

Candidate Resume:
{req.resume_text}

Tasks:
1) Write a 2–3 sentence summary tailored to the job.
2) Provide a short "improved resume notes" paragraph (what changed and why).
3) Draft a one-page cover letter (3–5 short paragraphs).
4) Provide 3–5 sections with improved bullet points.
Return ONLY valid JSON with:
{{
  "summary": "...",
  "improved_resume": "...",
  "cover_letter": "...",
  "sections": [
    {{
      "heading": "...",
      "bullets": ["...", "..."]
    }}
  ]
}}
"""

# ------------------ FILE PARSING HELPERS & ENDPOINT ------------------

def _docx_to_text(file_bytes: bytes) -> str:
    f = io.BytesIO(file_bytes)
    doc = Document(f)
    return "\n".join([p.text for p in doc.paragraphs])

def _pdf_to_text(file_bytes: bytes) -> str:
    f = io.BytesIO(file_bytes)
    return pdf_extract_text(f)

@app.post("/api/parse-resume")
def parse_resume(file: UploadFile = File(...)):
    """
    Accepts a PDF, DOCX, or TXT file and returns extracted text.
    """
    name = (file.filename or "").lower()
    content = file.file.read()
    try:
        if name.endswith(".docx"):
            text = _docx_to_text(content)
        elif name.endswith(".pdf"):
            text = _pdf_to_text(content)
        elif name.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload PDF, DOCX, or TXT."
            )
        # Safety cap in case user uploads a huge file
        return {"text": text[:200000]}
    finally:
        file.file.close()

# ------------------ AI TAILORING ENDPOINT ------------------

@app.post("/api/tailor", response_model=TailorResponse)
@limiter.limit("20/minute")
def tailor_resume(req: TailorRequest, request: Request):
    """
    AI-powered tailoring of resume + cover letter using OpenAI chat completions.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(req)},
    ]

    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        content = chat.choices[0].message.content
        data = json.loads(content)
        return TailorResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI error: {type(e).__name__}: {e}"
        )
