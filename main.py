# --- imports (top) ---
import os, json
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from openai import OpenAI
from pydantic import BaseModel

# --- app first ---
app = FastAPI(title="ResuMatch.ai")

# --- CORS next (optional but recommended) ---
FRONTEND_ORIGIN = "*"  # replace with your frontend URL when ready
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RATE LIMITING: only after app exists ---
limiter = Limiter(key_func=get_remote_address, enabled=True)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# --- health check ---
@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "ResuMatch backend is running"
    }


# --- (optional) key check ---
@app.get("/check-key")
def check_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"status": "missing", "message": "OPENAI_API_KEY not found"}
    return {"status": "ok", "key_starts_with": key[:7] + "..."}

# --- OpenAI client ---
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- models & schema for tailor endpoint ---
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

SYSTEM_PROMPT = """You are ResuMatch.ai, a senior resume writer.
Rules:
- Rewrite to fit the job description precisely.
- Keep facts; do not invent employers, dates, or certifications.
- Prefer action verbs, outcomes, and quantified metrics (%, $, time).
- Remove redundancies and irrelevant details.
- Mirror the job’s vocabulary when truthful.
- Output concise, scannable bullets.
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
1) 2–3 sentence summary tailored to the job.
2) Short "improved resume notes" (what changed & why).
3) One-page cover letter (3–5 short paragraphs).
4) 3–5 sections with improved bullet points (JSON only).
"""

JSON_SCHEMA = {
    "name": "TailorResponse",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "improved_resume": {"type": "string"},
            "cover_letter": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "bullets": {"type": "array", "items": {"type": "string"}, "minItems": 2}
                    },
                    "required": ["heading", "bullets"]
                },
                "minItems": 3
            }
        },
        "required": ["summary", "improved_resume", "cover_letter", "sections"],
        "additionalProperties": False
    },
    "strict": True
}

# --- AI endpoint WITH rate limit & Request param ---
@app.post("/api/tailor", response_model=TailorResponse)
@limiter.limit("20/minute")
def tailor_resume(req: TailorRequest, request: Request):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _user_prompt(req)},
    ]
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=messages,
            response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
            temperature=0.5,
        )
        if hasattr(resp, "output_parsed") and resp.output_parsed:
            return TailorResponse(**resp.output_parsed)
        if hasattr(resp, "output_text") and resp.output_text:
            import json as _json
            return TailorResponse(**_json.loads(resp.output_text))
    except Exception:
        # Fallback to Chat Completions JSON mode
        try:
            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            import json as _json
            return TailorResponse(**_json.loads(chat.choices[0].message.content))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI error: {type(e).__name__}: {e}")
