from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from openai import OpenAI
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# limit the AI endpoint to, say, 20/min per IP
@app.post("/api/tailor")
@limiter.limit("20/minute")
def tailor_resume(req: TailorRequest):
    ...
import os

app = FastAPI(title="ResuMatch.ai")

FRONTEND_ORIGIN = "https://resumatch-frontend-lwatlon.onrender.com"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health check
@app.get("/")
def health():
    return {"status": "ok", "message": "ResuMatch backend is running"}

# ✅ OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ✅ AI-powered resume tailoring
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
- Keep facts from the candidate, but rephrase for impact and clarity.
- Prefer action verbs, outcomes, and quantified metrics (%, $, time).
- Remove redundancies and irrelevant details.
- Mirror the job’s vocabulary when truthful.
- Output concise, scannable bullets.
"""

@app.post("/api/tailor", response_model=TailorResponse)
def tailor_resume(req: TailorRequest):
    role = req.target_title or "the target role"
    prompt = f"""
Target Title: {role}
Tone: {req.tone}
Extra Instructions: {req.instructions}

Job Description:
{req.job_text}

Candidate Resume:
{req.resume_text}

Tasks:
1) Write a 2–3 sentence summary tailored to the job.
2) Provide a short “improved resume notes” paragraph (what changed & why).
3) Draft a one-page cover letter (3–5 short paragraphs).
4) Provide 3–5 sections with improved bullet points (JSON only).
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
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
                                    "bullets": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["heading", "bullets"]
                            }
                        }
                    },
                    "required": ["summary", "improved_resume", "cover_letter", "sections"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        temperature=0.5,
    )

    parsed = response.output_parsed
    return TailorResponse(**parsed)

