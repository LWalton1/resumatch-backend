from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ResuMatch.ai", version="1.0")

@app.get("/")
def root():
    return {"status": "ok", "message": "ResuMatch backend is running"}

# ...followed by your TailorRequest, TailorResponse, and tailor_resume() code

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "ResuMatch backend is running"}
from pydantic import BaseModel
from typing import Optional

class TailorRequest(BaseModel):
    resume_text: str
    job_text: str
    target_title: Optional[str] = None
    tone: Optional[str] = "Professional"
    instructions: Optional[str] = ""
    generate_cover_letter: Optional[bool] = True

class TailorResponse(BaseModel):
    summary: str
    improved_resume: str
    cover_letter: Optional[str]

@app.post("/api/tailor", response_model=TailorResponse)
def tailor_resume(req: TailorRequest):
    resume = req.resume_text.lower()
    job = req.job_text.lower()
    common = [w for w in job.split() if w in resume]
    summary = f"Tailored resume for {req.target_title or 'target role'} with tone '{req.tone}'."
    improved = f"Optimized resume includes {len(common)} matched job keywords."
    cover = None
    if req.generate_cover_letter:
        cover = (
            f"Dear Hiring Manager,\n\nI am excited to apply for the {req.target_title or 'role'}. "
            f"My experience aligns with your requirements, especially in {', '.join(common[:10])}.\n\n"
            f"Sincerely,\nYour Name"
        )
    return TailorResponse(summary=summary, improved_resume=improved, cover_letter=cover)
    
