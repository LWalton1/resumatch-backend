from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ResuMatch.ai", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "ResuMatch backend is running"}
