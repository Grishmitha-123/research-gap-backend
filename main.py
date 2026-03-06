from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import pdfplumber
import requests
import json
import os
import re
import shutil
import uuid

# ---------------------------------------------------------------------------
# Constants and Persistence
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backend")
os.makedirs(BACKEND_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(BACKEND_DIR, "history.json")
COMMUNITY_FILE = os.path.join(BACKEND_DIR, "community.json")
COMMUNITY_UPLOADS = os.path.join(BACKEND_DIR, "uploads", "community")

os.makedirs(COMMUNITY_UPLOADS, exist_ok=True)

def _load_json(filepath: str) -> List[dict]:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def _save_json(filepath: str, items: List[dict]):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

# ---------------------------------------------------------------------------
# Load API key
# ---------------------------------------------------------------------------
def _load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key: return key
    
    # Search root and parent (if running from backend/)
    search_dirs = [os.getcwd(), os.path.dirname(__file__), os.path.dirname(os.path.dirname(__file__))]
    for d in search_dirs:
      api_txt = os.path.join(d, "api.txt")
      if os.path.exists(api_txt):
          try:
            with open(api_txt, "r") as f:
                key = f.read().strip()
                if key: 
                    print(f"DEBUG: Found API Key in {api_txt}")
                    return key
          except: continue
    print("WARNING: No API Key found in environment or api.txt")
    return ""

OPENROUTER_API_KEY = _load_api_key()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AnalysisResponse(BaseModel):
    id: str
    problem: str
    methodology: str
    contributions: str
    limitations: str
    gaps: str
    future: str
    novelty_score: Optional[int] = None
    impact_level: Optional[str] = None
    domain: Optional[str] = None
    filename: Optional[str] = None
    timestamp: Optional[str] = None

class GeneratorRequest(BaseModel):
    title: str
    domain: str
    problem: str
    methodology: str
    findings: str

class GeneratorResponse(BaseModel):
    title: str
    abstract: str
    introduction: str
    methodology: str
    results: str
    conclusion: str
    references: List[str]

class CommunityItem(BaseModel):
    id: str
    filename: str
    uploader_name: str
    uploader_bio: Optional[str] = ""
    uploader_affiliation: Optional[str] = ""
    uploader_linkedin: Optional[str] = ""
    timestamp: str
    domain: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try: return json.loads(text[start:end])
        except: pass
    return None

def _call_llm_with_fallback(prompt: str):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API Key")
    
    models = [
        "openrouter/free", # Recommended: Auto-selects active free model
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-3-12b-it:free",
    ]
    
    for model in models:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://researchgap.ai",
                    "X-Title": "ResearchGap",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                },
                timeout=30
            )
            res_json = response.json()
            if "error" in res_json: continue
            
            content = res_json["choices"][0]["message"]["content"]
            result = _extract_json(content)
            if result: return result
        except: continue
            
    raise HTTPException(status_code=500, detail="AI Service Busy")

# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "ResearchGap AI backend ready"}

@app.get("/history", response_model=List[AnalysisResponse])
async def get_history():
    return _load_json(HISTORY_FILE)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_pdf(file: UploadFile = File(...)):
    pdf_text = ""
    try:
        with pdfplumber.open(file.file) as pdf:
            pdf_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    except:
        raise HTTPException(status_code=400, detail="Invalid PDF")

    prompt = (
        "Return EXCLUSIVELY a JSON object with keys: "
        "problem, methodology, contributions, limitations, research_gaps, future_directions, "
        "novelty_score (1-10), impact_level (High/Medium/Low), domain.\n"
        f"Content: {pdf_text[:3000]}"
    )

    data = _call_llm_with_fallback(prompt)
    
    analysis = AnalysisResponse(
        id=str(uuid.uuid4()),
        problem=data.get("problem", "N/A"),
        methodology=data.get("methodology", "N/A"),
        contributions=data.get("contributions", "N/A"),
        limitations=data.get("limitations", "N/A"),
        gaps=data.get("research_gaps", "N/A"),
        future=data.get("future_directions", "N/A"),
        novelty_score=data.get("novelty_score"),
        impact_level=data.get("impact_level"),
        domain=data.get("domain", "General"),
        filename=file.filename,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    hist = _load_json(HISTORY_FILE)
    hist.insert(0, analysis.dict())
    _save_json(HISTORY_FILE, hist)
    return analysis

@app.post("/generate-paper", response_model=GeneratorResponse)
async def generate_paper(req: GeneratorRequest):
    prompt = (
        "Return EXCLUSIVELY a JSON object with keys: title, abstract, introduction, methodology, results, conclusion, references (list).\n"
        "Each section (abstract, introduction, methodology, results, conclusion) MUST be extremely comprehensive, multi-paragraph, technical, and verbose to provide professional academic depth.\n"
        f"Title: {req.title}\nDomain: {req.domain}\nProblem: {req.problem}\nMethodology: {req.methodology}\nFindings: {req.findings}"
    )
    data = _call_llm_with_fallback(prompt)
    return GeneratorResponse(**data)

@app.get("/community/list", response_model=List[CommunityItem])
async def list_papers():
    return _load_json(COMMUNITY_FILE)

@app.post("/community/upload")
async def upload_paper(
    uploader_name: str, 
    domain: str, 
    file: UploadFile = File(...)
):
    safe_filename = re.sub(r'[^a-zA-Z0-9.-]', '_', file.filename)
    path = os.path.join(COMMUNITY_UPLOADS, safe_filename)
    with open(path, "wb") as f: shutil.copyfileobj(file.file, f)
    
    item = CommunityItem(
        id=str(uuid.uuid4()),
        filename=safe_filename,
        uploader_name=uploader_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        domain=domain
    )
    comm = _load_json(COMMUNITY_FILE)
    comm.insert(0, item.dict())
    _save_json(COMMUNITY_FILE, comm)
    return item

@app.delete("/history/{item_id}")
async def delete_history(item_id: str):
    hist = _load_json(HISTORY_FILE)
    new_hist = [i for i in hist if i.get("id") != item_id]
    if len(new_hist) == len(hist):
        raise HTTPException(status_code=404, detail="Item not found")
    _save_json(HISTORY_FILE, new_hist)
    return {"message": "Success"}

@app.delete("/community/{item_id}")
async def delete_community(item_id: str):
    comm = _load_json(COMMUNITY_FILE)
    item = next((i for i in comm if i.get("id") == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    new_comm = [i for i in comm if i.get("id") != item_id]
    _save_json(COMMUNITY_FILE, new_comm)
    
    # Optional: Delete physical file
    try:
        file_path = os.path.join(COMMUNITY_UPLOADS, item["filename"])
        if os.path.exists(file_path):
            os.remove(file_path)
    except: pass
    
    return {"message": "Success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)