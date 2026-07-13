"""
Tachyra Diagnostics - Physician Demo Backend
Wraps tachyra_dx_engine.py behind a small FastAPI server so the browser-side
demo never touches the API key directly.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from tachyra_dx_engine import PatientCase, run_dx_engine

app = FastAPI(title="Tachyra Diagnostics - Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CaseRequest(BaseModel):
    chief_complaint: str
    symptoms: list[str]
    age: Optional[int] = None
    sex: Optional[str] = None
    current_medications: list[str] = []
    relevant_history: list[str] = []
    vitals: Optional[dict] = None
    high_acuity: bool = False

    # Expanded intake fields (previously collected by the form but dropped here)
    submitted_by: Optional[str] = None
    signs: list[str] = []
    onset: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    aggravating_factors: Optional[str] = None
    alleviating_factors: Optional[str] = None
    review_of_systems: list[str] = []
    temp_f: Optional[float] = None
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    resp_rate: Optional[int] = None
    o2_sat: Optional[int] = None
    physical_exam_findings: Optional[str] = None
    labs_imaging: Optional[str] = None
    allergies: Optional[str] = None
    family_history: Optional[str] = None
    social_history: Optional[str] = None

    # Attached diagnostic test files: each {"filename": str, "media_type": str, "data": base64 str}
    attached_files: list[dict] = []


def build_vitals_dict(req: CaseRequest) -> Optional[dict]:
    """Combine discrete vitals fields (and any legacy raw `vitals` dict) into one dict."""
    vitals = dict(req.vitals) if req.vitals else {}
    if req.temp_f is not None:
        vitals["temp_f"] = req.temp_f
    if req.bp_systolic is not None and req.bp_diastolic is not None:
        vitals["bp"] = f"{req.bp_systolic}/{req.bp_diastolic}"
    if req.heart_rate is not None:
        vitals["hr"] = req.heart_rate
    if req.resp_rate is not None:
        vitals["rr"] = req.resp_rate
    if req.o2_sat is not None:
        vitals["o2_sat"] = req.o2_sat
    return vitals or None


MAX_FILES = 5
MAX_TOTAL_BASE64_CHARS = 20_000_000  # roughly ~15MB of actual file data
ALLOWED_MEDIA_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/webp", "image/gif"}


def validate_attached_files(files: list[dict]) -> None:
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files attached (max {MAX_FILES}).")
    total_chars = 0
    for f in files:
        media_type = f.get("media_type", "")
        if media_type not in ALLOWED_MEDIA_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {media_type}")
        total_chars += len(f.get("data", ""))
    if total_chars > MAX_TOTAL_BASE64_CHARS:
        raise HTTPException(status_code=400, detail="Attached files are too large in total. Please reduce file sizes or count.")


@app.post("/diagnose")
def diagnose(req: CaseRequest):
    try:
        validate_attached_files(req.attached_files)
        case = PatientCase(
            chief_complaint=req.chief_complaint,
            symptoms=req.symptoms,
            age=req.age,
            sex=req.sex,
            current_medications=req.current_medications,
            relevant_history=req.relevant_history,
            vitals=build_vitals_dict(req),
            submitted_by=req.submitted_by,
            signs=req.signs,
            onset=req.onset,
            duration=req.duration,
            severity=req.severity,
            aggravating_factors=req.aggravating_factors,
            alleviating_factors=req.alleviating_factors,
            review_of_systems=req.review_of_systems,
            physical_exam_findings=req.physical_exam_findings,
            labs_imaging=req.labs_imaging,
            allergies=req.allergies,
            family_history=req.family_history,
            social_history=req.social_history,
            attached_files=req.attached_files,
        )
        result = run_dx_engine(case, high_acuity=req.high_acuity)
        return result
    except Exception as e:
        print("ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
