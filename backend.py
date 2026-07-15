"""
Tachyra Diagnostics - Physician Demo Backend
Wraps tachyra_dx_engine.py behind a small FastAPI server so the browser-side
demo never touches the API key directly.

Also handles demo access: subscribers get a unique code (generated and
stored server-side, in Postgres) instead of a code hardcoded in page JS.
"""
import os
import secrets
import re
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
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

# ---------------------------------------------------------------------------
# Database setup (Render Postgres)
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_db_connection():
    if not DATABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL is not configured on the server.",
        )
    # Render's internal URL sometimes uses postgres:// which psycopg2 accepts fine,
    # but we normalize just in case something upstream expects postgresql://
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url)


def init_db():
    """Create the subscribers table if it doesn't exist yet. Safe to run every startup."""
    if not DATABASE_URL:
        print("WARNING: DATABASE_URL not set. /subscribe and /verify-code will fail until it is.")
        return
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subscribers (
                        id SERIAL PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        code TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        last_used_at TIMESTAMPTZ,
                        use_count INTEGER NOT NULL DEFAULT 0
                    );
                    """
                )
    finally:
        conn.close()


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Subscribe / verify-code endpoints
# ---------------------------------------------------------------------------

EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def generate_code() -> str:
    return "TCH-" + secrets.token_hex(4).upper()


class SubscribeRequest(BaseModel):
    email: str


class SubscribeResponse(BaseModel):
    code: str


class VerifyCodeRequest(BaseModel):
    code: str


class VerifyCodeResponse(BaseModel):
    valid: bool


@app.post("/subscribe", response_model=SubscribeResponse)
def subscribe(req: SubscribeRequest):
    email = req.email.strip().lower()
    if not EMAIL_PATTERN.match(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # If this email already subscribed, return their existing code
                # instead of creating a duplicate one.
                cur.execute("SELECT code FROM subscribers WHERE email = %s;", (email,))
                existing = cur.fetchone()
                if existing:
                    return {"code": existing["code"]}

                # Generate a unique code, retrying on the rare collision.
                for _ in range(5):
                    code = generate_code()
                    try:
                        cur.execute(
                            "INSERT INTO subscribers (email, code) VALUES (%s, %s);",
                            (email, code),
                        )
                        return {"code": code}
                    except psycopg2.errors.UniqueViolation:
                        conn.rollback()
                        continue
                raise HTTPException(status_code=500, detail="Could not generate a unique code. Please try again.")
    finally:
        conn.close()


@app.post("/verify-code", response_model=VerifyCodeResponse)
def verify_code(req: VerifyCodeRequest):
    code = req.code.strip().upper()
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM subscribers WHERE code = %s;", (code,))
                row = cur.fetchone()
                if not row:
                    return {"valid": False}
                cur.execute(
                    """
                    UPDATE subscribers
                    SET last_used_at = %s, use_count = use_count + 1
                    WHERE id = %s;
                    """,
                    (datetime.now(timezone.utc), row[0]),
                )
                return {"valid": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Existing diagnose endpoint (unchanged)
# ---------------------------------------------------------------------------

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
