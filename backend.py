"""
Tachyra Diagnostics - Physician Demo Backend
Wraps tachyra_dx_engine.py behind a small FastAPI server so the browser-side
demo never touches the API key directly.

Also handles demo access: subscribers get a unique code (generated and
stored server-side, in Postgres) instead of a code hardcoded in page JS.

Physician accounts: physicians self-register (gated by an invite code) with
an email/password, log in to receive a session token, and that token is
required to call /diagnose. Sessions expire after 24 hours.

Registration optionally captures an NPI (National Provider Identifier).
This is NOT required and does not block registration - it's captured for
Barry's own reference/audit trail (e.g. spot-checking against the public
NPPES registry later), not verified automatically at signup time.

Subscription/access model (manually administered - no payment processor
wired up yet):
  - While the free-year promo is active (see the `settings` table /
    /admin/promo-status endpoints below), physicians who register get 365
    days of free access starting at registration.
  - Once the promo is closed, physicians who register get no free access -
    access_expires_at is set to their registration time, so /diagnose is
    paused immediately until an admin grants them access (e.g. once
    they've paid).
  - The promo has no fixed end date in code - it stays open for every new
    signup until Barry explicitly closes it via POST /admin/close-promo.
    This means it can run indefinitely and be shut off on demand, without
    a redeploy.
  - When a physician's access_expires_at passes, /diagnose is blocked with a
    clear "please renew" error - no grace period, no auto-downgrade.
  - An admin (Barry) grants/extends access via POST /admin/grant-access,
    protected by an ADMIN_SECRET the frontend never sees. This is a manual
    stand-in until real billing (e.g. Stripe) is integrated.

Patient records (added Aug 2026): physicians can now create patient
records and attach structured intake data, lab results, imaging reports,
and dated medication entries to them. /diagnose now requires a patient_id,
saves its result permanently (ai_analyses), and feeds that patient's
on-file history back into the AI's reasoning instead of treating every
call as a one-off. New patients default to is_test_patient=True - real
patient data should not be entered until the Anthropic BAA is executed
and hosting is confirmed BAA-covered (see project compliance notes).
SOAP note drafts can be generated from a saved analysis for physician
review/edit/sign-off.
"""
import os
import secrets
import re
import hashlib
import hmac
import json
from datetime import datetime, timezone, timedelta, date

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from tachyra_dx_engine import PatientCase, run_dx_engine, generate_soap_note

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
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url)


def init_db():
    """Create tables if they don't exist yet, and add new columns to existing
    tables if they're missing. Safe to run every startup."""
    if not DATABASE_URL:
        print("WARNING: DATABASE_URL not set. /subscribe, /verify-code, and physician auth will fail until it is.")
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
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS physicians (
                        id SERIAL PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        full_name TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        access_expires_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        pricing_tier TEXT NOT NULL DEFAULT 'unset',
                        committed_years INTEGER,
                        npi_number TEXT
                    );
                    """
                )
                cur.execute("ALTER TABLE physicians ADD COLUMN IF NOT EXISTS access_expires_at TIMESTAMPTZ NOT NULL DEFAULT now();")
                cur.execute("ALTER TABLE physicians ADD COLUMN IF NOT EXISTS pricing_tier TEXT NOT NULL DEFAULT 'unset';")
                cur.execute("ALTER TABLE physicians ADD COLUMN IF NOT EXISTS committed_years INTEGER;")
                cur.execute("ALTER TABLE physicians ADD COLUMN IF NOT EXISTS npi_number TEXT;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS physician_sessions (
                        token TEXT PRIMARY KEY,
                        physician_id INTEGER NOT NULL REFERENCES physicians(id),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        expires_at TIMESTAMPTZ NOT NULL
                    );
                    """
                )
                # Simple key/value settings table. Currently used only for the
                # free-year promo on/off switch, but generic enough to reuse
                # for other admin-toggled settings later.
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    """
                    INSERT INTO settings (key, value)
                    VALUES ('promo_active', 'true')
                    ON CONFLICT (key) DO NOTHING;
                    """
                )

                # ---------------------------------------------------------
                # Patient persistence (added Aug 2026)
                # ---------------------------------------------------------
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patients (
                        id SERIAL PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        date_of_birth DATE,
                        created_by_physician_id INTEGER NOT NULL REFERENCES physicians(id),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        is_test_patient BOOLEAN NOT NULL DEFAULT TRUE
                    );
                    """
                )
                # is_test_patient defaults to TRUE deliberately - real patient
                # data should not be entered until the Anthropic BAA is
                # executed and hosting is confirmed BAA-covered.

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS intake_submissions (
                        id SERIAL PRIMARY KEY,
                        patient_id INTEGER NOT NULL REFERENCES patients(id),
                        chief_complaint TEXT,
                        onset_frequency TEXT,
                        ongoing_conditions TEXT,
                        procedures_surgeries TEXT,
                        current_medications_patient_reported TEXT,
                        drug_allergies TEXT,
                        family_history TEXT,
                        illicit_drug_use TEXT,
                        supplements TEXT,
                        vaccination_history TEXT,
                        clinical_trial_participation TEXT,
                        additional_notes TEXT,
                        submitted_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clinical_data_on_file (
                        id SERIAL PRIMARY KEY,
                        patient_id INTEGER NOT NULL REFERENCES patients(id),
                        data_type TEXT NOT NULL CHECK (data_type IN ('lab_result', 'imaging')),
                        source TEXT NOT NULL DEFAULT 'manual_entry',
                        payload JSONB NOT NULL,
                        date_performed DATE NOT NULL,
                        entered_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                # date_performed is a real, NOT NULL column (not buried in
                # the JSONB payload) so absence of data is enforceable and
                # unambiguous - the direct fix for the AI assuming imaging
                # was normal when it was never actually collected.

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS medication_entries (
                        id SERIAL PRIMARY KEY,
                        patient_id INTEGER NOT NULL REFERENCES patients(id),
                        medication_name TEXT NOT NULL,
                        dose TEXT,
                        frequency TEXT,
                        start_date DATE,
                        stop_date DATE,
                        prescribing_physician TEXT,
                        source TEXT NOT NULL DEFAULT 'manual_entry',
                        entered_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ai_analyses (
                        id SERIAL PRIMARY KEY,
                        patient_id INTEGER NOT NULL REFERENCES patients(id),
                        physician_id INTEGER NOT NULL REFERENCES physicians(id),
                        intake_submission_id INTEGER REFERENCES intake_submissions(id),
                        raw_response JSONB NOT NULL,
                        prompt_version TEXT NOT NULL DEFAULT 'v2',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                # This table is what turns /diagnose from stateless into
                # persistent. prompt_version exists so future prompt
                # changes stay auditable.

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS soap_notes (
                        id SERIAL PRIMARY KEY,
                        analysis_id INTEGER NOT NULL REFERENCES ai_analyses(id),
                        patient_id INTEGER NOT NULL REFERENCES patients(id),
                        physician_id INTEGER NOT NULL REFERENCES physicians(id),
                        subjective TEXT,
                        objective TEXT,
                        assessment TEXT,
                        plan TEXT,
                        ai_generated_draft BOOLEAN NOT NULL DEFAULT TRUE,
                        physician_edited BOOLEAN NOT NULL DEFAULT FALSE,
                        finalized_at TIMESTAMPTZ,
                        generated_at TIMESTAMPTZ NOT NULL DEFAULT now()
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
                cur.execute("SELECT code FROM subscribers WHERE email = %s;", (email,))
                existing = cur.fetchone()
                if existing:
                    return {"code": existing["code"]}

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
# Physician accounts (self-registration + login + subscription access)
# ---------------------------------------------------------------------------

INVITE_CODE = os.environ.get("PHYSICIAN_INVITE_CODE", "tachyra-pilot-2026")
SESSION_HOURS = 24
FREE_YEAR_DAYS = 365

ADMIN_SECRET = os.environ.get("ADMIN_SECRET")  # set this on Render; no default

# Loose format check only - 10 digits, per the NPI spec. This is NOT a
# verification against the NPPES registry, just a sanity check so obviously
# malformed input doesn't get stored. NPI is optional; a blank value always
# passes through untouched.
NPI_PATTERN = re.compile(r"^\d{10}$")


def is_promo_active() -> bool:
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM settings WHERE key = 'promo_active';")
                row = cur.fetchone()
                return row[0].lower() == "true" if row else True
    finally:
        conn.close()


def hash_password(password: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"{salt}${h.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, h = stored_hash.split("$")
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return hmac.compare_digest(check.hex(), h)


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    invite_code: str
    npi_number: Optional[str] = None  # optional, not required to register


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/physician/register")
def register_physician(req: RegisterRequest):
    if req.invite_code != INVITE_CODE:
        raise HTTPException(status_code=403, detail="Invalid invite code.")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    email = req.email.strip().lower()
    if not EMAIL_PATTERN.match(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")

    npi = (req.npi_number or "").strip()
    if npi and not NPI_PATTERN.match(npi):
        raise HTTPException(status_code=400, detail="NPI number should be exactly 10 digits, or left blank.")
    npi = npi or None  # store NULL rather than an empty string

    now = datetime.now(timezone.utc)
    if is_promo_active():
        access_expires_at = now + timedelta(days=FREE_YEAR_DAYS)
        pricing_tier = "free_year_promo"
    else:
        access_expires_at = now
        pricing_tier = "unpaid"

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO physicians
                            (email, password_hash, full_name, access_expires_at, pricing_tier, npi_number)
                        VALUES (%s, %s, %s, %s, %s, %s);
                        """,
                        (email, hash_password(req.password), req.full_name.strip(),
                         access_expires_at, pricing_tier, npi),
                    )
                except psycopg2.errors.UniqueViolation:
                    raise HTTPException(status_code=400, detail="An account with this email already exists.")
    finally:
        conn.close()

    return {
        "status": "registered",
        "pricing_tier": pricing_tier,
        "access_expires_at": access_expires_at.isoformat(),
    }


@app.post("/physician/login")
def login_physician(req: LoginRequest):
    email = req.email.strip().lower()
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, password_hash FROM physicians WHERE email = %s AND is_active = TRUE;",
                    (email,),
                )
                row = cur.fetchone()
                if not row or not verify_password(req.password, row[1]):
                    raise HTTPException(status_code=401, detail="Invalid email or password.")
                physician_id = row[0]
                token = secrets.token_urlsafe(32)
                expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
                cur.execute(
                    "INSERT INTO physician_sessions (token, physician_id, expires_at) VALUES (%s, %s, %s);",
                    (token, physician_id, expires),
                )
    finally:
        conn.close()
    return {"token": token, "expires_in_hours": SESSION_HOURS}


def require_physician(authorization: Optional[str] = Header(None)) -> int:
    """Validates the session token and confirms the physician's paid/free
    access hasn't expired. Returns the physician's id if all checks pass."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    token = authorization.split(" ", 1)[1]

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT physician_id, expires_at FROM physician_sessions WHERE token = %s;",
                    (token,),
                )
                session_row = cur.fetchone()
                if not session_row:
                    raise HTTPException(status_code=401, detail="Session not found. Please log in again.")
                if session_row[1] < datetime.now(timezone.utc):
                    raise HTTPException(status_code=401, detail="Session expired. Please log in again.")

                physician_id = session_row[0]
                cur.execute(
                    "SELECT access_expires_at FROM physicians WHERE id = %s;",
                    (physician_id,),
                )
                access_row = cur.fetchone()
                if not access_row:
                    raise HTTPException(status_code=401, detail="Account not found.")
                if access_row[0] < datetime.now(timezone.utc):
                    raise HTTPException(
                        status_code=402,
                        detail="Your access has expired. Please contact Tachyra Diagnostics to renew.",
                    )
    finally:
        conn.close()
    return physician_id


# ---------------------------------------------------------------------------
# Admin: promo on/off switch, and manually grant/extend a physician's access
# (stand-in until a real payment processor is wired up). Both protected by
# ADMIN_SECRET, not by a physician session - for Barry's use only, never
# called by the frontend UI.
# ---------------------------------------------------------------------------

def _check_admin(x_admin_secret: Optional[str]) -> None:
    if not ADMIN_SECRET or x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Not authorized.")


@app.get("/admin/promo-status")
def promo_status(x_admin_secret: Optional[str] = Header(None)):
    _check_admin(x_admin_secret)
    return {"promo_active": is_promo_active()}


@app.post("/admin/close-promo")
def close_promo(x_admin_secret: Optional[str] = Header(None)):
    """Turns off the free-year promo for all future signups. Physicians who
    already registered under the promo keep their existing access_expires_at
    - this only affects new registrations from this point on."""
    _check_admin(x_admin_secret)
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE settings SET value = 'false', updated_at = now() WHERE key = 'promo_active';"
                )
    finally:
        conn.close()
    return {"status": "promo closed", "promo_active": False}


@app.post("/admin/reopen-promo")
def reopen_promo(x_admin_secret: Optional[str] = Header(None)):
    """Turns the free-year promo back on, in case it's closed and reopened later."""
    _check_admin(x_admin_secret)
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE settings SET value = 'true', updated_at = now() WHERE key = 'promo_active';"
                )
    finally:
        conn.close()
    return {"status": "promo reopened", "promo_active": True}


class GrantAccessRequest(BaseModel):
    email: str
    extend_days: int
    pricing_tier: str
    committed_years: Optional[int] = None


@app.post("/admin/grant-access")
def grant_access(req: GrantAccessRequest, x_admin_secret: Optional[str] = Header(None)):
    _check_admin(x_admin_secret)

    email = req.email.strip().lower()
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT access_expires_at FROM physicians WHERE email = %s;", (email,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="No physician with that email.")

                base = max(row[0], datetime.now(timezone.utc))
                new_expiry = base + timedelta(days=req.extend_days)

                cur.execute(
                    """
                    UPDATE physicians
                    SET access_expires_at = %s, pricing_tier = %s, committed_years = %s
                    WHERE email = %s;
                    """,
                    (new_expiry, req.pricing_tier, req.committed_years, email),
                )
    finally:
        conn.close()

    return {"status": "updated", "email": email, "new_access_expires_at": new_expiry.isoformat()}


# ---------------------------------------------------------------------------
# Patients: create, retrieve, list
# ---------------------------------------------------------------------------

class CreatePatientRequest(BaseModel):
    display_name: str
    date_of_birth: Optional[date] = None
    is_test_patient: bool = True  # default TRUE - real patients require BAA in place first


@app.post("/patient")
def create_patient(req: CreatePatientRequest, physician_id: int = Depends(require_physician)):
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO patients (display_name, date_of_birth, created_by_physician_id, is_test_patient)
                    VALUES (%s, %s, %s, %s) RETURNING id;
                    """,
                    (req.display_name.strip(), req.date_of_birth, physician_id, req.is_test_patient),
                )
                patient_id = cur.fetchone()[0]
    finally:
        conn.close()
    return {"id": patient_id, "display_name": req.display_name, "is_test_patient": req.is_test_patient}


@app.get("/patient/{patient_id}")
def get_patient(patient_id: int, physician_id: int = Depends(require_physician)):
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM patients WHERE id = %s;", (patient_id,))
                patient = cur.fetchone()
                if not patient:
                    raise HTTPException(status_code=404, detail="Patient not found.")
    finally:
        conn.close()
    return dict(patient)


@app.get("/physician/patients")
def list_my_patients(physician_id: int = Depends(require_physician)):
    """Patients created by the logged-in physician. Simple ownership model
    for now - worth revisiting once more than one physician needs to share
    a patient record."""
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM patients WHERE created_by_physician_id = %s ORDER BY created_at DESC;",
                    (physician_id,),
                )
                patients = cur.fetchall()
    finally:
        conn.close()
    return [dict(p) for p in patients]


# ---------------------------------------------------------------------------
# Intake submissions
# ---------------------------------------------------------------------------

class IntakeRequest(BaseModel):
    chief_complaint: Optional[str] = None
    onset_frequency: Optional[str] = None
    ongoing_conditions: Optional[str] = None
    procedures_surgeries: Optional[str] = None  # "Surgical History & Procedures" in UI
    current_medications_patient_reported: Optional[str] = None
    drug_allergies: Optional[str] = None
    family_history: Optional[str] = None
    illicit_drug_use: Optional[str] = None
    supplements: Optional[str] = None
    vaccination_history: Optional[str] = None
    clinical_trial_participation: Optional[str] = None
    additional_notes: Optional[str] = None


@app.post("/patient/{patient_id}/intake")
def submit_intake(patient_id: int, req: IntakeRequest, physician_id: int = Depends(require_physician)):
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM patients WHERE id = %s;", (patient_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Patient not found.")
                cur.execute(
                    """
                    INSERT INTO intake_submissions
                        (patient_id, chief_complaint, onset_frequency, ongoing_conditions,
                         procedures_surgeries, current_medications_patient_reported,
                         drug_allergies, family_history, illicit_drug_use, supplements,
                         vaccination_history, clinical_trial_participation, additional_notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (patient_id, req.chief_complaint, req.onset_frequency, req.ongoing_conditions,
                     req.procedures_surgeries, req.current_medications_patient_reported,
                     req.drug_allergies, req.family_history, req.illicit_drug_use, req.supplements,
                     req.vaccination_history, req.clinical_trial_participation, req.additional_notes),
                )
                intake_id = cur.fetchone()[0]
    finally:
        conn.close()
    return {"id": intake_id, "status": "saved"}


# ---------------------------------------------------------------------------
# Lab results / imaging (date_performed required - enforced at the DB
# level via NOT NULL on the column)
# ---------------------------------------------------------------------------

class LabResultRequest(BaseModel):
    test_name: str
    result: str
    date_performed: date
    reference_range: Optional[str] = None
    flag: Optional[str] = None  # "normal" | "high" | "low" | "critical"
    ordering_physician: Optional[str] = None


class ImagingReportRequest(BaseModel):
    modality: str
    findings: str
    date_performed: date
    region: Optional[str] = None
    reading_radiologist: Optional[str] = None


def _save_clinical_data(patient_id: int, data_type: str, payload: dict, date_performed: date):
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM patients WHERE id = %s;", (patient_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Patient not found.")
                cur.execute(
                    """
                    INSERT INTO clinical_data_on_file (patient_id, data_type, payload, date_performed)
                    VALUES (%s, %s, %s, %s) RETURNING id;
                    """,
                    (patient_id, data_type, json.dumps(payload), date_performed),
                )
                row_id = cur.fetchone()[0]
    finally:
        conn.close()
    return row_id


@app.post("/patient/{patient_id}/lab")
def submit_lab_result(patient_id: int, req: LabResultRequest, physician_id: int = Depends(require_physician)):
    payload = {
        "test_name": req.test_name,
        "result": req.result,
        "reference_range": req.reference_range,
        "flag": req.flag,
        "ordering_physician": req.ordering_physician,
    }
    row_id = _save_clinical_data(patient_id, "lab_result", payload, req.date_performed)
    return {"id": row_id, "status": "saved"}


@app.post("/patient/{patient_id}/imaging")
def submit_imaging_report(patient_id: int, req: ImagingReportRequest, physician_id: int = Depends(require_physician)):
    payload = {
        "modality": req.modality,
        "region": req.region,
        "findings": req.findings,
        "reading_radiologist": req.reading_radiologist,
    }
    row_id = _save_clinical_data(patient_id, "imaging", payload, req.date_performed)
    return {"id": row_id, "status": "saved"}


# ---------------------------------------------------------------------------
# Medications (structured, with start/stop dates)
# ---------------------------------------------------------------------------

class MedicationEntryRequest(BaseModel):
    medication_name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[date] = None
    stop_date: Optional[date] = None
    prescribing_physician: Optional[str] = None


@app.post("/patient/{patient_id}/medication")
def submit_medication_entry(patient_id: int, req: MedicationEntryRequest, physician_id: int = Depends(require_physician)):
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM patients WHERE id = %s;", (patient_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Patient not found.")
                cur.execute(
                    """
                    INSERT INTO medication_entries
                        (patient_id, medication_name, dose, frequency, start_date, stop_date, prescribing_physician)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
                    """,
                    (patient_id, req.medication_name, req.dose, req.frequency,
                     req.start_date, req.stop_date, req.prescribing_physician),
                )
                row_id = cur.fetchone()[0]
    finally:
        conn.close()
    return {"id": row_id, "status": "saved"}


# ---------------------------------------------------------------------------
# Helper: pull everything on file for a patient, for feeding into /diagnose
# ---------------------------------------------------------------------------

def get_patient_clinical_picture(patient_id: int) -> dict:
    """Gathers all on-file data for a patient: latest intake, labs,
    imaging, and current/past medications. This is what makes /diagnose
    aware of prior history instead of only the current request."""
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM intake_submissions WHERE patient_id = %s ORDER BY submitted_at DESC LIMIT 1;",
                    (patient_id,),
                )
                intake = cur.fetchone()

                cur.execute(
                    "SELECT * FROM clinical_data_on_file WHERE patient_id = %s AND data_type = 'lab_result' ORDER BY date_performed DESC;",
                    (patient_id,),
                )
                labs = cur.fetchall()

                cur.execute(
                    "SELECT * FROM clinical_data_on_file WHERE patient_id = %s AND data_type = 'imaging' ORDER BY date_performed DESC;",
                    (patient_id,),
                )
                imaging = cur.fetchall()

                cur.execute(
                    "SELECT * FROM medication_entries WHERE patient_id = %s ORDER BY start_date DESC;",
                    (patient_id,),
                )
                medications = cur.fetchall()
    finally:
        conn.close()

    return {
        "intake": dict(intake) if intake else None,
        "labs": [dict(r) for r in labs],
        "imaging": [dict(r) for r in imaging],
        "medications": [dict(r) for r in medications],
    }


# ---------------------------------------------------------------------------
# Existing diagnose endpoint - now requires a patient_id, saves its result,
# and feeds that patient's on-file history into the AI's reasoning.
# ---------------------------------------------------------------------------

class CaseRequest(BaseModel):
    patient_id: int
    chief_complaint: str
    symptoms: list[str]
    age: Optional[int] = None
    sex: Optional[str] = None
    current_medications: list[str] = []
    relevant_history: list[str] = []
    vitals: Optional[dict] = None
    high_acuity: bool = False

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

    attached_files: list[dict] = []


def build_vitals_dict(req: CaseRequest) -> Optional[dict]:
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
MAX_TOTAL_BASE64_CHARS = 20_000_000
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
def diagnose(req: CaseRequest, physician_id: int = Depends(require_physician)):
    conn_check = get_db_connection()
    try:
        with conn_check:
            with conn_check.cursor() as cur:
                cur.execute("SELECT id FROM patients WHERE id = %s;", (req.patient_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Patient not found.")
    finally:
        conn_check.close()

    try:
        validate_attached_files(req.attached_files)

        # Pull prior history and merge it in - this is what makes the
        # difference between "assumed normal" and "flagged as missing."
        clinical_picture = get_patient_clinical_picture(req.patient_id)

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
        result = run_dx_engine(case, high_acuity=req.high_acuity, clinical_picture=clinical_picture)

        # Save the result - this is the persistence piece that didn't
        # exist before.
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    intake_id = clinical_picture["intake"]["id"] if clinical_picture["intake"] else None
                    cur.execute(
                        """
                        INSERT INTO ai_analyses (patient_id, physician_id, intake_submission_id, raw_response)
                        VALUES (%s, %s, %s, %s) RETURNING id;
                        """,
                        (req.patient_id, physician_id, intake_id, json.dumps(result)),
                    )
                    analysis_id = cur.fetchone()[0]
        finally:
            conn.close()

        result["analysis_id"] = analysis_id
        return result
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# SOAP note generation - drafts from an existing saved analysis.
# ---------------------------------------------------------------------------

class SoapNoteEditRequest(BaseModel):
    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    finalize: bool = False


@app.post("/analysis/{analysis_id}/soap-note")
def create_soap_note(analysis_id: int, physician_id: int = Depends(require_physician)):
    """Generates a first-draft SOAP note from an existing saved analysis.
    This is a DRAFT ONLY - the physician must review, edit as needed, and
    finalize before it's treated as a real clinical document."""
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM ai_analyses WHERE id = %s;", (analysis_id,))
                analysis = cur.fetchone()
                if not analysis:
                    raise HTTPException(status_code=404, detail="Analysis not found.")
    finally:
        conn.close()

    patient_id = analysis["patient_id"]
    clinical_picture = get_patient_clinical_picture(patient_id)
    ai_analysis = analysis["raw_response"]

    soap_json = generate_soap_note(
        intake=clinical_picture["intake"],
        labs=clinical_picture["labs"],
        imaging=clinical_picture["imaging"],
        medications=clinical_picture["medications"],
        ai_analysis=ai_analysis,
    )

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO soap_notes
                        (analysis_id, patient_id, physician_id, subjective, objective, assessment, plan)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
                    """,
                    (analysis_id, patient_id, physician_id,
                     soap_json.get("subjective", ""), soap_json.get("objective", ""),
                     soap_json.get("assessment", ""), soap_json.get("plan", "")),
                )
                soap_id = cur.fetchone()[0]
    finally:
        conn.close()

    return {"id": soap_id, **soap_json, "ai_generated_draft": True}


@app.patch("/soap-note/{soap_note_id}")
def edit_soap_note(soap_note_id: int, req: SoapNoteEditRequest, physician_id: int = Depends(require_physician)):
    fields, values = [], []
    for col, val in [("subjective", req.subjective), ("objective", req.objective),
                      ("assessment", req.assessment), ("plan", req.plan)]:
        if val is not None:
            fields.append(f"{col} = %s")
            values.append(val)
    if fields:
        fields.append("physician_edited = %s")
        values.append(True)
    if req.finalize:
        fields.append("finalized_at = now()")
    if not fields:
        return {"status": "no changes"}

    values.append(soap_note_id)
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE soap_notes SET {', '.join(fields)} WHERE id = %s;",
                    values,
                )
    finally:
        conn.close()
    return {"status": "updated"}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
