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


@app.post("/diagnose")
def diagnose(req: CaseRequest):
    try:
        case = PatientCase(
            chief_complaint=req.chief_complaint,
            symptoms=req.symptoms,
            age=req.age,
            sex=req.sex,
            current_medications=req.current_medications,
            relevant_history=req.relevant_history,
            vitals=req.vitals,
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
