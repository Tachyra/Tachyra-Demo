"""
Tachyra Diagnostics - Differential Diagnosis & Medication Interaction Engine
Phase 0 Pilot - Live Anthropic API Integration

This module replaces mocked/stub logic with real calls to the Anthropic API.
Physician retains final clinical authority; this engine only ranks and flags.

Setup:
    pip install anthropic --break-system-packages
    export ANTHROPIC_API_KEY="your-key-here"
"""

import os
import json
import anthropic
from dataclasses import dataclass, field
from typing import Optional

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-5"          # default: best speed/cost balance, near-Opus quality
HIGH_ACUITY_MODEL = "claude-opus-4-8"  # escalate here for complex/high-acuity cases


SYSTEM_PROMPT = """You are a clinical decision support engine used by licensed physicians. \
You do NOT diagnose patients or make treatment decisions. You surface a ranked list of \
possible differential diagnoses and flag medication interactions/dosage concerns for the \
physician to review. The physician retains full clinical authority at all times.

Rules:
- Never state a diagnosis as fact. Always frame as "possible" or "consider."
- Rank differentials by clinical likelihood given the presented data, most likely first.
- For each differential, give a one-line rationale tied to the specific symptoms/history given.
- Flag any medication interactions or dosage concerns explicitly, with severity (low/moderate/high).
- If information is insufficient for a confident ranking, say so rather than guessing.
- Output ONLY valid JSON. No markdown, no preamble, no commentary outside the JSON object.
"""

OUTPUT_SCHEMA_HINT = """
Respond with exactly this JSON shape:
{
  "differentials": [
    {"condition": "string", "likelihood_rank": 1, "rationale": "string"}
  ],
  "medication_flags": [
    {"medications_involved": ["string"], "concern": "string", "severity": "low|moderate|high"}
  ],
  "insufficient_data_note": "string or null"
}
"""


@dataclass
class PatientCase:
    chief_complaint: str
    symptoms: list[str]
    age: Optional[int] = None
    sex: Optional[str] = None
    current_medications: list[str] = field(default_factory=list)
    relevant_history: list[str] = field(default_factory=list)
    vitals: Optional[dict] = None

    def to_prompt(self) -> str:
        parts = [f"Chief complaint: {self.chief_complaint}"]
        if self.age is not None or self.sex:
            parts.append(f"Patient: {self.age or 'unknown age'}, {self.sex or 'sex unspecified'}")
        if self.symptoms:
            parts.append(f"Symptoms: {', '.join(self.symptoms)}")
        if self.current_medications:
            parts.append(f"Current medications: {', '.join(self.current_medications)}")
        if self.relevant_history:
            parts.append(f"Relevant history: {', '.join(self.relevant_history)}")
        if self.vitals:
            parts.append(f"Vitals: {json.dumps(self.vitals)}")
        return "\n".join(parts)


def run_dx_engine(case: PatientCase, high_acuity: bool = False) -> dict:
    """Send a patient case to the model and return structured differential + med-flag JSON.

    Set high_acuity=True to route complex/critical cases to the stronger model.
    """
    model = HIGH_ACUITY_MODEL if high_acuity else MODEL
    user_content = f"{case.to_prompt()}\n\n{OUTPUT_SCHEMA_HINT}"

    response = client.messages.create(
        model=model,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Defensive cleanup in case the model wraps output in code fences despite instructions
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "error": "Failed to parse model output as JSON",
            "raw_output": raw_text,
        }

    result["_meta"] = {
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return result


if __name__ == "__main__":
    # Example pilot test case
    test_case = PatientCase(
        chief_complaint="Progressive fatigue and joint pain over 3 weeks",
        symptoms=["fatigue", "bilateral joint pain", "low-grade fever", "morning stiffness"],
        age=47,
        sex="female",
        current_medications=["lisinopril 10mg daily", "ibuprofen 400mg as needed"],
        relevant_history=["hypertension", "family history of autoimmune disease"],
        vitals={"temp_f": 99.8, "bp": "138/86", "hr": 82},
    )

    output = run_dx_engine(test_case)
    print(json.dumps(output, indent=2))
