"""
Tachyra Diagnostics - Differential Diagnosis & Medication Interaction Engine
Phase 0 Pilot - Live Anthropic API Integration

This module replaces mocked/stub logic with real calls to the Anthropic API.
Physician retains final clinical authority; this engine only ranks and flags.

Setup:
    pip install anthropic requests --break-system-packages
    export ANTHROPIC_API_KEY="your-key-here"
"""

import os
import re
import json
import anthropic
import requests
from dataclasses import dataclass, field
from typing import Optional
from json_repair import repair_json

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-5"          # default: best speed/cost balance, near-Opus quality
HIGH_ACUITY_MODEL = "claude-opus-4-8"  # escalate here for complex/high-acuity cases

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"


def fetch_drug_facts(drug_name: str, timeout: float = 4.0) -> Optional[dict]:
    """Look up verified FDA label data for a medication via the public openFDA API.

    Best-effort only: this is a live, real-time lookup against actual FDA label
    filings (active/inactive ingredients, warnings, documented drug interactions,
    adverse reactions) rather than relying solely on the model's trained knowledge.
    Any failure (drug not found, network issue, rate limit) returns None and must
    never block the diagnostic engine - the model falls back to its own knowledge
    for that medication and the physician is told which medications lacked
    verified label data.
    """
    # Strip dosage-form descriptors the RxNorm/RxTerms autocomplete appends
    # to selected medication names (e.g. "ELIQUIS (Oral Pill)"), and the
    # " — Frequency" suffix this app appends when a dosing frequency was
    # selected - neither exists in FDA's brand_name/generic_name fields, so
    # leaving either in silently breaks lookups for real medications.
    without_freq = drug_name.split('—')[0]
    without_form = re.sub(r'\s*\([^)]*\)\s*', ' ', without_freq)
    name_only = re.split(r'\d', without_form)[0].strip().rstrip('.,').strip()
    if not name_only:
        return None

    query = (
        f'openfda.brand_name:"{name_only}" '
        f'OR openfda.generic_name:"{name_only}" '
        f'OR openfda.substance_name:"{name_only}"'
    )
    try:
        resp = requests.get(
            OPENFDA_LABEL_URL,
            params={"search": query, "limit": 1},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results")
        if not results:
            return None
        label = results[0]

        def first_or_none(field_name: str, max_chars: Optional[int] = 700) -> Optional[str]:
            val = label.get(field_name)
            if not val:
                return None
            return val[0][:max_chars] if max_chars is not None else val[0]

        facts = {
            "queried_as": name_only,
            "active_ingredient": first_or_none("active_ingredient"),
            "inactive_ingredient": first_or_none("inactive_ingredient", max_chars=None),
            "warnings": first_or_none("boxed_warning") or first_or_none("warnings"),
            "drug_interactions": first_or_none("drug_interactions"),
            "adverse_reactions": first_or_none("adverse_reactions"),
        }
        # Only return if we actually found at least one useful field
        if any(v for k, v in facts.items() if k != "queried_as"):
            return facts
        return None
    except Exception:
        return None


SYSTEM_PROMPT = """You are a clinical decision support engine used by licensed physicians. \
You do NOT diagnose patients or make treatment decisions. You surface a ranked list of \
possible differential diagnoses and flag medication interactions/dosage concerns for the \
physician to review. The physician retains full clinical authority at all times.

You may also receive attached diagnostic test files (e.g. EKG tracings, echocardiogram reports, \
lab reports, imaging studies) as documents or images alongside the case text. Review any attached \
files carefully and incorporate their findings into your differentials and flags exactly as you \
would findings typed into the case text - cite specific findings from the attachment (e.g. "EKG \
shows sinus tachycardia" or "echo report notes reduced ejection fraction") where relevant. If an \
attachment is illegible, ambiguous, or you are not confident in reading it, say so explicitly in \
insufficient_data_note rather than guessing at its contents.

You may also receive a "Verified FDA label data" section below the case details, containing \
real-time lookups from the official FDA label database for some of the patient's medications. \
Where present, treat this data as authoritative ground truth and prioritize it over your own \
general knowledge for that medication - especially for active/inactive ingredients, warnings, \
and documented interactions. If a medication has no verified label data available, rely on your \
general clinical knowledge for it, but note in insufficient_data_note that verified label data \
was unavailable for that specific medication so the physician knows which flags rest on \
verified data versus general knowledge.

You may also receive an "ON-FILE PATIENT HISTORY" section, containing this patient's data from \
prior visits (labs, imaging, structured medication history, surgical history, and other intake \
fields), clearly separated from what's reported for today's visit above it. Treat this as real \
prior clinical history, not as part of today's presentation, but factor it into your differentials \
and flags the same way you would a chart review before seeing a patient.

Rules:
- Never state a diagnosis as fact. Always frame as "possible" or "consider."
- Rank differentials by clinical likelihood given the presented data, most likely first.
- For each differential, give a one-line rationale tied to the specific symptoms/history given.
- Flag any medication interactions or dosage concerns explicitly, with severity (low/moderate/high).
- Be thorough rather than conservative when flagging: include documented interactions and side \
effects even if less common or lower severity, not only the most well-known ones. Do not omit a \
documented concern merely because it is rare - instead, flag it and note its rarity/severity \
honestly.
- Every flag must be grounded in documented pharmacological evidence (e.g. FDA labeling, \
peer-reviewed literature, established drug references). Never flag a concern based on \
speculation, anecdote, or unverified belief. If evidence for a concern is limited, \
post-marketing, or observational rather than well-established, say so explicitly rather than \
presenting it with unwarranted certainty - but still include it rather than omitting it.
- For EVERY reported symptom, check it against EVERY current medication for a documented side \
effect link, even if no other diagnosis is otherwise indicated. Report each symptom-medication \
pair separately from drug-drug interactions and dosage concerns - do not omit this check even \
if it seems obvious or well known.
- For each symptom-medication pair, classify the correlation as exactly one of:
  "definite" - the symptom is a well-established, commonly documented side effect of that medication.
  "possible" - the symptom is documented but uncommon, or evidence is limited/post-marketing/observational.
  "no_issue" - no documented link exists between that symptom and that medication.
- Include "no_issue" pairs in the output too, not just definite/possible ones, so the physician \
can see what was checked and ruled out, not only what was flagged.
- Check the patient's current medications against EACH OTHER for ingredient duplication - cases \
where two or more medications share the same ACTIVE ingredient (e.g. a branded and generic \
product containing the same active ingredient, or two different products both containing \
acetaminophen), or the same therapeutic class such that taking them together provides no added \
benefit and raises the risk of unintentional overdose or additive effects. ALSO separately check \
for shared INACTIVE ingredients (excipients, fillers, dyes, preservatives, common allergens such \
as lactose or gluten) that could matter given the patient's documented allergies or sensitivities. \
Flag each such duplication explicitly, grounded in documented pharmacology, and label whether it \
is an active-ingredient or inactive-ingredient duplication.
- Separately from ingredient duplication, check the patient's current medications for drugs that \
are documented to RAISE serum potassium indirectly, even though potassium is not an ingredient - \
including but not limited to ACE inhibitors, ARBs, potassium-sparing diuretics (e.g. \
spironolactone, amiloride, triamterene), and NSAIDs. If the patient is also on a potassium \
supplement or any potassium-containing product, flag this combination explicitly as a \
hyperkalemia risk, grounded in documented pharmacology, and state the mechanism (e.g. "reduced \
renal potassium excretion") in the concern field. Elevate the severity when the patient has \
cardiac history, given the arrhythmia risk of hyperkalemia in that population, and state that \
elevated-risk rationale explicitly when it applies.
- For each current medication where the case includes a specific dose, check whether that dose \
appears high or atypical GIVEN THIS PATIENT'S documented age, weight, and renal/hepatic history \
(when provided). Flag this as a dosage concern with the clinical basis stated (e.g. "reduced \
renal clearance in patients over 65"). Do NOT suggest a corrected or alternative dose - flag the \
concern only; the physician determines any dosing change.
- Separately, assess the patient's FULL medication list together (not just pairwise) for \
cumulative/additive burden - documented concepts such as total sedative load, total \
anticholinergic burden, or cumulative CNS depression risk from multiple drugs each contributing \
a small amount. This is especially relevant for patients on several concurrent medications. \
Ground this in established clinical pharmacology concepts (e.g. anticholinergic burden scales, \
Beers Criteria-type reasoning for polypharmacy), not speculation. Report the total number of \
current medications and any such cumulative concern found, even if no single pairwise \
interaction was flagged.
- If information is insufficient for a confident ranking, say so rather than guessing.
- When your top-ranked differential is reasonably well-supported by the case (not highly uncertain) \
AND represents a condition with well-established, guideline-recognized first-line treatment \
approaches, include treatment_options: a short list of options a physician might consider for \
that specific condition, each with case-specific rationale. Frame every option as something "to \
consider," never as an instruction or a recommendation to prescribe - the physician decides \
whether, when, and how to treat. Include non-pharmacological options where clinically relevant, \
not only medications. NEVER state a specific starting dose or titration schedule for a \
treatment-naive patient - name the medication or intervention and its rationale only; dosing is \
the physician's determination, same as with existing medications elsewhere in this schema. Always \
name any major contraindications or cautions relevant to this specific patient (age, other \
conditions, other medications) in the "cautions" field. If the differential is too uncertain, or \
the condition has no well-established first-line approach, omit treatment_options entirely (an \
empty list) rather than speculating.
- You will be told explicitly which data categories are on file for this patient from prior \
visits (labs, imaging, surgical history, medication start/stop dates, etc.), clearly separated \
from what's reported for today's visit. Any category explicitly marked "none on file" or "none \
recorded" must be treated as UNKNOWN or NOT YET ASSESSED, never as normal, negative, or \
absent-of-findings. Never assume a body system, lab value, or imaging study is normal just \
because no data about it is on file. "No imaging on file" and "imaging shows no acute findings" \
are different facts - do not collapse them. Populate missing_data_flags with every clinically \
relevant category that is missing given the presenting complaint (e.g. a chest pain case with no \
troponin or ECG on file should flag both), and separately populate suggested_tests with what \
you'd want ordered to close those gaps.
- Output ONLY valid JSON. No markdown, no preamble, no commentary outside the JSON object.
"""

OUTPUT_SCHEMA_HINT = """
Respond with exactly this JSON shape:
{
  "differentials": [
    {"condition": "string", "likelihood_rank": 1, "rationale": "string"}
  ],
  "medication_flags": [
    {"medications_involved": ["string"], "concern": "string", "severity": "low|moderate|high", "evidence_level": "established|limited/post-marketing"}
  ],
  "symptom_medication_correlations": [
    {"symptom": "string", "medication": "string", "correlation": "definite|possible|no_issue", "note": "string"}
  ],
  "therapeutic_duplications": [
    {"medications_involved": ["string"], "shared_ingredient_or_class": "string", "ingredient_type": "active|inactive", "concern": "string", "severity": "low|moderate|high"}
  ],
  "dosage_concerns": [
    {"medication": "string", "concern": "string", "basis": "string", "severity": "low|moderate|high"}
  ],
  "polypharmacy_assessment": {
    "medication_count": 0,
    "cumulative_concerns": [
      {"concern": "string", "medications_involved": ["string"], "severity": "low|moderate|high"}
    ],
    "note": "string or null"
  },
  "treatment_options": [
    {"option": "string", "type": "pharmacological|non-pharmacological", "for_condition": "string (the differential this addresses)", "rationale": "string", "cautions": "string or null"}
  ],
  "missing_data_flags": [
    {"category": "labs|imaging|surgical_history|medication_dates|vaccination_history|substance_use|other", "why_it_matters": "string"}
  ],
  "suggested_tests": [
    {"test": "string", "rationale": "string"}
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

    # Expanded intake fields
    submitted_by: Optional[str] = None
    signs: list[str] = field(default_factory=list)
    onset: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    aggravating_factors: Optional[str] = None
    alleviating_factors: Optional[str] = None
    review_of_systems: list[str] = field(default_factory=list)
    physical_exam_findings: Optional[str] = None
    labs_imaging: Optional[str] = None
    allergies: Optional[str] = None
    family_history: Optional[str] = None
    social_history: Optional[str] = None

    # Attached diagnostic test files (EKG, echo, labs, imaging reports).
    # Each item: {"filename": str, "media_type": str, "data": base64-encoded str}
    attached_files: list[dict] = field(default_factory=list)

    def to_prompt(self) -> str:
        parts = [f"Chief complaint: {self.chief_complaint}"]
        if self.submitted_by:
            parts.append(f"Submitted by: {self.submitted_by}")
        if self.age is not None or self.sex:
            parts.append(f"Patient: {self.age or 'unknown age'}, {self.sex or 'sex unspecified'}")
        if self.symptoms:
            parts.append(f"Symptoms (patient-reported): {', '.join(self.symptoms)}")
        if self.signs:
            parts.append(f"Signs (clinician-observed): {', '.join(self.signs)}")
        if self.onset or self.duration or self.severity:
            char_parts = []
            if self.onset:
                char_parts.append(f"onset {self.onset}")
            if self.duration:
                char_parts.append(f"duration {self.duration}")
            if self.severity:
                char_parts.append(f"severity {self.severity}")
            parts.append(f"Symptom characteristics: {', '.join(char_parts)}")
        if self.aggravating_factors:
            parts.append(f"Worse with: {self.aggravating_factors}")
        if self.alleviating_factors:
            parts.append(f"Better with: {self.alleviating_factors}")
        if self.review_of_systems:
            parts.append(f"Review of systems (positive findings): {', '.join(self.review_of_systems)}")
        if self.current_medications:
            parts.append(f"Current medications: {', '.join(self.current_medications)}")
        if self.allergies:
            parts.append(f"Allergies: {self.allergies}")
        if self.relevant_history:
            parts.append(f"Relevant medical history: {', '.join(self.relevant_history)}")
        if self.family_history:
            parts.append(f"Family history: {self.family_history}")
        if self.social_history:
            parts.append(f"Social history: {self.social_history}")
        if self.physical_exam_findings:
            parts.append(f"Physical exam findings: {self.physical_exam_findings}")
        if self.labs_imaging:
            parts.append(f"Labs/imaging: {self.labs_imaging}")
        if self.vitals:
            parts.append(f"Vitals: {json.dumps(self.vitals)}")
        return "\n".join(parts)


def build_drug_facts_section(medications: list[str]) -> tuple[str, list[str], list[str]]:
    """Look up verified FDA label data for each current medication and format it
    for inclusion in the prompt. Returns (section_text, found_meds, not_found_meds)
    so callers can also surface which medications had verified data available.
    """
    if not medications:
        return "", [], []

    found_blocks = []
    found_meds = []
    not_found = []
    for med in medications:
        facts = fetch_drug_facts(med)
        if not facts:
            not_found.append(med)
            continue
        found_meds.append(med)
        lines = [f"- {med} (matched as \"{facts['queried_as']}\"):"]
        if facts.get("active_ingredient"):
            lines.append(f"    Active ingredient: {facts['active_ingredient']}")
        if facts.get("inactive_ingredient"):
            lines.append(f"    Inactive ingredients: {facts['inactive_ingredient']}")
        if facts.get("warnings"):
            lines.append(f"    Warnings: {facts['warnings']}")
        if facts.get("drug_interactions"):
            lines.append(f"    Documented drug interactions: {facts['drug_interactions']}")
        if facts.get("adverse_reactions"):
            lines.append(f"    Adverse reactions: {facts['adverse_reactions']}")
        found_blocks.append("\n".join(lines))

    section_parts = []
    if found_blocks:
        section_parts.append("Verified FDA label data (openFDA, real-time lookup):\n" + "\n".join(found_blocks))
    if not_found:
        section_parts.append(
            "No verified FDA label data found for: " + ", ".join(not_found) +
            ". Rely on general clinical knowledge for these and note this in insufficient_data_note."
        )
    return "\n\n".join(section_parts), found_meds, not_found


def build_clinical_picture_section(clinical_picture: Optional[dict]) -> str:
    """Formats a patient's on-file history (from prior visits, saved via
    the /patient/{id}/intake, /lab, /imaging, /medication endpoints) for
    inclusion in the prompt, clearly separated from what's typed fresh
    into this visit's case.

    Critical: every category is stated explicitly as present or "none on
    file" - never omitted when empty. An omitted section reads to the
    model as "wasn't asked about"; an explicit "none on file" reads as
    "checked and confirmed absent." That distinction is the whole point
    of this function.
    """
    if not clinical_picture:
        return (
            "ON-FILE PATIENT HISTORY: none available. This appears to be "
            "this patient's first recorded visit, or no prior data has "
            "been entered. Treat all history categories below as unknown, "
            "not as negative."
        )

    lines = ["ON-FILE PATIENT HISTORY (from prior visits, distinct from what's reported for today's visit above):"]

    intake = clinical_picture.get("intake")
    if intake:
        lines.append(
            f"- Surgical history on file: {intake['procedures_surgeries']}"
            if intake.get("procedures_surgeries") else
            "- Surgical history on file: none recorded"
        )
        lines.append(
            f"- Illicit/recreational drug use on file: {intake['illicit_drug_use']}"
            if intake.get("illicit_drug_use") else
            "- Illicit/recreational drug use on file: none recorded"
        )
        lines.append(
            f"- Supplements/OTC products on file: {intake['supplements']}"
            if intake.get("supplements") else
            "- Supplements/OTC products on file: none recorded"
        )
        lines.append(
            f"- Vaccination history on file: {intake['vaccination_history']}"
            if intake.get("vaccination_history") else
            "- Vaccination history on file: none recorded"
        )
        lines.append(
            f"- Clinical trial participation on file: {intake['clinical_trial_participation']}"
            if intake.get("clinical_trial_participation") else
            "- Clinical trial participation on file: none recorded"
        )
    else:
        lines.append("- No prior intake on file for any history category above.")

    labs = clinical_picture.get("labs") or []
    if labs:
        lines.append(f"- Lab results on file ({len(labs)}):")
        for lab in labs:
            payload = lab.get("payload", {})
            date_str = lab.get("date_performed", "unknown date")
            lines.append(
                f"    {date_str}: {payload.get('test_name', 'unnamed test')} = "
                f"{payload.get('result', 'no result recorded')} "
                f"[{payload.get('flag', 'flag not recorded')}]"
            )
    else:
        lines.append("- Lab results on file: NONE. Do not assume any lab value is normal -- "
                      "flag relevant missing labs in missing_data_flags.")

    imaging = clinical_picture.get("imaging") or []
    if imaging:
        lines.append(f"- Imaging/radiology reports on file ({len(imaging)}):")
        for img in imaging:
            payload = img.get("payload", {})
            date_str = img.get("date_performed", "unknown date")
            lines.append(
                f"    {date_str}: {payload.get('modality', 'unnamed study')} "
                f"({payload.get('region', 'region unspecified')}): "
                f"{payload.get('findings', 'no findings recorded')}"
            )
    else:
        lines.append("- Imaging/radiology reports on file: NONE. Do not assume imaging is normal "
                      "or that any structural finding has been ruled out -- flag relevant missing "
                      "imaging in missing_data_flags.")

    medications = clinical_picture.get("medications") or []
    if medications:
        lines.append(f"- Structured medication history on file ({len(medications)}, distinct from "
                      f"'Current medications' above which reflects today's visit):")
        for med in medications:
            start = med.get("start_date", "start date unknown")
            stop = med.get("stop_date") or "still active"
            lines.append(
                f"    {med.get('medication_name', 'unnamed')} "
                f"{med.get('dose', '')} {med.get('frequency', '')} "
                f"(started {start}, {stop})"
            )
    else:
        lines.append("- Structured medication history on file: none recorded beyond today's "
                      "reported current medications.")

    return "\n".join(lines)


def run_dx_engine(case: PatientCase, high_acuity: bool = False, clinical_picture: Optional[dict] = None) -> dict:
    """Send a patient case to the model and return structured differential + med-flag JSON.

    Set high_acuity=True to route complex/critical cases to the stronger model.
    Attached files (EKG, echo, labs, imaging) are sent as documents/images alongside
    the case text so the model can read them directly. Current medications are looked
    up against the live FDA label database before the case is sent to the model.
    clinical_picture (optional) carries this patient's on-file history from prior
    visits - labs, imaging, structured medications, and intake fields - gathered by
    the caller (backend.py's get_patient_clinical_picture()). Pass None for a
    patient's first visit or when no history exists yet.
    """
    model = HIGH_ACUITY_MODEL if high_acuity else MODEL

    drug_facts_section, verified_meds, unverified_meds = build_drug_facts_section(case.current_medications)
    prompt_parts = [case.to_prompt()]
    if drug_facts_section:
        prompt_parts.append(drug_facts_section)
    prompt_parts.append(build_clinical_picture_section(clinical_picture))
    prompt_parts.append(OUTPUT_SCHEMA_HINT)
    text_block = "\n\n".join(prompt_parts)

    if case.attached_files:
        content = []
        for f in case.attached_files:
            media_type = f.get("media_type", "")
            block_type = "document" if media_type == "application/pdf" else "image"
            content.append({
                "type": block_type,
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": f["data"],
                },
            })
        content.append({"type": "text", "text": text_block})
        user_content = content
    else:
        user_content = text_block

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Defensive cleanup in case the model wraps output in code fences or adds
    # stray text before/after the JSON despite instructions not to.
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    result = None
    stop_reason = getattr(response, "stop_reason", None)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(cleaned, return_objects=True)
            result = repaired if isinstance(repaired, dict) and repaired else None
        except Exception:
            result = None
    """
    except json.JSONDecodeError:
        # Fallback: extract the substring between the first '{' and the last '}'
        # in case the model added commentary around the JSON, or the output was
        # truncated mid-object (check stop_reason == "max_tokens" below).
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                result = json.loads(candidate)
            except json.JSONDecodeError:
                result = None
    """

    if result is None:
        print(
            f"DX ENGINE PARSE ERROR (stop_reason={stop_reason}). Raw model output:",
            raw_text,
            flush=True,
        )
        result = {
            "error": "Failed to parse model output as JSON",
            "raw_output": raw_text,
            "truncated": stop_reason == "max_tokens",
        }

    # Defensive default so the frontend can always safely read this key,
    # even on older cached responses or partial model output.
    result.setdefault("symptom_medication_correlations", [])
    result.setdefault("therapeutic_duplications", [])
    result.setdefault("dosage_concerns", [])
    result.setdefault("polypharmacy_assessment", {"medication_count": len(case.current_medications), "cumulative_concerns": [], "note": None})
    result.setdefault("treatment_options", [])
    result.setdefault("missing_data_flags", [])
    result.setdefault("suggested_tests", [])

    result["_meta"] = {
        "model": model,
        "submitted_by": case.submitted_by,
        "verified_drug_data_for": verified_meds,
        "no_verified_drug_data_for": unverified_meds,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "stop_reason": stop_reason,
    }
    return result


# ---------------------------------------------------------------------------
# SOAP note generation
# ---------------------------------------------------------------------------

SOAP_SYSTEM_PROMPT = """You are drafting a first-draft SOAP note for a licensed physician's \
review, based on structured intake data, on-file clinical data, and an already-generated \
differential/medication analysis. This is a DRAFT ONLY - every section must read as \
physician-editable shorthand, not a finalized clinical record, and the physician must review \
and sign off before it is used for care or billing.

Rules:
- Subjective: summarize patient-reported history in clinical shorthand (chief complaint, HPI, \
relevant history) - do not add anything the patient/intake didn't report.
- Objective: include only data actually on file (vitals, labs, imaging with dates). If a \
category is missing, write "Not on file" rather than omitting it silently or inferring a \
normal value.
- Assessment: list the differential considerations already generated, framed as "consider" \
language, never as a settled diagnosis. Include medication flags here if clinically relevant.
- Plan: draft reasonable next steps (suggested tests already identified, referrals, monitoring) \
as a starting point for the physician to edit, not a final order set.
- Never include a definitive diagnosis, a specific drug dosage recommendation, or any statement \
implying the AI is the author of record. This is the physician's note; you are drafting \
language for them to review, edit, and sign.
- Output ONLY valid JSON. No markdown, no preamble.
"""

SOAP_OUTPUT_SCHEMA_HINT = """
Respond with exactly this JSON shape:
{
  "subjective": "string",
  "objective": "string",
  "assessment": "string",
  "plan": "string"
}
"""


def generate_soap_note(intake: Optional[dict], labs: list[dict], imaging: list[dict],
                        medications: list[dict], ai_analysis: dict) -> dict:
    """Generates a first-draft SOAP note from a patient's on-file data and an
    already-generated AI analysis. Uses the same client/model pattern as
    run_dx_engine(). Called from backend.py's /analysis/{id}/soap-note route."""
    user_content = (
        f"INTAKE DATA:\n{json.dumps(intake, default=str)}\n\n"
        f"LABS ON FILE:\n{json.dumps(labs, default=str)}\n\n"
        f"IMAGING ON FILE:\n{json.dumps(imaging, default=str)}\n\n"
        f"MEDICATIONS ON FILE:\n{json.dumps(medications, default=str)}\n\n"
        f"AI DIFFERENTIAL/MEDICATION ANALYSIS ALREADY GENERATED:\n{json.dumps(ai_analysis, default=str)}\n\n"
        f"{SOAP_OUTPUT_SCHEMA_HINT}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SOAP_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    stop_reason = getattr(response, "stop_reason", None)
    block_types = [getattr(b, "type", "unknown") for b in response.content]
    print(
        f"SOAP NOTE DEBUG: stop_reason={stop_reason}, "
        f"num_blocks={len(response.content)}, block_types={block_types}, "
        f"usage={getattr(response, 'usage', None)}",
        flush=True,
    )

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    if not cleaned:
        print(
            f"SOAP NOTE EMPTY RESPONSE. stop_reason={stop_reason}, "
            f"full response.content={response.content}",
            flush=True,
        )
        return {
            "subjective": "",
            "objective": "",
            "assessment": "",
            "plan": "",
            "error": f"Model returned empty content (stop_reason={stop_reason})",
        }

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        print("SOAP NOTE PARSE ERROR. Raw model output:", raw_text, flush=True)
        return {
            "subjective": "",
            "objective": "",
            "assessment": "",
            "plan": "",
            "error": "Failed to parse SOAP note output as JSON",
        }


if __name__ == "__main__":
    # Example pilot test case - patient on a medication with headache as a known side effect,
    # while also reporting headache as a symptom. This should now surface in
    # "symptom_medication_correlations" instead of being silently absorbed into the
    # general differential list.
    test_case = PatientCase(
        chief_complaint="Headache and fatigue over 2 weeks",
        symptoms=["headache", "fatigue", "mild dizziness"],
        age=58,
        sex="male",
        current_medications=["amlodipine 10mg daily", "atorvastatin 20mg daily"],
        relevant_history=["hypertension"],
        vitals={"temp_f": 98.4, "bp": "128/80", "hr": 74},
    )

    output = run_dx_engine(test_case)
    print(json.dumps(output, indent=2))
