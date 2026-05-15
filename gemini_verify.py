import os
import json
import time
from google import genai

MODEL = "gemini-2.5-flash"

PROMPT = """
You are verifying a bus lane violation.

RULE:
Crossing a DOUBLE SOLID lane marking is a violation.
Crossing dashed or single solid lanes is NOT a violation.

Analyze the provided dashcam video.

Return ONLY valid JSON:
{
  "decision": "Violation" | "NotViolation" | "Uncertain",
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}
"""

def _build_client() -> genai.Client:
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if api_key:
        return genai.Client(api_key=api_key)

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("VERTEX_LOCATION")
    if use_vertex in {"1", "true", "yes"} and project and location:
        return genai.Client(vertexai=True, project=project, location=location)

    raise RuntimeError(
        "Gemini credentials not configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY), "
        "or enable Vertex AI with GOOGLE_GENAI_USE_VERTEXAI=true plus "
        "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION."
    )

def verify_video(video_path: str):
    client = _build_client()

    uploaded_file = client.files.upload(file=video_path)

    # Wait for the file to become ACTIVE (videos can take time to process)
    file_name = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "id", None)
    if file_name:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                f = client.files.get(name=file_name)
            except Exception:
                f = None
            state = getattr(f, "state", None) if f is not None else None
            if state == "ACTIVE":
                break
            time.sleep(2)

    response = client.models.generate_content(
        model=MODEL,
        contents=[PROMPT, uploaded_file],
    )

    text = (response.text or "").strip()

    # extract JSON safely
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return {
            "decision": "Uncertain",
            "confidence": 0.0,
            "reason": "Invalid AI response"
        }

    try:
        parsed = json.loads(text[start:end+1])
        decision = (parsed.get("decision") or "").strip()
        normalized = decision.lower().replace(" ", "")
        if normalized in ("violation",):
            parsed["decision"] = "Violation"
        elif normalized in ("notviolation", "noviolation", "nonviolation"):
            parsed["decision"] = "NotViolation"
        elif normalized in ("uncertain",):
            parsed["decision"] = "Uncertain"
        else:
            parsed["decision"] = "Uncertain"
        return parsed
    except Exception:
        return {
            "decision": "Uncertain",
            "confidence": 0.0,
            "reason": "JSON parsing failed"
        }


