import os
from google import genai

def build_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("VERTEX_LOCATION")
    if use_vertex in {"1", "true", "yes"} and project and location:
        return genai.Client(vertexai=True, project=project, location=location)

    raise RuntimeError(
        "Gemini credentials not configured. Set GEMINI_API_KEY/GOOGLE_API_KEY, "
        "or configure Vertex AI env vars."
    )

client = build_client()

# ✅ Use a model that exists in your ListModels output.
# Common working choices are often: "gemini-2.0-flash" or "gemini-2.5-flash-lite"
MODEL = "gemini-2.5-flash"

resp = client.models.generate_content(
    model=MODEL,
    contents="Say exactly: OK"
)

print(resp.text)
