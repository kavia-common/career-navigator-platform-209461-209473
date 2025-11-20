import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# PUBLIC_INTERFACE
class RecommendRequest(BaseModel):
    """Input payload for skill recommendation generation."""
    skillName: str = Field(..., description="The name of the skill for which to generate recommendations.")
    skillDescription: Optional[str] = Field(
        None, description="A short description of the skill to provide additional context."
    )
    requiredLevel: Optional[str] = Field(
        None, description="Target proficiency level (e.g., Beginner, Intermediate, Advanced)."
    )

# PUBLIC_INTERFACE
class RecommendResponse(BaseModel):
    """Recommendation response including steps, resources, and projects."""
    steps: List[str] = Field(..., description="Three concrete skill-improvement steps.")
    resources: List[str] = Field(..., description="Three curated learning resources.")
    projects: List[str] = Field(..., description="Two practical project ideas.")

def _deterministic_fallback(req: RecommendRequest) -> RecommendResponse:
    """
    Deterministic fallback generator that does not call external services.
    Produces predictable, structured recommendations based on inputs.
    """
    skill = req.skillName.strip()
    level = (req.requiredLevel or "Intermediate").strip().capitalize()
    desc = (req.skillDescription or "").strip()

    # Simple templated steps
    steps = [
        f"Master the core {skill} fundamentals{'' if not desc else f' ({desc})'} using official documentation and a 7-day study plan.",
        f"Build small, daily exercises applying {skill} at a {level} level; track progress and note gaps.",
        f"Create one end-to-end mini project focusing on key {skill} patterns, then solicit feedback and iterate."
    ]

    # Simple templated resources
    resources = [
        f"Official {skill} documentation and guides.",
        f"One structured {skill} course suited for {level} learners (e.g., Coursera/Udemy/edX).",
        f"A community-curated {skill} GitHub awesome list or roadmap for {level} path."
    ]

    # Simple templated project ideas
    projects = [
        f"{skill} Capstone: Implement a practical tool solving a real problem you have (scope: 1–2 weeks).",
        f"{skill} Portfolio Project: Write a tutorial or demo repo showing {level}-level concepts with clear README."
    ]

    return RecommendResponse(steps=steps[:3], resources=resources[:3], projects=projects[:2])

def _openai_enabled() -> bool:
    """
    Check if OpenAI is configured via environment variable.
    """
    return bool(os.getenv("OPENAI_API_KEY"))

def _maybe_generate_with_openai(req: RecommendRequest) -> Optional[RecommendResponse]:
    """
    Attempt to generate recommendations using OpenAI if available.
    This function must be resilient and return None if anything fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # Lazy import to avoid dependency if not used
    try:
        import httpx
    except Exception:
        # If httpx is not available (should be per requirements), fallback to deterministic
        return None

    # Use OpenAI API (Responses API style or chat completions). We'll use responses endpoint schema-like prompt.
    # We avoid adding openai package dependency; we'll call REST API directly with httpx.
    # Model name chosen to be broadly compatible; adjust if environment maps differently.
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = (
        "You are an assistant that returns STRICT JSON ONLY with keys: steps (array of 3 strings), "
        "resources (array of 3 strings), projects (array of 2 strings). "
        "Do not include any extra keys or text. "
        f"Skill: {req.skillName}\n"
        f"Description: {req.skillDescription or ''}\n"
        f"Required Level: {req.requiredLevel or 'Intermediate'}\n"
        "Return only JSON."
    )
    # We'll use Chat Completions for broader compatibility.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only. No prose."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # Parse JSON strictly
            import json as _json
            parsed = _json.loads(content)
            # Validate and coerce into Pydantic response
            return RecommendResponse(
                steps=[str(s) for s in (parsed.get("steps") or [])][:3],
                resources=[str(s) for s in (parsed.get("resources") or [])][:3],
                projects=[str(s) for s in (parsed.get("projects") or [])][:2],
            )
    except Exception:
        return None

# Initialize FastAPI app with metadata and tags for OpenAPI
app = FastAPI(
    title="LLM Recommendation Service",
    description=(
        "Microservice that generates learning steps, resources, and project ideas for a given skill.\n"
        "Provides deterministic fallback generation without external API calls, and attempts OpenAI usage when OPENAI_API_KEY is set."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "Health", "description": "Health and diagnostics"},
        {"name": "Recommendation", "description": "LLM recommendation generation"},
    ],
)

# Enable permissive CORS for MVP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PUBLIC_INTERFACE
@app.get("/", tags=["Health"], summary="Health Check")
def health_check():
    """
    Health/readiness endpoint to verify the service is operational.

    Returns:
        dict: Simple health message for monitoring and readiness checks.
    """
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.post(
    "/recommend",
    tags=["Recommendation"],
    summary="Generate recommendations for a skill",
    response_model=RecommendResponse,
)
def post_recommend(payload: RecommendRequest) -> RecommendResponse:
    """
    Generate recommendations using OpenAI when available, otherwise fallback deterministically.

    Parameters:
        payload (RecommendRequest): Input with skillName, optional skillDescription, requiredLevel.

    Returns:
        RecommendResponse: JSON-only with keys steps[3], resources[3], projects[2].

    Notes:
        - If OPENAI_API_KEY is configured, the service attempts to call OpenAI.
        - On any failure or absence of the key, a deterministic fallback is returned.
    """
    if not payload.skillName or not payload.skillName.strip():
        raise HTTPException(status_code=422, detail="skillName is required and must be non-empty")

    # Try OpenAI if enabled, fallback on failure
    if _openai_enabled():
        generated = _maybe_generate_with_openai(payload)
        if generated and len(generated.steps) == 3 and len(generated.resources) == 3 and len(generated.projects) == 2:
            return generated

    # Deterministic fallback
    return _deterministic_fallback(payload)


# Uvicorn entrypoint helper (bind to 0.0.0.0:8001)
if __name__ == "__main__":
    import uvicorn

    port_str = os.getenv("PORT", "8001")
    try:
        port = int(port_str)
    except ValueError:
        port = 8001

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
