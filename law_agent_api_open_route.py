"""
Law Information Extraction API — FastAPI + LangChain Tool Calling + OpenRouter
------------------------------------------------------------------------------
Run:
  1. Set OPENROUTER_API_KEY in your environment or a .env file
  2. pip install fastapi uvicorn langchain langchain-openai langgraph pydantic python-dotenv
  3. uvicorn law_agent_api_open_route:app --reload
  4. Open http://localhost:8000/docs
"""

import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI          # ← OpenRouter-compatible
from langgraph.prebuilt import create_react_agent

load_dotenv()  # loads OPENROUTER_API_KEY from .env if present

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ClaimantRequest(BaseModel):
    """Input schema: the claimant's summary profile as free text."""
    profile: str = Field(
        ...,
        min_length=10,
        examples=[
            "Name: Ahmad, Age: 34, Jurisdiction: Malaysia, "
            "Incident: Slipped at mall, fractured wrist. Date: 15 March 2024."
        ],
    )
    # Any OpenRouter model slug — see https://openrouter.ai/models
    model: str = Field(
        default="qwen/qwen3-8b",
        examples=[
            "qwen/qwen3-8b",
            "google/gemma-3-27b-it:free",
        ],
    )

class ToolCallLog(BaseModel):
    tool_name: str
    tool_input: dict | None = None
    tool_output: str

class LawExtractionResponse(BaseModel):
    claimant_profile: str
    legal_analysis: str
    model: str
    tools_used: list[ToolCallLog] = []


# ── Law Knowledge Base ────────────────────────────────────────────────────────

LAW_DATABASE = {
    "personal_injury": {
        "malaysia": {
            "applicable_laws": [
                "Civil Law Act 1956 (Act 67) — Section 7: Wrongful act causing death",
                "Civil Law Act 1956 — Section 28A: Damages for personal injuries",
                "Limitation Act 1953 (Act 254) — Section 6(1): 6-year limitation",
                "Employees Social Security Act 1969 (SOCSO)",
            ],
            "limitation_period": "6 years from date of incident",
            "remedies": ["General damages", "Special damages", "Future loss of earnings", "Loss of amenities"],
            "key_precedents": ["Ong Ah Long v Dr S Underwood [1983]", "Chan Chin Ming v Lim Yok Eng [1994]"],
        },
        "singapore": {
            "applicable_laws": [
                "Work Injury Compensation Act (WICA)",
                "Limitation Act (Cap 163) — Section 24A: 3-year limitation",
            ],
            "limitation_period": "3 years from date of injury",
            "remedies": ["Pain and suffering damages", "Loss of future earnings", "Medical expenses"],
            "key_precedents": ["Lai Wai Keong Eugene v Loo Wei Yen [2014]"],
        },
    },
    "employment_dispute": {
        "malaysia": {
            "applicable_laws": [
                "Employment Act 1955 (Act 265)",
                "Industrial Relations Act 1967 — Section 20: Unfair dismissal",
                "Employment (Amendment) Act 2022",
            ],
            "limitation_period": "60 days to file unfair dismissal complaint",
            "remedies": ["Reinstatement", "Back wages (max 24 months)", "Compensation in lieu"],
            "key_precedents": ["Milan Auto Sdn Bhd v Wong Seh Yen [1995]"],
        },
    },
    "motor_vehicle_accident": {
        "malaysia": {
            "applicable_laws": [
                "Road Transport Act 1987 (Act 333)",
                "Civil Law Act 1956 — Section 28A",
                "Motor Vehicles (Third-Party Risks) Act",
            ],
            "limitation_period": "6 years for civil claims; 3 years for fatal accident claims",
            "remedies": ["Third-party insurance claims", "General & special damages", "Dependency claims"],
            "key_precedents": ["Takong Tabari v Government of Sarawak [1998]"],
        },
    },
    "medical_negligence": {
        "malaysia": {
            "applicable_laws": [
                "Civil Law Act 1956",
                "Private Healthcare Facilities and Services Act 1998",
                "Medical Act 1971",
            ],
            "limitation_period": "6 years from date of negligent act",
            "remedies": ["Compensatory damages", "Aggravated damages", "MMC complaint"],
            "key_precedents": ["Foo Fio Na v Dr Soo Fook Mun [2007]"],
        },
    },
}


# ── Custom Tool ───────────────────────────────────────────────────────────────

@tool
def extract_law_info(category: str, jurisdiction: str = "malaysia") -> str:
    """
    Extracts relevant law information for a given incident category and jurisdiction.

    Use this tool when you receive a claimant profile and need to find applicable laws,
    limitation periods, remedies, and legal precedents.

    Args:
        category: One of 'personal_injury', 'employment_dispute',
                  'motor_vehicle_accident', 'medical_negligence'.
        jurisdiction: 'malaysia' or 'singapore'.

    Returns:
        JSON with applicable_laws, limitation_period, remedies, key_precedents.
    """
    category = category.lower().strip().replace(" ", "_")
    jurisdiction = jurisdiction.lower().strip()

    if category not in LAW_DATABASE:
        return json.dumps({"error": f"Unknown category. Available: {list(LAW_DATABASE.keys())}"})
    if jurisdiction not in LAW_DATABASE[category]:
        return json.dumps({"error": f"Unknown jurisdiction. Available: {list(LAW_DATABASE[category].keys())}"})

    return json.dumps(LAW_DATABASE[category][jurisdiction], indent=2)


TOOLS = [extract_law_info]

SYSTEM_PROMPT = """You are a legal research assistant specializing in Malaysian and Singaporean law.

When you receive a CLAIMANT PROFILE, you must:
1. Identify the incident category (personal_injury, employment_dispute, motor_vehicle_accident, medical_negligence).
2. Identify the jurisdiction from the profile.
3. Use the extract_law_info tool to retrieve applicable laws.
4. Present a structured legal summary: applicable laws, limitation period, remedies, and precedents.

Always use the tool. Never guess at legal information."""


# ── Agent Factory ─────────────────────────────────────────────────────────────

def build_law_agent(model_name: str, temperature: float = 0.0):
    """
    Builds a LangGraph ReAct agent backed by OpenRouter.
    OpenRouter is OpenAI-API-compatible, so ChatOpenAI works directly.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Export it or add it to a .env file."
        )

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        # Optional: OpenRouter recommends these headers for analytics/ranking
        default_headers={
            "HTTP-Referer": "http://localhost:8000",   # your site URL
            "X-Title": "Law Information Extraction API",
        },
    )
    agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
    return agent


# ── FastAPI App ───────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen/qwen-2.5-7b-instruct:free"
default_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global default_agent
    print(f"⏳ Building default law agent ({DEFAULT_MODEL}) via OpenRouter ...")
    try:
        default_agent = build_law_agent(DEFAULT_MODEL, temperature=0.0)
        print("✅ Law Agent ready.")
    except RuntimeError as e:
        print(f"⚠️  Agent not initialised: {e}")
    yield
    print("🛑 Shutting down.")


app = FastAPI(
    title="Law Information Extraction API",
    description=(
        "Submit a claimant's summary profile and receive a structured legal analysis "
        "powered by a LangGraph ReAct agent via OpenRouter."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "backend": "openrouter",
        "default_model": DEFAULT_MODEL,
        "tools": [t.name for t in TOOLS],
    }


@app.post("/law/analyze", response_model=LawExtractionResponse)
def analyze_claimant(req: ClaimantRequest):
    """
    Accepts a claimant summary profile and returns a legal analysis
    with applicable laws, limitation periods, remedies, and precedents.
    """
    # Re-use the warmed-up default agent when possible
    if req.model == DEFAULT_MODEL and default_agent:
        agent = default_agent
    else:
        try:
            agent = build_law_agent(req.model, temperature=0.0)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=req.profile)]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # Extract tool call logs
    tool_names = {t.name for t in TOOLS}
    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "name") and msg.name in tool_names:
            tools_used.append(ToolCallLog(
                tool_name=msg.name,
                tool_input=getattr(msg, "tool_input", None),
                tool_output=str(msg.content),
            ))

    answer = result["messages"][-1].content

    return LawExtractionResponse(
        claimant_profile=req.profile,
        legal_analysis=answer,
        model=req.model,
        tools_used=tools_used,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("law_agent_api_open_route:app", host="0.0.0.0", port=8000, reload=True)