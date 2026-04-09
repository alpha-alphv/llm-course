# Building LLM Applications: Agents, Memory & RAG

A hands-on course notebook covering LLM application development with LangChain, LangGraph, and local/cloud models.

## Main Content

Work through **`LLM_Course_Notebook.ipynb`** — this is the primary course material.

### Day 1

| Session | Topic |
|---------|-------|
| Session 1 | Prompt Reliability Patterns |
| Session 2 | Local vs Cloud Models + Running with Ollama |
| Session 3 | Tool Calling + Agent Service (standalone script → FastAPI) |

### Day 2

| Session | Topic |
|---------|-------|
| Session 4 | RAG Grounding with LangChain + ChromaDB |
| Session 5 | Orchestration Patterns |
| Session 6 | Tracing + Evaluations with LangSmith |

### Setup

```bash
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Register the kernel for Jupyter
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

Then open the notebook and select the **Python (venv)** kernel.

---

## Bonus: Law Agent API

A practical FastAPI + LangGraph ReAct agent that performs legal analysis for Malaysian/Singaporean law. Submit a claimant profile and get back applicable laws, limitation periods, remedies, and precedents.

### Option A — Ollama (local, free)

**Prerequisites:** [Ollama](https://ollama.com) installed and running.

```bash
ollama serve
ollama pull qwen2.5:1.5b // Depends on which models did you prefer (Remember to check your device's specification for compatability)
uvicorn law_agent_api:app --reload
```

Open `http://localhost:8000/docs` and POST to `/law/analyze`.

### Option B — OpenRouter (cloud)

**Prerequisites:** An [OpenRouter](https://openrouter.ai) API key.

```bash
# Create a .env file with:
# OPENROUTER_API_KEY=sk-or-...

uvicorn law_agent_api_open_route:app --reload
```

Open `http://localhost:8000/docs` and POST to `/law/analyze`.

Supports any OpenRouter model slug (default: `qwen/qwen-2.5-7b-instruct:free`).

### Example request

```json
{
  "profile": "Name: Ahmad, Age: 34, Jurisdiction: Malaysia, Incident: Slipped at mall, fractured wrist. Date: 15 March 2024."
}
```

### Supported categories & jurisdictions

| Category | Jurisdictions |
|----------|--------------|
| `personal_injury` | Malaysia, Singapore |
| `employment_dispute` | Malaysia |
| `motor_vehicle_accident` | Malaysia |
| `medical_negligence` | Malaysia |
