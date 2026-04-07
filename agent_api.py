"""
FastAPI Agent API with LangChain Tool Calling + Ollama
------------------------------------------------------
Run:
  1. ollama serve
  2. ollama pull qwen2.5:1.5b
  3. pip install fastapi uvicorn langchain langchain-ollama langgraph python-dotenv
  4. uvicorn agent_api:app --reload
  5. Open http://localhost:8000/docs
"""

import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["What is the current date?"])
    model: str = Field(default="qwen2.5:1.5b", examples=["qwen2.5:1.5b", "llama3.2:1b"])
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)

class ToolCall(BaseModel):
    tool_name: str
    tool_output: str

class AgentResponse(BaseModel):
    query: str
    answer: str
    model: str
    tools_used: list[ToolCall] = []

# ── Custom Tools ──────────────────────────────────────────────────────────────
@tool
def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current date and time.
    Use this tool whenever the user asks for the current date, time, or both.
    """
    try:
        return datetime.datetime.now().strftime(format)
    except Exception as e:
        return f"Error formatting date/time: {e}"

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Use this tool when the user asks to calculate something.
    Only supports basic arithmetic: +, -, *, /, ().
    """
    allowed = set("0123456789+-*/.() ")
    if not all(ch in allowed for ch in expression):
        return "Error: expression contains invalid characters."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"

TOOLS = [get_current_datetime, calculator]

# ── Agent Factory ─────────────────────────────────────────────────────────────
def build_agent(model_name: str, temperature: float):
    """Build a LangGraph ReAct agent for a given model and temperature."""
    llm = ChatOllama(model=model_name, temperature=temperature)
    agent = create_react_agent(llm, TOOLS)
    return agent

# ── FastAPI App ───────────────────────────────────────────────────────────────
default_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global default_agent
    print("⏳ Building default agent (qwen2.5:1.5b) ...")
    default_agent = build_agent("qwen2.5:1.5b", temperature=0.0)
    print("✅ Agent ready.")
    yield
    print("🛑 Shutting down.")

app = FastAPI(
    title="LangChain Agent API",
    description="FastAPI wrapper around a LangGraph ReAct agent powered by Ollama.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/agent/chat", response_model=AgentResponse)
def agent_chat(req: AgentRequest):
    """Send a query to the agent and get a response with tool call logs."""
    
    if req.model == "qwen2.5:1.5b" and req.temperature == 0.0 and default_agent:
        agent = default_agent
    else:
        agent = build_agent(req.model, req.temperature)

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=req.query)]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # Extract tool calls from message history
    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "name") and msg.name in [t.name for t in TOOLS]:
            tools_used.append(ToolCall(
                tool_name=msg.name,
                tool_output=str(msg.content)
            ))

    answer = result["messages"][-1].content

    return AgentResponse(
        query=req.query,
        answer=answer,
        model=req.model,
        tools_used=tools_used,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_api:app", host="0.0.0.0", port=8000, reload=True)


# print("✅ agent_api.py saved! Run it with:")
# print("   uvicorn agent_api:app --reload")