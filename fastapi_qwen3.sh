#!/bin/bash

echo "📦 Installing dependencies..."
pip install fastapi uvicorn requests --break-system-packages 2>/dev/null || pip install fastapi uvicorn requests

echo "📝 Creating main.py..."
cat > main.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "running", "model": MODEL}

@app.post("/chat")
def chat(req: PromptRequest):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": req.prompt,
        "stream": False
    })
    result = response.json()
    return {"response": result.get("response", "")}
EOF

echo "🚀 Starting Ollama..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

echo "⏳ Waiting for Ollama..."
until curl -s http://localhost:11434 > /dev/null 2>&1; do
    sleep 1
done
echo "✅ Ollama ready!"

echo "📥 Pulling qwen3:8b..."
ollama pull qwen3:8b

echo "🌐 Starting FastAPI on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000