# serve_agent.py
from fastapi import FastAPI
import ollama

app = FastAPI()
SYSTEM = "You are a precise, terse dev agent. Use tools when needed."

@app.post("/chat")
async def chat(q: dict):
    messages = [{"role":"system","content":SYSTEM},
                {"role":"user","content": q["prompt"]}]
    resp = ollama.chat(model="llama3.1", messages=messages)  # add tools/RAG here
    return {"answer": resp.message.content}
