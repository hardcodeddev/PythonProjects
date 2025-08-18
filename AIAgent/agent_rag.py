# agent_rag.py
import os, glob, textwrap, ollama, chromadb

def read_texts(folder="docs"):
    paths = glob.glob(os.path.join(folder, "**/*.md"), recursive=True)
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            yield p, f.read()

# 1) Build the vector store
client = chromadb.Client()
collection = client.get_or_create_collection("kb")

if collection.count() == 0:
    docs = list(read_texts("docs"))  # put your .md/.txt/.cs here
    for i, (path, text) in enumerate(docs):
        emb = ollama.embed(model="mxbai-embed-large", input=text[:8000])  # chunk if huge
        collection.add(ids=[path], embeddings=emb["embeddings"], documents=[text])

# 2) RAG tool the model can call
def search_docs(query: str) -> str:
    q = ollama.embed(model="mxbai-embed-large", input=query)
    res = collection.query(query_embeddings=[q["embeddings"]], n_results=3)
    return "\n\n---\n\n".join(res["documents"][0])

# 3) Chat with tool calling + RAG
messages = [
  {"role":"system","content":"Use tools when helpful. If you call search_docs, cite snippets."},
  {"role":"user","content":"In our repo, how do we configure Serilog and log to Seq?"}
]

resp = ollama.chat(model="llama3.1", messages=messages, tools=[search_docs])
messages.append(resp.message)

for call in resp.message.tool_calls or []:
    if call.function.name == "search_docs":
        out = search_docs(**call.function.arguments)
        messages.append({"role":"tool","name":"search_docs","content": out})

final = ollama.chat(model="llama3.1", messages=messages)
print(final.message.content)
