import ollama

messages = [
    {"role": "system", "content": "You are a concise senior dev assistant"},
    {"role": "user", "content": "In one sentence, explain what an embedding is."}
]

resp = ollama.chat(model="llama3.1", messages = messages)
print (resp.message.content)