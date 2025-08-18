import ollama, requests

def add_two_numbers(a: int, b:int) -> int:
    return a + b

def fetch_url(method: str, url:str) -> str:
    r = requests.request(method=method, url=url, timeout=10)
    return r.text[:1000]

availabe = {
    "add_two_numbers": add_two_numbers,
    "fetch_url": fetch_url
}

messages = [
    {"role": "system", "content": "You are an agent that may call tools when needed."},
    {"role": "user", "content": "What is 123 + 456? then fetch the ollama.com homepage title"}
]

resp = ollama.chat(
    model="llama3.1",
    messages=messages,
    tools=[add_two_numbers, fetch_url],
)

messages.append(resp.message)
for call in resp.message.tool_calls or []:
    fn = availabe.get(call.function.name)
    if not fn:
        continue
    out = fn(**call.function.arguments)
    messages.append({
        "role": "tool",
        "content": str(out),
        "name": call.function.name,
    })
    
    final = ollama.chat(model="llama3.1", messages=messages)
    print(final.message.content)