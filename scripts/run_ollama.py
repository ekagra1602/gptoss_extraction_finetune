import json, requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def run_ollama(prompt: str, model: str = "mistral:7b") -> str:
    r = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt}, stream=True, timeout=600)
    out = []
    for line in r.iter_lines():
        if not line: continue
        data = json.loads(line.decode("utf-8"))
        if "response" in data:
            out.append(data["response"])
    return "".join(out)

def strip_to_json(text: str) -> str:
    # grab first {...} block; helps when model adds fences
    s = text.find("{"); e = text.rfind("}")
    return text[s:e+1] if s != -1 and e != -1 and e > s else text
