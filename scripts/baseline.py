import os, json, glob, re
from jsonschema import validate
from datetime import datetime
from extract_text import extract_text
from run_ollama import run_ollama, strip_to_json

SCHEMA_MAP = json.load(open("configs/schema.json"))

# Build a minimal JSON Schema validator from your map
SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_id": {"type": "string"},
        "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "total": {"type": "number"}
    },
    "required": ["invoice_id", "date", "total"]
}

def normalize(parsed: dict) -> dict:
    out = dict(parsed)
    # date → ISO
    if "date" in out:
        m = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", str(out["date"]))
        if m:
            out["date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # total → float
    if "total" in out:
        try:
            out["total"] = float(str(out["total"]).replace(",", "").replace("$", ""))
        except Exception:
            pass
    return out

def make_prompt(doc_text: str) -> str:
    return f"""
You are an information extraction system.
Return ONLY valid JSON matching this schema (no comments, no code fences):

{json.dumps(SCHEMA)}

Document:
{doc_text[:12000]}
"""

def extract_one(pdf_path: str, model="mistral:7b") -> dict:
    text = extract_text(pdf_path)
    raw = run_ollama(make_prompt(text), model=model)
    cand = strip_to_json(raw)

    def validate_or_raise(js: str):
        data = json.loads(js)
        data = normalize(data)
        validate(instance=data, schema=SCHEMA)
        return data

    try:
        return validate_or_raise(cand)
    except Exception as e:
        # one retry with validator error
        repair_prompt = f"""{make_prompt(text)}

The previous JSON was invalid because: {str(e)}.
Return corrected JSON only (no prose, no fences).
"""
        repaired = strip_to_json(run_ollama(repair_prompt, model=model))
        return validate_or_raise(repaired)

if __name__ == "__main__":
    os.makedirs("reports/preds_baseline", exist_ok=True)
    pdfs = sorted(glob.glob("data/raw_pdfs/*.pdf"))
    results = []
    for pdf in pdfs:
        doc_id = os.path.splitext(os.path.basename(pdf))[0]
        try:
            data = extract_one(pdf)
            with open(f"reports/preds_baseline/{doc_id}.json","w") as f:
                json.dump({"doc_id": doc_id, "pred": data}, f, indent=2)
            ok = True; err = ""
        except Exception as e:
            ok = False; err = str(e)
        results.append({"doc_id": doc_id, "ok": ok, "error": err})

    run_report = {
        "run": "baseline",
        "model": "mistral:7b",
        "schema_fields": list(SCHEMA_MAP.keys()),
        "results": results,
    }
    with open("reports/baseline_run.json","w") as f:
        json.dump(run_report, f, indent=2)
    print("Saved outputs to reports/preds_baseline/ and reports/baseline_run.json")
