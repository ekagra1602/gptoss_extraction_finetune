import json, glob, math
from rapidfuzz import fuzz

def load_gold(path="data/annotations/test.jsonl"):
    gold = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            gold[row["doc_id"]] = row["labels"]
    return gold

def load_preds(dir_="reports/preds_baseline"):
    preds = {}
    for p in glob.glob(f"{dir_}/*.json"):
        row = json.load(open(p))
        preds[row["doc_id"]] = row["pred"]
    return preds

def str_ok(a,b):                               # robust string compare
    return fuzz.token_sort_ratio(str(a).strip().lower(), str(b).strip().lower()) >= 95

def num_ok(a,b):                                # numeric tolerance
    try:
        return abs(float(a)-float(b)) <= 0.01
    except Exception:
        return False

FIELDS = ["invoice_id","date","total"]

def score(gold, preds):
    validity = 0; total_docs = 0
    tp=fp=fn=0                                  # counts across all docs & fields (micro)
    for did, g in gold.items():
        if did not in preds:
            fn += len(FIELDS); total_docs += 1; continue
        p = preds[did]; total_docs += 1

        # validity: all required fields present
        valid = all(k in p for k in FIELDS)
        validity += 1 if valid else 0

        for k in FIELDS:
            if k not in p: fn += 1; continue
            ok = False
            if k == "total":
                ok = num_ok(p[k], g[k])
            else:
                ok = str_ok(p[k], g[k])    # dates are strings; we normalized earlier
            if ok: tp += 1
            else: fp += 1

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec / (prec+rec+1e-9)
    return {"F1_micro": round(f1,4), "precision": round(prec,4), "recall": round(rec,4),
            "valid_json_rate": round(validity/total_docs,4)}

if __name__ == "__main__":
    gold = load_gold()
    preds = load_preds()
    print(score(gold, preds))
