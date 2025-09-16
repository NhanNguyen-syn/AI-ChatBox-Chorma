"""
Offline QA evaluation for document-based Q&A (focus on Vietnamese).
- Loads test cases from JSON (see cases/*.json)
- Calls the running FastAPI backend /api/chat/send endpoint via HTTP
- Computes accuracy@1, exact-match, and fuzzy F1 over answers
- Logs per-case results to CSV for tracking regressions

Run:
  uvicorn main:app --reload  # in another terminal
  python backend/tests/qa_eval.py --token <JWT> --cases backend/tests/cases/sample_vi.json

The JWT can be taken from the frontend after login or via /api/auth/login.
"""
from __future__ import annotations
import argparse, json, csv, os, time, re
import requests
from difflib import SequenceMatcher


def normalize_vi_text(s: str) -> str:
    try:
        import unicodedata
        s = unicodedata.normalize('NFKD', s or '')
        s = ''.join([c for c in s if not unicodedata.combining(c)])
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s
    except Exception:
        return (s or '').lower().strip()


def f1_score(pred: str, gold: str) -> float:
    p = set(normalize_vi_text(pred).split())
    g = set(normalize_vi_text(gold).split())
    if not p or not g:
        return 0.0
    tp = len(p & g)
    precision = tp / len(p)
    recall = tp / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default=os.getenv('QA_API', 'http://localhost:8000/api/chat/send'))
    ap.add_argument('--token', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--session', default=None)
    ap.add_argument('--out', default='qa_eval_results.csv')
    args = ap.parse_args()

    with open(args.cases, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    headers = {'Authorization': f'Bearer {args.token}'}

    rows = []
    n = len(cases)
    hits = 0
    f1s = []
    for i, case in enumerate(cases, 1):
        q = case['q']
        gold = case['a']
        payload = {'message': q}
        if args.session:
            payload['session_id'] = args.session
        t0 = time.time()
        r = requests.post(args.api, json=payload, headers=headers, timeout=60)
        ms = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            print(f"[{i}/{n}] HTTP {r.status_code}: {r.text}")
            rows.append([q, gold, '', 0, 0.0, ms])
            continue
        resp = r.json()
        pred = resp.get('response', '')
        em = int(normalize_vi_text(pred) == normalize_vi_text(gold))
        sim = SequenceMatcher(None, normalize_vi_text(pred), normalize_vi_text(gold)).ratio()
        f1 = f1_score(pred, gold)
        hits += em
        f1s.append(f1)
        rows.append([q, gold, pred, em, round(f1, 4), ms, round(sim, 4)])
        print(f"[{i}/{n}] EM={em} F1={f1:.3f} time={ms}ms | Q: {q}\n-> {pred}\n")

    acc = hits / max(1, n)
    avg_f1 = sum(f1s) / max(1, len(f1s))
    print(f"Summary: cases={n} acc@1={acc:.3f} avgF1={avg_f1:.3f}")

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['question', 'gold', 'pred', 'exact', 'f1', 'latency_ms', 'sim'])
        for r in rows:
            w.writerow(r)
    print('Saved:', args.out)


if __name__ == '__main__':
    main()

