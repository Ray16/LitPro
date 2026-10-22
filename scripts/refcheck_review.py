#!/usr/bin/env python3
"""
refcheck_review.py — turn drafted questions into a verified question set.

  clean   drop malformed drafts (bad answer_type, numeric without a value, empty question) before the refcheck
  split   read a refcheck run: questions most checker models answered correctly are marked verified;
          the rest go to to_review.jsonl, with each checker's answer next to the reference
  merge   fold your edits of to_review.jsonl back into questions.jsonl

  python refcheck_review.py clean --drafts questions.draft.jsonl --out questions.clean.jsonl
  python refcheck_review.py split --run runs/rc_claude runs/rc_gemini runs/rc_geminipro --questions questions.clean.jsonl
  python refcheck_review.py merge --questions questions.jsonl --reviewed to_review.jsonl

One domain at a time: pass --areas to the refcheck runs and to split, and give each domain its own files
(--out questions.<domain>.jsonl --review to_review.<domain>.jsonl). Concatenate the per-domain files at the end.

Editing to_review.jsonl: for each line either fix "reference" (and "question" if needed) and set
"verified": true, keep it as is and set "verified": true (reference right, models wrong: a hard question),
or set "drop": true.
"""
from __future__ import annotations

import argparse, collections, json, sys
from pathlib import Path

QTYPES = {"text", "table", "figure", "calculation", "conflict", "absent"}


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def cmd_clean(a):
    rows, kept, why = read_jsonl(a.drafts), [], collections.Counter()
    for r in rows:
        ref = r.get("reference") or {}
        at = r.get("answer_type")
        if at == "not_reported" or r.get("qtype") == "absent":
            r["answer_type"], r["qtype"] = "not_reported", "absent"
        if not (r.get("question") or "").strip():
            why["empty question"] += 1; continue
        if r["answer_type"] not in ("numeric", "text", "not_reported"):
            why[f"answer_type {at!r}"] += 1; continue
        if r["answer_type"] == "numeric":
            try:
                ref["value"] = float(ref.get("value"))
            except (TypeError, ValueError):
                why["numeric without a value"] += 1; continue
        if r["answer_type"] == "text" and not (ref.get("text") or ref.get("accept")):
            why["text without a reference"] += 1; continue
        if r.get("qtype") not in QTYPES:
            r["qtype"] = "text"
        r["reference"] = ref
        kept.append(r)
    write_jsonl(a.out, kept)
    print(f"kept {len(kept)} of {len(rows)} drafts -> {a.out}")
    for k, v in why.most_common():
        print(f"  dropped {v}: {k}")


def cmd_split(a):
    qs = read_jsonl(a.questions)
    if a.areas:
        keep = set(a.areas.split(","))
        qs = [q for q in qs if q["paper_id"].rsplit("_", 1)[0] in keep]
        if not qs:
            sys.exit(f"no questions for areas {a.areas}")
    import csv
    by_q, flags = collections.defaultdict(list), set()
    for run in map(Path, a.run):            # one or several refcheck runs (e.g. one per checker model)
        for i in read_jsonl(run / "items.jsonl"):
            by_q[i["qid"]].append(i)
        if (run / "judge_flags.csv").exists():
            flags |= {r["qid"] for r in csv.DictReader((run / "judge_flags.csv").open())}
    out, review, stats = [], [], collections.Counter()
    per_area = collections.defaultdict(lambda: [0, 0])
    for q in qs:
        its = [i for i in by_q.get(q["qid"], []) if i["outcome"] != "error"]
        n_ok = sum(i["outcome"] == "correct" for i in its)
        reason = None
        if len(its) < 2:
            reason = f"only {len(its)} valid checker answer(s)"
        elif n_ok * 2 <= len(its):          # with 2 checkers: both must agree; with 3: at least 2
            reason = f"{n_ok}/{len(its)} checkers agree with the reference"
        elif q["qid"] in flags:
            reason = "judges disagreed on a checker's answer"
        area = q["paper_id"].rsplit("_", 1)[0]
        per_area[area][1] += 1
        if reason:
            per_area[area][0] += 1
            stats[reason.split(" checker")[0] if "checkers agree" in reason else reason] += 1
            q = dict(q, verified=False)
            review.append(dict(q, review_reason=reason, checker_answers=[
                dict(model=i["extractor"], answer=i.get("answer"), unit=i.get("unit"), status=i.get("status"),
                     outcome=i["outcome"], evidence=i.get("evidence"), location=i.get("location"))
                for i in by_q.get(q["qid"], [])]))
        else:
            q = dict(q, verified=True, verified_by=f"refcheck: {n_ok}/{len(its)} checkers agree")
        out.append(q)
    write_jsonl(a.out, out)
    write_jsonl(a.review, review)
    print(f"{len(out) - len(review)} verified, {len(review)} to review -> {a.review}\n")
    for area, (n_rev, n) in sorted(per_area.items()):
        print(f"  {area:18s} {n_rev:4d}/{n:<4d} to review ({n_rev / n:.0%})")
    print()
    for k, v in stats.most_common():
        print(f"  {v:4d}  {k}")


def cmd_merge(a):
    qs = {q["qid"]: q for q in read_jsonl(a.questions)}
    n_drop = n_ok = n_left = 0
    for r in read_jsonl(a.reviewed):
        if r["qid"] not in qs:
            continue
        if r.get("drop"):
            del qs[r["qid"]]; n_drop += 1; continue
        r = {k: v for k, v in r.items() if k not in ("review_reason", "checker_answers")}
        if r.get("verified"):
            r.setdefault("verified_by", "manual review"); n_ok += 1
        else:
            n_left += 1
        qs[r["qid"]] = r
    write_jsonl(a.questions, list(qs.values()))
    print(f"merged: {n_ok} verified, {n_drop} dropped, {n_left} still unverified; {len(qs)} questions in {a.questions}")
    if n_left:
        print("the run refuses unverified questions: finish them, drop them, or pass --allow-unverified")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clean"); c.add_argument("--drafts", required=True); c.add_argument("--out", default="questions.clean.jsonl")
    s = sub.add_parser("split"); s.add_argument("--run", required=True, nargs="+"); s.add_argument("--questions", required=True)
    s.add_argument("--out", default="questions.jsonl"); s.add_argument("--review", default="to_review.jsonl")
    s.add_argument("--areas", default=None, help="only these domains, e.g. electrocatalysis (comma list)")
    m = sub.add_parser("merge"); m.add_argument("--questions", default="questions.jsonl")
    m.add_argument("--reviewed", default="to_review.jsonl")
    a = ap.parse_args()
    {"clean": cmd_clean, "split": cmd_split, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()