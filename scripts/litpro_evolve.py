#!/usr/bin/env python3
"""
litpro_evolve.py — budgeted self-evolving prompt optimization for paper QA extraction.

For each (area, extractor model):
  iter 0   : evaluate the seed prompt on the TRAIN papers of that area
  iter 1..N: optimizer reads the failures -> proposes a new prompt (instructions + partition)
             -> evaluate on TRAIN -> keep it if it beats the best so far
  stop     : after max_iters, or when the next iteration would exceed max_tokens
  finally  : evaluate the seed prompt and the best prompt on the held-out TEST papers

Grading: numeric answers are scored in code against the reference (tolerance);
text answers are graded by a cross-provider judge panel against the reference (majority vote);
"not reported" golds are correct only if the model abstains. Judge disagreements are flagged.

Subcommands
  draft   draft 5 questions + reference answers per paper (for HUMAN verification)
  run     run the experiment
  report  rebuild plots / report.md from an existing run directory

Env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY)
"""
from __future__ import annotations

import argparse, base64, csv, hashlib, json, math, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# Config / corpus / questions
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        if yaml is None:
            sys.exit("pip install pyyaml (or use a .json config)")
        return yaml.safe_load(text)
    return json.loads(text)


def load_papers(repo: Path) -> dict:
    """paper_id -> metadata, from LitPro/paper/manifest.json."""
    manifest = json.loads((repo / "paper" / "manifest.json").read_text())
    papers = {}
    for p in manifest["papers"]:
        pid = f"{p['field']}_{int(p['number']):02d}"
        pdf = repo / "paper" / p["files"][0]["file"]
        papers[pid] = dict(paper_id=pid, area=p["field"], number=int(p["number"]),
                           title=p["title"], doi=p["doi"], path=str(pdf),
                           pages=p.get("verification", {}).get("pdf_pages"))
    return papers


def load_questions(path: Path, papers: dict, allow_unverified: bool) -> list[dict]:
    qs = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        q = json.loads(line)
        for k in ("qid", "paper_id", "question", "answer_type", "reference"):
            if k not in q:
                sys.exit(f"{path}:{i+1} missing '{k}'")
        if q["paper_id"] not in papers:
            sys.exit(f"{path}:{i+1} unknown paper_id {q['paper_id']}")
        if q["answer_type"] not in ("numeric", "text", "not_reported"):
            sys.exit(f"{path}:{i+1} answer_type must be numeric|text|not_reported")
        if not q.get("verified", False) and not allow_unverified:
            sys.exit(f"{path}:{i+1} ({q['qid']}) is not verified. Have an expert check it, "
                     f"set \"verified\": true, or pass --allow-unverified for a smoke test.")
        q["area"] = papers[q["paper_id"]]["area"]
        q.setdefault("qtype", "unspecified")
        qs.append(q)
    return qs


def split_papers(paper_ids: list[str], train_frac: float, seed: int,
                 qtypes_by_paper: dict | None = None, mode: str = "stratified") -> tuple[list, list]:
    """random: seeded shuffle. stratified: among all splits of the right size, pick the one whose
    question-type counts in train are closest to train_frac of the total for every type, requiring each
    type that occurs at least twice to appear in both train and test. Ties broken by the seed."""
    ids = sorted(paper_ids)
    k = max(1, round(len(ids) * train_frac))
    if mode != "stratified" or not qtypes_by_paper:
        rng = random.Random(seed); rng.shuffle(ids)
        return sorted(ids[:k]), sorted(ids[k:])
    from itertools import combinations
    from collections import Counter
    per = {p: Counter(qtypes_by_paper.get(p, [])) for p in ids}
    total = sum(per.values(), Counter())
    rng = random.Random(seed)
    combos = list(combinations(ids, k)) if math.comb(len(ids), k) <= 200000 else \
        [tuple(sorted(rng.sample(ids, k))) for _ in range(20000)]
    def cost(tr):
        c = sum((per[p] for p in tr), Counter())
        miss = sum(1 for t, n in total.items() if n >= 2 and (c[t] == 0 or c[t] == n))
        return (miss, sum(abs(c[t] - n * train_frac) for t, n in total.items()))
    scored = sorted(((cost(tr), rng.random(), tr) for tr in combos))
    tr = list(scored[0][2])
    return sorted(tr), sorted(p for p in ids if p not in tr)


_pdf_cache: dict[str, bytes] = {}
_pdf_lock = threading.Lock()

def pdf_bytes(path: str) -> bytes:
    with _pdf_lock:
        if path not in _pdf_cache:
            _pdf_cache[path] = Path(path).read_bytes()
        return _pdf_cache[path]

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@dataclass
class CallResult:
    text: str = ""
    in_tokens: int = 0        # all input tokens, cached included
    cached_tokens: int = 0    # input tokens served from cache (read)
    cache_write_tokens: int = 0
    out_tokens: int = 0       # all output tokens, reasoning/thinking included
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None
    truncated: bool = False   # output hit the max-token cap

    @property
    def total_tokens(self) -> int:
        return self.in_tokens + self.out_tokens


class Providers:
    """Thin wrappers around the three SDKs (+ a mock for dry runs)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._clients = {}
        self._sems = {p: threading.Semaphore(n) for p, n in
                      cfg.get("concurrency", {"anthropic": 4, "openai": 4, "gemini": 4, "mock": 16}).items()}
        self.max_retries = cfg.get("max_retries", 4)

    def model(self, name: str) -> dict:
        m = self.cfg["models"][name]
        m.setdefault("name", name)
        return m

    def _client(self, provider: str):
        if provider not in self._clients:
            if provider == "anthropic":
                import anthropic
                # identity-linked / multi-workspace keys need the workspace id on every request
                ws = os.environ.get("ANTHROPIC_WORKSPACE_ID") or next(
                    (m.get("workspace_id") for m in self.cfg["models"].values()
                     if m.get("provider") == "anthropic" and m.get("workspace_id")), None)
                self._clients[provider] = anthropic.Anthropic(
                    default_headers={"anthropic-workspace-id": ws} if ws else None)
            elif provider == "openai":
                import openai
                self._clients[provider] = openai.OpenAI()
            elif provider == "gemini":
                from google import genai
                self._clients[provider] = genai.Client(
                    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            elif provider == "mock":
                self._clients[provider] = None
        return self._clients[provider]

    def call(self, model_name: str, system: str, user: str, pdf_path: str | None = None,
             max_tokens: int | None = None, cache_key: str | None = None) -> CallResult:
        m = self.model(model_name)
        prov = m["provider"]
        max_tokens = max_tokens or m.get("max_tokens", 8000)
        sem = self._sems.setdefault(prov, threading.Semaphore(4))
        last_err = None
        for attempt in range(self.max_retries + 1):
            with sem:
                t0 = time.time()
                try:
                    r = getattr(self, f"_call_{prov}")(m, system, user, pdf_path, max_tokens, cache_key)
                    r.latency_s = time.time() - t0
                    return r
                except Exception as e:  # rate limits, 5xx, timeouts ...
                    last_err = f"{type(e).__name__}: {e}"
                    code = getattr(e, "status_code", None) or getattr(e, "code", None)
                    if code in (400, 401, 403, 404) or type(e).__name__ in (
                            "AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError"):
                        return CallResult(error=last_err[:1000])   # retrying will not help
            time.sleep(min(60, 2 ** attempt + random.random()))
        return CallResult(error=last_err)

    # --- Anthropic: PDF as a base64 document block, cached so per-question calls reuse it
    def _call_anthropic(self, m, system, user, pdf_path, max_tokens, cache_key):
        content = []
        if pdf_path:
            content.append({"type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf",
                                       "data": base64.b64encode(pdf_bytes(pdf_path)).decode()},
                            "cache_control": {"type": "ephemeral"}})
        content.append({"type": "text", "text": user})
        resp = self._client("anthropic").messages.create(
            model=m["model_id"], max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": content}], **m.get("extra", {}))
        u = resp.usage
        read = getattr(u, "cache_read_input_tokens", 0) or 0
        write = getattr(u, "cache_creation_input_tokens", 0) or 0
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return CallResult(text=text, in_tokens=u.input_tokens + read + write, cached_tokens=read,
                          cache_write_tokens=write, out_tokens=u.output_tokens,
                          truncated=getattr(resp, "stop_reason", "") == "max_tokens")

    # --- OpenAI Responses API: PDF as input_file (file_data), prefix caching is automatic
    def _call_openai(self, m, system, user, pdf_path, max_tokens, cache_key):
        content = []
        if pdf_path:
            f = {"type": "input_file", "filename": Path(pdf_path).name,
                 "file_data": "data:application/pdf;base64," + base64.b64encode(pdf_bytes(pdf_path)).decode()}
            if m.get("file_detail"):
                f["detail"] = m["file_detail"]
            content.append(f)
        content.append({"type": "input_text", "text": user})
        kw = dict(model=m["model_id"], instructions=system, max_output_tokens=max_tokens,
                  input=[{"role": "user", "content": content}], **m.get("extra", {}))
        if cache_key:
            kw["prompt_cache_key"] = cache_key
        resp = self._client("openai").responses.create(**kw)
        u = resp.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        reas = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        return CallResult(text=resp.output_text or "", in_tokens=u.input_tokens, cached_tokens=cached,
                          out_tokens=u.output_tokens, reasoning_tokens=reas,
                          truncated=getattr(resp, "status", "") == "incomplete")

    # --- Gemini: PDF inline as bytes (files here are < 9 MB), implicit caching
    def _call_gemini(self, m, system, user, pdf_path, max_tokens, cache_key):
        from google.genai import types
        parts = []
        if pdf_path:
            parts.append(types.Part.from_bytes(data=pdf_bytes(pdf_path), mime_type="application/pdf"))
        parts.append(user)
        cfg = dict(system_instruction=system, max_output_tokens=max_tokens)
        if m.get("thinking_budget") is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=m["thinking_budget"])
        if m.get("thinking_level") is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=m["thinking_level"])
        cfg.update(m.get("extra", {}))
        resp = self._client("gemini").models.generate_content(
            model=m["model_id"], contents=parts, config=types.GenerateContentConfig(**cfg))
        u = resp.usage_metadata
        thoughts = u.thoughts_token_count or 0
        fr = str(getattr(resp.candidates[0], "finish_reason", "") if resp.candidates else "")
        return CallResult(text=resp.text or "", in_tokens=u.prompt_token_count or 0,
                          cached_tokens=u.cached_content_token_count or 0,
                          out_tokens=(u.candidates_token_count or 0) + thoughts, reasoning_tokens=thoughts,
                          truncated="MAX_TOKENS" in fr)

    # --- Mock: exercises the whole pipeline without API keys
    def _call_mock(self, m, system, user, pdf_path, max_tokens, cache_key):
        rng = random.Random(hash((system, user)) ^ int(time.time() * 1e6))
        time.sleep(rng.uniform(0.01, 0.05))
        if "GRADER" in system:
            text = json.dumps({"verdict": rng.choice([0, 1, 1]), "reason": "mock"})
        elif "PROMPT-OPTIMIZER" in system:
            text = json.dumps({"reasoning": "mock reasoning",
                               "changes": ["mock change"],
                               "partition": rng.choice(["joint", "per_question"]),
                               "instructions": "You are a careful reader. Quote evidence. "
                                               f"Prefer tables for numbers. (variant {rng.randint(0, 999)})"})
        elif "QUESTION-WRITER" in system:
            text = json.dumps({"questions": [{"question": f"Mock question {i}?", "answer_type": "numeric",
                                              "qtype": "text", "reference": {"value": 1.0, "unit": "mV"},
                                              "evidence": "", "location": "Results"} for i in range(6)]})
        else:
            qids = re.findall(r"\[qid=([^\]]+)\]", user)
            ans = []
            for q in qids:
                if rng.random() < 0.15:
                    ans.append({"qid": q, "status": "not_reported", "answer": "", "unit": "",
                                "evidence": "", "location": "", "confidence": 0})
                else:
                    ans.append({"qid": q, "status": "answered",
                                "answer": str(round(rng.choice([1.0, 1.0, 2.0, 240]), 2)), "unit": "mV",
                                "evidence": "mock", "location": "Fig. 1", "confidence": 0.8})
            text = json.dumps({"answers": ans})
        n_in = (60000 if pdf_path else 0) + len(system + user) // 4
        return CallResult(text=text, in_tokens=n_in, cached_tokens=int(n_in * rng.random() * 0.5),
                          out_tokens=len(text) // 4 + rng.randint(0, 800))


def cost_usd(cfg: dict, model_name: str, r: CallResult) -> float | None:
    p = cfg["models"][model_name].get("price_per_mtok")
    if not p or p.get("input") is None:
        return None
    uncached = r.in_tokens - r.cached_tokens - r.cache_write_tokens
    c = uncached * p["input"] + r.cached_tokens * p.get("cached_input", p["input"]) \
        + r.cache_write_tokens * p.get("cache_write", p["input"]) + r.out_tokens * p["output"]
    return c / 1e6

# ---------------------------------------------------------------------------
# Logging / accounting
# ---------------------------------------------------------------------------

class Ledger:
    """Append-only call log + per-bucket token/cost/time accounting."""

    def __init__(self, run_dir: Path, cfg: dict):
        self.path = run_dir / "calls.jsonl"
        self.cfg = cfg
        self.lock = threading.Lock()
        self.buckets: dict[str, dict] = {}

    def record(self, bucket: str, role: str, model: str, r: CallResult, **meta) -> None:
        c = cost_usd(self.cfg, model, r)
        row = dict(ts=datetime.now().isoformat(timespec="seconds"), bucket=bucket, role=role, model=model,
                   in_tokens=r.in_tokens, cached_tokens=r.cached_tokens, cache_write_tokens=r.cache_write_tokens,
                   out_tokens=r.out_tokens, reasoning_tokens=r.reasoning_tokens,
                   latency_s=round(r.latency_s, 3), cost_usd=c, error=r.error, truncated=r.truncated, **meta)
        with self.lock:
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            b = self.buckets.setdefault(bucket, dict(tokens=0, in_tokens=0, cached_tokens=0, out_tokens=0,
                                                     cost=0.0, cost_known=True, api_s=0.0, calls=0, errors=0))
            b["tokens"] += r.total_tokens; b["in_tokens"] += r.in_tokens
            b["cached_tokens"] += r.cached_tokens; b["out_tokens"] += r.out_tokens
            b["api_s"] += r.latency_s; b["calls"] += 1; b["errors"] += bool(r.error)
            if c is None:
                b["cost_known"] = False
            else:
                b["cost"] += c

    def snapshot(self, bucket: str) -> dict:
        with self.lock:
            return dict(self.buckets.get(bucket, dict(tokens=0, in_tokens=0, cached_tokens=0, out_tokens=0,
                                                      cost=0.0, cost_known=True, api_s=0.0, calls=0, errors=0)))

    @staticmethod
    def diff(a: dict, b: dict) -> dict:
        return {k: (b[k] - a[k]) if isinstance(b[k], (int, float)) and not isinstance(b[k], bool) else b[k]
                for k in b}

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def parse_json(text: str):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

OUTPUT_CONTRACT = """\
Return ONLY a JSON object, no prose and no code fences:
{"answers": [{"qid": "<qid>", "status": "answered" | "not_reported",
              "answer": "<value or short answer; numbers without units>", "unit": "<unit or empty>",
              "evidence": "<verbatim quote from the paper, <= 30 words>",
              "location": "<section / table / figure>", "confidence": <0..1>}]}
One entry per qid. If the paper does not state the answer, use status "not_reported" and leave answer empty.
Keep each evidence quote on one line: collapse table spacing to single spaces, never pad with whitespace."""


def question_line(q: dict) -> str:
    unit = q["reference"].get("unit") if q["answer_type"] == "numeric" else None
    s = f"[qid={q['qid']}] {q['question']}"
    if unit:
        s += f" (report the number in {unit})"
    return s


def extraction_message(qs: list[dict]) -> str:
    return "Questions about the attached paper:\n" + "\n".join(question_line(q) for q in qs) \
        + "\n\n" + OUTPUT_CONTRACT


def run_extraction(P: Providers, L: Ledger, model: str, prompt: dict, paper: dict, qs: list[dict],
                   bucket: str, meta: dict, closed_book: bool = False) -> dict[str, dict]:
    """Returns qid -> raw answer dict. Joint: one call per paper. Per-question: sequential calls
    (so the cached PDF prefix is warm after the first)."""
    groups = [qs] if prompt["partition"] == "joint" else [[q] for q in qs]
    out = {}
    pre = (f"No paper is attached. Answer from what you already know about the paper \"{paper['title']}\" "
           f"(doi {paper['doi']}).\n\n") if closed_book else ""
    for g in groups:
        msg = pre + extraction_message(g)
        for attempt in range(2):
            r = P.call(model, prompt["instructions"], msg, pdf_path=None if closed_book else paper["path"],
                       max_tokens=P.model(model).get("max_tokens"), cache_key=f"litpro-{paper['paper_id']}")
            L.record(bucket, "extract" if attempt == 0 else "extract_retry", model, r,
                     paper_id=paper["paper_id"], qids=[q["qid"] for q in g], **meta)
            parsed = parse_json(r.text) or {}
            got = {a.get("qid"): a for a in parsed.get("answers", []) if isinstance(a, dict)}
            if r.error or all(q["qid"] in got for q in g) or (len(g) == 1 and len(got) == 1):
                break
            msg = (pre + extraction_message(g) + "\n\nYour previous reply was "
                   + ("cut off by the length limit" if r.truncated else "not valid JSON or missed some qids")
                   + ". Reply again with ONLY the JSON object, short one-line evidence quotes.")
        for q in g:
            a = got.get(q["qid"])
            if a is None and len(g) == 1 and len(got) == 1:   # model mangled the qid
                a = next(iter(got.values()))
            out[q["qid"]] = a or {"status": "error", "answer": "", "raw": (r.error or r.text)[:500]}
    return out

# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"[-−–]?\d+(?:[.,]\d+)?(?:\s*(?:[eE]|[x×]\s*10\^?)\s*[-−–]?\d+)?")

def parse_number(s) -> float | None:
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None
    m = NUM_RE.search(s.replace("−", "-").replace("–", "-"))
    if not m:
        return None
    t = m.group(0).replace(",", "").replace(" ", "")
    t = re.sub(r"[x×]10\^?", "e", t)
    try:
        return float(t)
    except ValueError:
        return None


def tolerance(ref: dict, cfg: dict) -> float:
    """Absolute tolerance for a numeric reference. Null fields count as missing: rel_tol and abs_tol are
    combined if both are given; abs_tol alone is used as is; neither -> default_rel_tol."""
    v = abs(float(ref["value"]))
    rel, ab = ref.get("rel_tol"), ref.get("abs_tol")
    if rel is None and ab is None:
        rel = cfg.get("default_rel_tol", 0.05)
    return max(v * float(rel or 0), float(ab or 0))


def is_abstain(a: dict) -> bool:
    if a.get("status") == "not_reported":
        return True
    ans = str(a.get("answer", "")).strip().lower()
    # "none" is NOT an abstention: it is a real answer to questions like "what treatment was applied?"
    return ans in ("", "not reported", "not stated", "not given", "not provided", "n/a", "unknown")


JUDGE_SYSTEM = """You are a GRADER. You compare a candidate answer to an expert reference answer for a
question about a scientific paper. You do not have the paper; the reference is ground truth.
Grade 1 if the candidate states the same fact as the reference (equivalent wording, units or notation are
fine; extra correct detail is fine), 0 if it is wrong, incomplete on a required element, hedges between
alternatives, or contradicts the reference. Apply the acceptance criteria if given.
The acceptance criteria describe required CONTENT, not exact wording: equivalent section names ("main text"
for "Results"), a reference electrode or unit left implicit when the question already fixes it, and
rounding within the stated tolerance all count as correct.
Return ONLY JSON: {"verdict": 0 or 1, "reason": "<one sentence>"}"""


class JudgeCache:
    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.d = json.loads(path.read_text()) if path.exists() else {}

    def key(self, judge, q, a):
        norm = (str(a.get("answer", "")).strip().lower() + "|" + str(a.get("unit", "")).strip().lower())
        return hashlib.sha1(f"{judge}|{q['qid']}|{norm}".encode()).hexdigest()

    def get(self, k):
        with self.lock:
            return self.d.get(k)

    def put(self, k, v):
        with self.lock:
            self.d[k] = v

    def save(self):
        with self.lock:
            self.path.write_text(json.dumps(self.d))


def grade(P: Providers, L: Ledger, J: JudgeCache, cfg: dict, q: dict, a: dict, meta: dict) -> dict:
    """outcome: correct | wrong | abstain | error. method: rule | numeric | judges."""
    ref, at = q["reference"], q["answer_type"]
    if a.get("status") == "error":
        return dict(outcome="error", method="rule", verdicts={}, flag=False)
    if at == "not_reported":
        return dict(outcome="correct" if is_abstain(a) else "wrong", method="rule", verdicts={}, flag=False)
    if is_abstain(a):
        return dict(outcome="abstain", method="rule", verdicts={}, flag=False)
    if at == "numeric" and ref.get("value") is not None:
        ans_s = str(a.get("answer", ""))
        nums = {round(float(parse_number(m) or 0), 9) for m in NUM_RE.findall(ans_s)}
        wordy = re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten|twice|half)(fold)?\b", ans_s, re.I)
        x = parse_number(a.get("answer")) if (len(nums) <= 1 and not wordy) else None   # ambiguous -> judges
        if x is not None:
            v = float(ref["value"])
            tol = tolerance(ref, cfg)
            if ref.get("log_tol") and x > 0 and v > 0:
                ok = abs(math.log10(x) - math.log10(v)) <= ref["log_tol"]
            else:
                ok = abs(x - v) <= tol + 1e-12
            return dict(outcome="correct" if ok else "wrong", method="numeric", verdicts={}, flag=False)
    # text (or unparseable number) -> judge panel against the reference
    ref_txt = ref.get("text") or (f"{ref.get('value')} {ref.get('unit', '')}".strip())
    if at == "numeric" and ref.get("value") is not None:
        tol = tolerance(ref, cfg)
        ref_txt = f"{ref.get('value')} {ref.get('unit', '')} (any value within ±{tol:.3g} is correct)"
    user = (f"Question: {q['question']}\nReference answer: {ref_txt}\n"
            + (f"Acceptance criteria: {ref['accept']}\n" if ref.get("accept") else "")
            + f"Candidate answer: {a.get('answer', '')} {a.get('unit', '') or ''}".strip())
    verdicts, reasons = {}, {}
    for judge in cfg["judges"]:
        k = J.key(judge, q, a)
        hit = J.get(k)
        if hit is None:
            r = P.call(judge, JUDGE_SYSTEM, user, max_tokens=P.model(judge).get("judge_max_tokens", 2000))
            L.record("judges", "judge", judge, r, qid=q["qid"], **meta)
            j = parse_json(r.text) or {}
            hit = {"verdict": j.get("verdict"), "reason": j.get("reason", r.error or "")}
            if hit["verdict"] in (0, 1):
                J.put(k, hit)
        verdicts[judge], reasons[judge] = hit["verdict"], hit["reason"]
    valid = [v for v in verdicts.values() if v in (0, 1)]
    ones = sum(valid)
    ok = len(valid) > 0 and ones * 2 > len(valid)       # strict majority; ties -> 0
    flag = len(set(valid)) > 1                            # judges that answered disagree
    return dict(outcome="correct" if ok else "wrong", method="judges", verdicts=verdicts,
                reasons=reasons, flag=flag, judges_missing=len(verdicts) - len(valid))

# ---------------------------------------------------------------------------
# Evaluation of one prompt on a set of papers
# ---------------------------------------------------------------------------

class ExtractionFailed(RuntimeError):
    pass


def evaluate(P, L, J, cfg, model, prompt, papers, qs_by_paper, paper_ids, bucket, meta, run_dir,
             closed_book=False, items_file="items.jsonl"):
    snap0 = L.snapshot(bucket)
    t0 = time.time()
    with ThreadPoolExecutor(cfg.get("paper_workers", 6)) as ex:
        futs = {pid: ex.submit(run_extraction, P, L, model, prompt, papers[pid], qs_by_paper[pid], bucket, meta,
                               closed_book) for pid in paper_ids}
        answers = {pid: f.result() for pid, f in futs.items()}
    extract_wall = time.time() - t0
    usage = Ledger.diff(snap0, L.snapshot(bucket))
    errs = [a.get("raw", "") for pa in answers.values() for a in pa.values() if a.get("status") == "error"]
    n_ans = sum(len(pa) for pa in answers.values())
    if errs and len(errs) == n_ans:
        raise ExtractionFailed(f"all {n_ans} answers failed for {model}; first error: {errs[0][:400]}")
    if errs:
        print(f"  ! {model} {meta.get('split')} iter {meta.get('iter')}: {len(errs)}/{n_ans} answers failed "
              f"(first: {errs[0][:200]})", flush=True)

    items = []
    def _g(pid, q):
        a = answers[pid][q["qid"]]
        try:
            g = grade(P, L, J, cfg, q, a, meta)
        except Exception as e:
            print(f"  ! grading failed for {q['qid']}: {type(e).__name__}: {e}", flush=True)
            g = dict(outcome="error", method="grade_error", verdicts={}, flag=False)
        row = dict(meta)
        row.update(qid=q["qid"], paper_id=pid, area=q["area"], qtype=q["qtype"], answer_type=q["answer_type"],
                   answer=a.get("answer"), unit=a.get("unit"), status=a.get("status"),
                   evidence=a.get("evidence"), location=a.get("location"), confidence=a.get("confidence"), **g)
        return row
    with ThreadPoolExecutor(cfg.get("judge_workers", 8)) as ex:
        items = list(ex.map(lambda t: _g(*t), [(pid, q) for pid in paper_ids for q in qs_by_paper[pid]]))
    J.save()
    with (run_dir / items_file).open("a") as f:
        for it in items:
            f.write(json.dumps(it, default=str) + "\n")
    n = len(items)
    c = sum(it["outcome"] == "correct" for it in items)
    ab = sum(it["outcome"] == "abstain" for it in items)
    answered = sum(it["outcome"] in ("correct", "wrong") for it in items)
    jm = sum(it.get("judges_missing", 0) for it in items)
    if jm:
        print(f"  ! {jm} judge verdicts missing in this evaluation (see calls.jsonl, role=judge)", flush=True)
    return dict(score=c / n if n else 0.0, n=n, correct=c, abstain_rate=ab / n if n else 0,
                precision=(c / answered) if answered else None,
                errors=sum(it["outcome"] == "error" for it in items),
                judge_flags=sum(bool(it.get("flag")) for it in items),
                extract_wall_s=extract_wall, usage=usage, items=items)

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

OPT_SYSTEM = """You are a PROMPT-OPTIMIZER. A language model reads one scientific paper (PDF) and answers
questions about it. You improve its SYSTEM INSTRUCTIONS so it answers more questions correctly on NEW,
unseen papers in the same research area.

You see the current instructions, the train score, and the items it got wrong or abstained on, with the
expert reference answers. Diagnose the failure patterns, then rewrite the instructions.

Hard rules:
- Rules must be GENERAL. Never mention a specific paper, compound, sample name, numeric value, table or
  figure number from the feedback. Write rules that would help on a different paper in this area.
- Do not describe the JSON output format; the harness appends it.
- Keep the instructions under {max_words} words. Shorter is better when equally good.
- You may also choose the partition: "joint" (all questions about a paper in one call) or
  "per_question" (one call per question; costs more tokens, may help focus).
Return ONLY JSON:
{"diagnosis": "<2-4 sentences>", "changes": ["<change 1>", ...],
 "partition": "joint" | "per_question", "instructions": "<the full new instructions>"}"""

OPT_SYSTEM_SCORE = """You are a PROMPT-OPTIMIZER agent. A language model reads one scientific paper (PDF) and
answers a set of questions about it. You write its SYSTEM INSTRUCTIONS. Your goal is to maximize its score.

After each attempt you receive ONLY a score: the fraction of questions answered correctly over a fixed set
of training papers. You do not see the questions, the answers, or which ones failed. Use the history of
your attempts and their scores to decide what to try next: change wording, emphasis, structure, level of
detail, or how the questions are sent. Explore when progress stalls; exploit what has worked.

Constraints:
- The harness appends the questions and a fixed JSON output format; do not describe the output format.
- Keep the instructions under {max_words} words.
- partition: "joint" sends all questions about a paper in one call; "per_question" sends one call per
  question (about 5x the cost per evaluation). Each evaluation is charged to your budget.
Return ONLY JSON:
{"reasoning": "<2-4 sentences: what you infer from the scores and what you will try>",
 "changes": ["<change 1>", ...], "partition": "joint" | "per_question",
 "instructions": "<the full new instructions>"}"""

SPECIFIC_REF_RE = re.compile(r"\b(?:Table|Tab\.|Fig\.?|Figure|Supplementary\s+(?:Table|Fig\.?|Figure))\s*S?\d+",
                             re.I)


def leak_check(instr: str, train_qs: list[dict], papers: dict) -> list[str]:
    """Return reasons the candidate prompt looks paper-specific."""
    probs = []
    if SPECIFIC_REF_RE.search(instr):
        probs.append(f"mentions a specific table/figure: '{SPECIFIC_REF_RE.search(instr).group(0)}'")
    low = instr.lower()
    for q in train_qs:
        ref = q["reference"]
        for s in [ref.get("text"), q.get("evidence")]:
            if s and len(s) >= 12 and s.lower()[:40] in low:
                probs.append(f"copies reference/evidence text of {q['qid']}")
        v = ref.get("value")
        if v is not None:
            vs = f"{v:g}" if isinstance(v, (int, float)) else str(v)
            if len(vs.replace(".", "").replace("-", "")) >= 3 and re.search(rf"(?<![\d.]){re.escape(vs)}(?![\d])", instr):
                probs.append(f"contains reference value {vs} ({q['qid']})")
    for pid in {q["paper_id"] for q in train_qs}:
        title = papers[pid]["title"].lower()
        if title[:50] in low:
            probs.append(f"mentions paper title {pid}")
    return sorted(set(probs))


def feedback_block(ev: dict, qmap: dict, max_items: int) -> str:
    bad = [it for it in ev["items"] if it["outcome"] != "correct"]
    random.Random(0).shuffle(bad)
    lines = []
    for it in bad[:max_items]:
        q = qmap[it["qid"]]
        ref = q["reference"]
        ref_txt = ref.get("text") or f"{ref.get('value')} {ref.get('unit', '')}".strip()
        if q["answer_type"] == "not_reported":
            ref_txt = "NOT REPORTED in this paper (correct behavior is to abstain)"
        why = "; ".join(f"{v}" for v in (it.get("reasons") or {}).values() if v)
        lines.append(json.dumps(dict(question=q["question"], question_type=q["qtype"], outcome=it["outcome"],
                                     reference=ref_txt, model_status=it.get("status"),
                                     model_answer=f"{it.get('answer') or ''} {it.get('unit') or ''}".strip(),
                                     model_evidence=it.get("evidence"), model_location=it.get("location"),
                                     model_confidence=it.get("confidence"), grader_notes=why or None),
                                ensure_ascii=False))
    by_type = {}
    for it in ev["items"]:
        t = by_type.setdefault(it["qtype"], [0, 0])
        t[0] += it["outcome"] == "correct"; t[1] += 1
    type_line = ", ".join(f"{k}: {a}/{b}" for k, a, b in ((k, *v) for k, v in sorted(by_type.items())))
    return (f"Train score: {ev['correct']}/{ev['n']}  (abstain rate {ev['abstain_rate']:.0%})\n"
            f"By question type: {type_line}\n\nFailed items ({len(bad)} total, showing {min(len(bad), max_items)}):\n"
            + "\n".join(lines))


def propose(P, L, cfg, opt_model, area, current, ev, history, qmap, train_qs, papers, bucket, meta,
            budget_note=None):
    score_only = cfg.get("feedback", "score") == "score"
    if score_only:
        hist = "\n\n".join(
            f"Attempt {h['iter']} (partition={h['partition']}): score {h['score']:.3f}"
            + (" <- current best" if h.get("is_best") else "") + f"\n<<<\n{h.get('instructions', '')}\n>>>"
            for h in history)
        user = (f"Research area: {area}. Training set: {ev['n']} questions over "
                f"{len({i['paper_id'] for i in ev['items']})} papers.\n\nYour attempts so far:\n\n{hist}"
                + (f"\n\n{budget_note}" if budget_note else ""))
        system = OPT_SYSTEM_SCORE.replace("{max_words}", str(cfg.get("max_prompt_words", 600)))
    else:
        hist = "\n".join(f"- iter {h['iter']}: score {h['score']:.3f}, partition {h['partition']}, "
                         f"{'ACCEPTED' if h['accepted'] else 'rejected'}; changes: {h.get('changes')}"
                         for h in history) or "(none)"
        user = (f"Research area: {area}\n\nCurrent best instructions (partition={current['partition']}):\n"
                f"<<<\n{current['instructions']}\n>>>\n\nHistory of attempts:\n{hist}\n\n"
                f"{feedback_block(ev, qmap, cfg.get('feedback_items', 25))}"
                + (f"\n\n{budget_note}" if budget_note else ""))
        system = OPT_SYSTEM.replace("{max_words}", str(cfg.get("max_prompt_words", 600)))
    for attempt in range(2):
        r = P.call(opt_model, system, user, max_tokens=P.model(opt_model).get("optimizer_max_tokens", 16000))
        L.record(bucket, "optimize", opt_model, r, attempt=attempt, **meta)
        j = parse_json(r.text) or {}
        instr, part = (j.get("instructions") or "").strip(), j.get("partition", current["partition"])
        if not instr:
            user += "\n\nYour previous reply was not valid JSON with an 'instructions' field. Try again."
            continue
        probs = [] if score_only else leak_check(instr, train_qs, papers)
        if probs:
            user += ("\n\nYour previous proposal was rejected because it is paper-specific: "
                     + "; ".join(probs) + ". Rewrite it as general rules.")
            continue
        return dict(instructions=instr, partition=part if part in ("joint", "per_question") else "joint",
                    diagnosis=j.get("diagnosis") or j.get("reasoning"), changes=j.get("changes"))
    return None

# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_loop(P, L, J, cfg, run_dir, area, model, papers, qs, train_ids, test_ids, writer):
    budget = cfg["budget"]
    opt_model = model if cfg.get("optimizer", "same") == "same" else cfg["optimizer"]
    bucket = f"loop:{area}:{model}"
    qmap = {q["qid"]: q for q in qs}
    qs_by_paper = {}
    for q in qs:
        qs_by_paper.setdefault(q["paper_id"], []).append(q)
    train_qs = [q for q in qs if q["paper_id"] in train_ids]
    seed_prompt = dict(instructions=cfg["seed_prompt"].strip(), partition=cfg.get("seed_partition", "joint"))
    pdir = run_dir / "prompts"; pdir.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{area}__{model}")

    cum = dict(tokens=0, cost=0.0, cost_known=True, runtime_s=0.0)
    def account(usage, wall_s):
        cum["tokens"] += usage["tokens"]; cum["runtime_s"] += wall_s
        cum["cost"] += usage["cost"]; cum["cost_known"] &= usage["cost_known"]

    def write_row(it, split, prompt, ev, accepted, opt_usage=None, opt_wall=0.0):
        row = dict(area=area, model=model, iter=it, split=split, partition=prompt["partition"],
                   prompt_words=len(prompt["instructions"].split()), accepted=accepted,
                   score=round(ev["score"], 4), correct=ev["correct"], n=ev["n"],
                   precision=None if ev["precision"] is None else round(ev["precision"], 4),
                   abstain_rate=round(ev["abstain_rate"], 4), errors=ev["errors"], judge_flags=ev["judge_flags"],
                   eval_tokens=ev["usage"]["tokens"], eval_in_tokens=ev["usage"]["in_tokens"],
                   eval_cached_tokens=ev["usage"]["cached_tokens"], eval_out_tokens=ev["usage"]["out_tokens"],
                   eval_cost=round(ev["usage"]["cost"], 5) if ev["usage"]["cost_known"] else None,
                   eval_wall_s=round(ev["extract_wall_s"], 2), eval_api_s=round(ev["usage"]["api_s"], 2),
                   opt_tokens=(opt_usage or {}).get("tokens", 0),
                   opt_cost=round((opt_usage or {}).get("cost", 0.0), 5),
                   opt_wall_s=round(opt_wall, 2),
                   cum_tokens=cum["tokens"] if split == "train" else None,
                   cum_cost=round(cum["cost"], 5) if (split == "train" and cum["cost_known"]) else None,
                   cum_runtime_s=round(cum["runtime_s"], 2) if split == "train" else None)
        writer(row)
        return row

    def save_prompt(it, prompt, extra=None):
        (pdir / f"{safe}__iter{it}.json").write_text(json.dumps(dict(prompt, **(extra or {})), indent=2))

    meta = dict(area=area, extractor=model)
    # iter 0
    ev = evaluate(P, L, J, cfg, model, seed_prompt, papers, qs_by_paper, train_ids, bucket,
                  dict(meta, iter=0, split="train"), run_dir)
    account(ev["usage"], ev["usage"]["api_s"])
    save_prompt(0, seed_prompt)
    write_row(0, "train", seed_prompt, ev, True)
    best, best_ev = dict(seed_prompt, iter=0), ev
    history = [dict(iter=0, score=ev["score"], partition=seed_prompt["partition"], accepted=True,
                    instructions=seed_prompt["instructions"], is_best=True, changes="seed prompt")]
    seed_train_score = ev["score"]
    last_eval_tokens = {seed_prompt["partition"]: ev["usage"]["tokens"]}
    last_opt_tokens = 0
    stop_reason = "max_iters"

    q_per_paper = max(1, round(len(train_qs) / max(1, len(train_ids))))
    def est_eval(part):
        if part in last_eval_tokens:
            return last_eval_tokens[part]
        j = last_eval_tokens.get("joint")
        if j is not None and part == "per_question":
            return j * q_per_paper          # upper bound (ignores cache discounts)
        pq = last_eval_tokens.get("per_question")
        return pq // q_per_paper if pq else 0
    # budget meter: dollars if max_usd is set and prices are known, else tokens
    use_usd = bool(budget.get("max_usd")) and cum["cost_known"]
    if budget.get("max_usd") and not use_usd:
        print(f"  ! {model}: max_usd set but prices missing in config; using max_tokens instead", flush=True)
    cap = budget.get("max_usd") if use_usd else budget.get("max_tokens")
    unit = "USD" if use_usd else "tokens"
    used = lambda: cum["cost"] if use_usd else cum["tokens"]
    last_eval_cost = {seed_prompt["partition"]: ev["usage"]["cost"]}
    last_opt_cost = 0.0
    def est(part):
        if not use_usd:
            return est_eval(part)
        if part in last_eval_cost:
            return last_eval_cost[part]
        j = last_eval_cost.get("joint")
        if j is not None and part == "per_question":
            return j * q_per_paper
        pq = last_eval_cost.get("per_question")
        return pq / q_per_paper if pq else 0.0
    fmt = (lambda x: f"${x:.2f}") if use_usd else (lambda x: f"{int(x):,}")
    last_opt = lambda: last_opt_cost if use_usd else last_opt_tokens
    min_gain = cfg.get("accept_min_gain", 1)            # keep a proposal if it scores higher than the best
    cheaper_frac = cfg.get("accept_cheaper_frac", 0.2) if cfg.get("accept_cheaper_ties", False) else None
    def eval_price(e):
        return e["usage"]["cost"] if use_usd else e["usage"]["tokens"]

    for it in range(1, budget["max_iters"] + 1):
        # stop if even the cheapest next iteration would exceed the budget
        cheapest = last_opt() + min(est("joint"), est("per_question"))
        if cap and used() + cheapest > cap:
            stop_reason = f"budget (used {fmt(used())} of {fmt(cap)}, cheapest next iteration ~{fmt(cheapest)})"
            break
        budget_note = None
        if cap:
            budget_note = (f"Budget: {fmt(cap - used())} ({unit}) left for this loop. "
                           f"Approximate cost per train evaluation: joint ~{fmt(est('joint'))}, "
                           f"per_question ~{fmt(est('per_question'))}; each of your proposals costs "
                           f"~{fmt(last_opt()) if last_opt() else 'a little'}. Iterations left: "
                           f"{budget['max_iters'] - it + 1}. A proposal is accepted if it gets at least "
                           f"{min_gain} more train item(s) right"
                           + (f", or the same number at least {cheaper_frac:.0%} cheaper." if cheaper_frac else "."))
        s0, t0 = L.snapshot(bucket), time.time()
        cand = propose(P, L, cfg, opt_model, area, best, best_ev, history, qmap, train_qs, papers, bucket,
                       dict(meta, iter=it), budget_note)
        opt_wall, opt_usage = time.time() - t0, Ledger.diff(s0, L.snapshot(bucket))
        account(opt_usage, opt_usage["api_s"])
        last_opt_tokens, last_opt_cost = opt_usage["tokens"], opt_usage["cost"]
        if cand is None:
            history.append(dict(iter=it, score=float("nan"), partition="-", accepted=False,
                                changes="optimizer failed / proposal rejected as paper-specific",
                                instructions="(no valid proposal)"))
            continue
        if cap and used() + est(cand["partition"]) > cap:
            other = "joint" if cand["partition"] == "per_question" else "per_question"
            if used() + est(other) <= cap:    # keep the new instructions, use the partition we can afford
                print(f"  ! {model} iter {it}: {cand['partition']} unaffordable, evaluating with {other}", flush=True)
                cand = dict(cand, partition=other,
                            changes=(cand.get("changes") or []) + [f"partition forced to {other} by budget"])
        if cap and used() + est(cand["partition"]) > cap:
            stop_reason = (f"budget: proposal at iter {it} ({cand['partition']}) would need "
                           f"~{fmt(est(cand['partition']))}, {fmt(cap - used())} left")
            save_prompt(it, cand, dict(train_score=None, accepted=False, note="not evaluated: budget"))
            break
        ev = evaluate(P, L, J, cfg, model, cand, papers, qs_by_paper, train_ids, bucket,
                      dict(meta, iter=it, split="train"), run_dir)
        account(ev["usage"], ev["usage"]["api_s"])
        last_eval_tokens[cand["partition"]] = ev["usage"]["tokens"]
        last_eval_cost[cand["partition"]] = ev["usage"]["cost"]
        gain = ev["correct"] - best_ev["correct"]
        accepted = gain >= min_gain or (cheaper_frac is not None and gain >= 0
                                        and eval_price(ev) <= (1 - cheaper_frac) * eval_price(best_ev))
        save_prompt(it, cand, dict(train_score=ev["score"], accepted=accepted))
        write_row(it, "train", cand, ev, accepted, opt_usage, opt_wall)
        history.append(dict(iter=it, score=ev["score"], partition=cand["partition"], accepted=accepted,
                            changes=cand.get("changes"), instructions=cand["instructions"]))
        if accepted:
            best, best_ev = dict(cand, iter=it), ev
            for h in history:
                h["is_best"] = h["iter"] == it

    # held-out test: seed vs best (measurement only; never fed back to the optimizer)
    tb = f"test:{area}:{model}"
    reps = cfg.get("test_repeats", 1)
    test = {}
    to_test = [("seed", seed_prompt, 0)] + ([("best", best, best["iter"])] if best["iter"] != 0 else [])
    if not test_ids:
        print(f"  ! {area}: no test papers (only {len(train_ids)} paper(s) have questions); skipping test", flush=True)
        to_test = []
    for label, prompt, it in to_test:
        evs = []
        for rep in range(reps):
            e = evaluate(P, L, J, cfg, model, prompt, papers, qs_by_paper, test_ids, tb,
                         dict(meta, iter=it, split="test", which=label, rep=rep), run_dir)
            write_row(it, f"test_{label}", prompt, e, None)
            evs.append(e)
        test[label] = evs
    summary = dict(area=area, model=model, stop_reason=stop_reason, best_iter=best["iter"],
                   best_partition=best["partition"], train_seed=seed_train_score,
                   train_best=best_ev["score"], cum_tokens=cum["tokens"],
                   cum_cost=cum["cost"] if cum["cost_known"] else None, cum_runtime_s=cum["runtime_s"],
                   test_seed=[e["score"] for e in test.get("seed", [])],
                   test_best=[e["score"] for e in test.get("best", test.get("seed", []))],
                   best_instructions=best["instructions"])
    (run_dir / "loops").mkdir(exist_ok=True)
    (run_dir / "loops" / f"{safe}.json").write_text(json.dumps(summary, indent=2))
    print(f"[{area} | {model}] best iter {best['iter']} ({best['partition']}), train {best_ev['score']:.2f}, "
          f"test seed {summary['test_seed']} -> best {summary['test_best']}; stop: {stop_reason}", flush=True)
    return summary


def cmd_run(args):
    cfg = load_config(args.config)
    repo = Path(args.repo)
    papers = load_papers(repo)
    qs = load_questions(Path(args.questions), papers, args.allow_unverified)
    if args.dry_run:
        cfg = mock_config(cfg)
    run_dir = Path(args.out) / (args.name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.used.json").write_text(json.dumps(cfg, indent=2))
    (run_dir / "run_args.json").write_text(json.dumps(dict(questions=str(Path(args.questions).resolve()),
                                                           repo=str(Path(args.repo).resolve())), indent=2))
    P, L = Providers(cfg), Ledger(run_dir, cfg)
    J = JudgeCache(run_dir / "judge_cache.json")

    if args.extractors:
        cfg["extractors"] = args.extractors.split(",")
    if args.max_iters is not None:
        cfg["budget"]["max_iters"] = args.max_iters
    if args.max_tokens is not None:
        cfg["budget"]["max_tokens"] = args.max_tokens
    if args.max_usd is not None:
        cfg["budget"]["max_usd"] = args.max_usd
    if cfg["budget"].get("max_usd"):
        need = set(cfg["extractors"]) | ({cfg["optimizer"]} if cfg.get("optimizer", "same") != "same" else set())
        missing = [m for m in need if (cfg["models"][m].get("price_per_mtok") or {}).get("input") is None
                   or cfg["models"][m]["price_per_mtok"].get("output") is None]
        if missing:
            sys.exit(f"budget.max_usd is set but price_per_mtok is missing for: {', '.join(sorted(missing))}")
    areas = args.areas.split(",") if args.areas else sorted({q["area"] for q in qs})
    split_info = {}
    csv_path = run_dir / "iterations.csv"
    lock = threading.Lock()
    header_written = [csv_path.exists()]
    def writer(row):
        with lock, csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not header_written[0]:
                w.writeheader(); header_written[0] = True
            w.writerow(row)

    summaries = []
    for area in areas:
        aq = [q for q in qs if q["area"] == area]
        pids = sorted({q["paper_id"] for q in aq})
        qt = {}
        for q in aq:
            qt.setdefault(q["paper_id"], []).append(q["qtype"])
        train_ids, test_ids = split_papers(pids, cfg.get("train_frac", 0.5), cfg.get("split_seed", 0),
                                           qt, cfg.get("split_mode", "stratified"))
        from collections import Counter
        cov = {s: dict(sorted(Counter(t for p in ids for t in qt.get(p, [])).items()))
               for s, ids in (("train", train_ids), ("test", test_ids))}
        split_info[area] = dict(mode=cfg.get("split_mode", "stratified"), train=train_ids, test=test_ids,
                                qtype_counts=cov)
        print(f"{area}: {len(aq)} questions ({cfg.get('split_mode', 'stratified')} split)\n"
              f"  train {train_ids} {cov['train']}\n  test  {test_ids} {cov['test']}", flush=True)
        for model in cfg["extractors"]:
            try:
                summaries.append(run_loop(P, L, J, cfg, run_dir, area, model, papers, aq, train_ids, test_ids, writer))
            except ExtractionFailed as e:
                print(f"[{area} | {model}] SKIPPED: {e}", flush=True)
                (run_dir / "failed_loops.txt").open("a").write(f"{area}\t{model}\t{e}\n")
        if cfg.get("closed_book", False):
            qbp = {}
            for q in aq:
                qbp.setdefault(q["paper_id"], []).append(q)
            seed = dict(instructions=cfg["seed_prompt"].strip(), partition="joint")
            for model in cfg["extractors"]:
                try:
                    e = evaluate(P, L, J, cfg, model, seed, papers, qbp, pids, f"closed:{area}:{model}",
                                 dict(area=area, extractor=model, iter=0, split="closed_book"), run_dir,
                                 closed_book=True, items_file="items_closed_book.jsonl")
                    print(f"[{area} | {model}] closed-book score {e['score']:.2f}", flush=True)
                except ExtractionFailed as ex:
                    print(f"[{area} | {model}] closed-book skipped: {ex}", flush=True)
    (run_dir / "splits.json").write_text(json.dumps(split_info, indent=2))
    report(run_dir, cfg)


def mock_config(cfg):
    cfg = json.loads(json.dumps(cfg))
    for name in list(cfg["models"]):
        cfg["models"][name]["provider"] = "mock"
        pr = cfg["models"][name].setdefault("price_per_mtok", {})
        if pr.get("input") is None:
            pr.update(input=2.0, cached_input=0.2, cache_write=2.5, output=10.0)
    return cfg

# ---------------------------------------------------------------------------
# Report: plots + markdown
# ---------------------------------------------------------------------------

def bootstrap_ci(items, n_boot=2000, seed=0):
    """Paper-level bootstrap CI for accuracy."""
    by_p = {}
    for it in items:
        by_p.setdefault(it["paper_id"], []).append(it["outcome"] == "correct")
    ps = list(by_p.values())
    if not ps:
        return (None, None)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = [x for p in (rng.choice(ps) for _ in ps) for x in p]
        stats.append(sum(s) / len(s))
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def paired_diff_ci(items_a, items_b, n_boot=2000, seed=0):
    """Paired paper-level bootstrap of accuracy(b) - accuracy(a) on the same items."""
    A = {(i["paper_id"], i["qid"]): i["outcome"] == "correct" for i in items_a}
    B = {(i["paper_id"], i["qid"]): i["outcome"] == "correct" for i in items_b}
    keys = sorted(set(A) & set(B))
    by_p = {}
    for k in keys:
        by_p.setdefault(k[0], []).append(B[k] - A[k])
    ps = list(by_p.values())
    if not ps:
        return (None, None, None)
    point = sum(x for p in ps for x in p) / len(keys)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = [x for p in (rng.choice(ps) for _ in ps) for x in p]
        stats.append(sum(s) / len(s))
    stats.sort()
    return point, stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def report(run_dir: Path, cfg: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = cfg or json.loads((run_dir / "config.used.json").read_text())
    rows = list(csv.DictReader((run_dir / "iterations.csv").open()))
    items = [json.loads(l) for l in (run_dir / "items.jsonl").open()]
    if not rows:
        print("no rows"); return
    if (run_dir / "failed_loops.txt").exists():
        print("failed loops:\n" + (run_dir / "failed_loops.txt").read_text())
    use_cost = all(r["cum_cost"] not in ("", None) for r in rows if r["split"] == "train")
    xkey, xlab = ("cum_cost", "cumulative cost (USD)") if use_cost else ("cum_tokens", "cumulative tokens")
    areas = sorted({r["area"] for r in rows})
    models = sorted({r["model"] for r in rows})
    colors = {m: c for m, c in zip(models, plt.rcParams["axes.prop_cycle"].by_key()["color"])}
    fig, axes = plt.subplots(len(areas), 2, figsize=(12, 3.6 * len(areas)), squeeze=False)
    md = [f"# LitPro prompt-evolution run `{run_dir.name}`\n"]
    for ai, area in enumerate(areas):
        md.append(f"\n## {area}\n\n| model | best iter | partition | train seed → best | "
                  f"test seed [95% CI] | test best [95% CI] | Δ test (paired) | tokens | cost | runtime (s) | errors |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|")
        for m in models:
            tr = [r for r in rows if r["area"] == area and r["model"] == m and r["split"] == "train"]
            if not tr:
                continue
            tr.sort(key=lambda r: int(r["iter"]))
            xs = [float(r[xkey]) for r in tr]
            ts = [float(r["cum_runtime_s"]) for r in tr]
            ys = [float(r["score"]) for r in tr]
            best_so_far = [max(ys[: i + 1]) for i in range(len(ys))]
            for ax, X in ((axes[ai][0], xs), (axes[ai][1], ts)):
                ax.plot(X, ys, "o", color=colors[m], alpha=0.35, ms=4)
                ax.step(X, best_so_far, where="post", color=colors[m], label=f"{m} (train, best so far)")
            # test points: seed at x of iter 0, best at final x
            ti = {w: [i for i in items if i["area"] == area and i["extractor"] == m and i["split"] == "test"
                      and i.get("which") == w] for w in ("seed", "best")}
            loop = json.loads((run_dir / "loops" /
                               (re.sub(r"[^A-Za-z0-9_.-]", "_", f"{area}__{m}") + ".json")).read_text())
            test_cells = {}
            for w, xi in (("seed", 0), ("best", -1)):
                its = ti[w] or (ti["seed"] if w == "best" else [])
                if not its:
                    continue
                acc = sum(i["outcome"] == "correct" for i in its) / len(its)
                lo, hi = bootstrap_ci(its)
                test_cells[w] = f"{acc:.2f} [{lo:.2f}, {hi:.2f}]"
                for ax, X in ((axes[ai][0], xs), (axes[ai][1], ts)):
                    ax.errorbar(X[xi], acc, yerr=[[acc - lo], [hi - acc]], fmt="D" if w == "best" else "s",
                                color=colors[m], mfc="white", capsize=3,
                                label=f"{m} test {w}" if ai == 0 else None)
            if not ti["seed"]:
                d = (None, None, None)
            else:
                d = paired_diff_ci(ti["seed"], ti["best"]) if ti["best"] else (0.0, 0.0, 0.0)
            dcell = "—" if d[0] is None else f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"
            cost = f"${loop['cum_cost']:.2f}" if loop.get("cum_cost") is not None else "n/a"
            md.append(f"| {m} | {loop['best_iter']} | {loop['best_partition']} | {ys[0]:.2f} → {max(ys):.2f} | "
                      f"{test_cells.get('seed', '—')} | {test_cells.get('best', '—')} | {dcell} | "
                      f"{loop['cum_tokens']:,} | {cost} | {loop['cum_runtime_s']:.0f} | "
                      f"{sum(int(r['errors']) for r in rows if r['area'] == area and r['model'] == m)} |")
        for m in models:
            p = run_dir / "loops" / (re.sub(r"[^A-Za-z0-9_.-]", "_", f"{area}__{m}") + ".json")
            if p.exists():
                lp = json.loads(p.read_text())
                md.append(f"\n<details><summary>{m}: best instructions (iter {lp['best_iter']}, "
                          f"{lp['best_partition']}; stop: {lp['stop_reason']})</summary>\n\n```\n"
                          f"{lp['best_instructions']}\n```\n</details>")
        axes[ai][0].set(title=f"{area}: score vs {'cost' if use_cost else 'tokens'}", xlabel=xlab, ylabel="score")
        axes[ai][1].set(title=f"{area}: score vs runtime", xlabel="cumulative API time (s)", ylabel="score")
        for ax in axes[ai]:
            ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(6, len(l)), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06 if len(areas) == 1 else 0.03, 1, 1))
    fig.savefig(run_dir / "score_vs_cost_runtime.png", dpi=150)

    # accuracy by question type (seed prompt; train and test)
    qtypes = sorted({i["qtype"] for i in items})
    md.append("\n## Accuracy by question type (seed prompt)\n\n| area | model | split | "
              + " | ".join(qtypes) + " |\n|" + "---|" * (3 + len(qtypes)))
    for area in areas:
        for m in models:
            for split in ("train", "test"):
                sel = [i for i in items if i["area"] == area and i["extractor"] == m and i["split"] == split
                       and int(i["iter"]) == 0]
                if not sel:
                    continue
                cells = []
                for t in qtypes:
                    st = [i for i in sel if i["qtype"] == t]
                    cells.append(f"{sum(i['outcome'] == 'correct' for i in st)}/{len(st)}" if st else "—")
                md.append(f"| {area} | {m} | {split} | " + " | ".join(cells) + " |")

    # closed-book control
    cb_path = run_dir / "items_closed_book.jsonl"
    if cb_path.exists():
        cb = [json.loads(l) for l in cb_path.open()]
        ans = [i for i in cb if i["answer_type"] != "not_reported"]
        md.append("\n## Closed-book control (no PDF, answerable questions only)\n\n| area | model | correct without the paper |\n|---|---|---|")
        for area in areas:
            for m in sorted({i["extractor"] for i in ans}):
                sel = [i for i in ans if i["area"] == area and i["extractor"] == m]
                if sel:
                    md.append(f"| {area} | {m} | {sum(i['outcome'] == 'correct' for i in sel)}/{len(sel)} |")
        by_q = {}
        for i in ans:
            by_q.setdefault(i["qid"], []).append(i["outcome"] == "correct")
        mem = sorted(q for q, v in by_q.items() if sum(v) >= 2)
        md.append(f"\nAnswered correctly without the paper by 2+ models ({len(mem)}): "
                  + (", ".join(f"`{q}`" for q in mem) if mem else "none")
                  + ". These test recall rather than extraction; consider replacing them.")

    # judge disagreement flags
    flags = [i for i in items if i.get("flag")]
    with (run_dir / "judge_flags.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["area", "extractor", "iter", "split", "qid", "answer", "unit", "verdicts", "reasons"])
        for i in flags:
            w.writerow([i["area"], i["extractor"], i["iter"], i["split"], i["qid"], i.get("answer"), i.get("unit"),
                        json.dumps(i.get("verdicts")), json.dumps(i.get("reasons"), ensure_ascii=False)])
    by_q = {}
    for i in flags:
        by_q[i["qid"]] = by_q.get(i["qid"], 0) + 1
    judged = sum(i["method"] == "judges" for i in items)
    missing = sum(i.get("judges_missing", 0) for i in items)
    md.append(f"\n## Judge disagreement\n\n{len(flags)} of {judged} judge-graded items were not unanimous "
              f"(full list in `judge_flags.csv`). Questions flagged most often — check their reference "
              f"answers first:\n"
              + (f"\n**{missing} judge verdicts are missing** (a judge call failed); those items were decided by "
                 f"the remaining judges, and a 1–1 split counts as wrong.\n" if missing else ""))
    for q, c in sorted(by_q.items(), key=lambda t: -t[1])[:15]:
        md.append(f"- `{q}`: {c}")
    # judge-vs-judge pairwise agreement
    judges = cfg.get("judges", [])
    jitems = [i for i in items if i["method"] == "judges"]
    if len(judges) > 1 and jitems:
        md.append("\nPairwise judge agreement on judge-graded items:\n")
        for a in range(len(judges)):
            for b in range(a + 1, len(judges)):
                pairs = [(i["verdicts"].get(judges[a]), i["verdicts"].get(judges[b])) for i in jitems]
                pairs = [p for p in pairs if p[0] in (0, 1) and p[1] in (0, 1)]
                if pairs:
                    md.append(f"- {judges[a]} vs {judges[b]}: {sum(x == y for x, y in pairs) / len(pairs):.2f} "
                              f"(n={len(pairs)})")
    # self-preference: does the judge from the extractor's own provider vote differently from the others?
    prov = {k: v.get("provider") for k, v in cfg.get("models", {}).items()}
    sp = {}
    for i in jitems:
        ep = prov.get(i["extractor"])
        for jm, v in i["verdicts"].items():
            if prov.get(jm) != ep or v not in (0, 1):
                continue
            others = [w for k, w in i["verdicts"].items() if k != jm and w in (0, 1)]
            if not others:
                continue
            maj = 1 if sum(others) * 2 > len(others) else 0
            d = sp.setdefault(i["extractor"], [0, 0, 0])
            d[0] += v == 1 and maj == 0; d[1] += v == 0 and maj == 1; d[2] += 1
    if sp:
        md.append("\nSame-provider judge vs the other judges (self-preference check):\n\n"
                  "| extractor | items | own judge more lenient | own judge stricter |\n|---|---|---|---|")
        for m, (a, b, n) in sorted(sp.items()):
            md.append(f"| {m} | {n} | {a} | {b} |")
    (run_dir / "report.md").write_text("\n".join(md) + "\n")
    print(f"wrote {run_dir/'report.md'} and {run_dir/'score_vs_cost_runtime.png'}")

# ---------------------------------------------------------------------------
# Draft questions (for human verification)
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = """You are a QUESTION-WRITER building a HARD benchmark that tests whether a model can extract
information from a scientific paper. Read the attached paper and write exactly 6 questions, one of each type,
whose answers are unambiguous and checkable from the MAIN TEXT, TABLES OR FIGURES of this PDF (no SI).

Types (qtype) and what makes each hard:
- "text": a value or fact stated only in Results or Methods, not in the abstract or conclusions.
- "table": needs BOTH a row and a column condition, in a table with at least 3 rows.
- "figure": the answer appears ONLY in a figure (plot label, inset table, annotated value or legend), never
  in the running text. Do not use scale bars or other non-scientific annotations.
- "calculation": start the question with "Using values reported in the paper, calculate". Needs at least two
  inputs taken from DIFFERENT places in the paper; the inputs must not appear in the question.
- "conflict": the paper states at least three candidate values for similar quantities (different conditions,
  materials, electrolytes or sections); the question gives the condition that selects exactly one.
- "absent": the paper does NOT state the answer, but reports a closely related quantity (another condition,
  material or metric), so abstaining requires careful reading. Never use a property no paper of this kind
  would report.
If the paper has no table or no figure value that qualifies, write a second "conflict" or "calculation"
question instead and tag it with its real type.

Rules: no question answerable from the abstract alone; never name the table, figure or section in the
question; numeric answers are a single number in the stated unit, with a tolerance that accepts reasonable
rounding (rel_tol 0.02-0.05, or abs_tol for small integers); text answers state the required content in
"accept", not exact wording.
Return ONLY JSON:
{"questions": [{"question": "...", "answer_type": "numeric" | "text" | "not_reported",
  "qtype": "text" | "table" | "figure" | "calculation" | "conflict" | "absent",
  "reference": {"value": <number or null>, "unit": "<unit>", "rel_tol": <number or null>, "abs_tol": <number or null>,
                "text": "<reference answer for text questions>", "accept": "<required content>"},
  "evidence": "<verbatim quote supporting the answer, <= 30 words; for absent: the closely related value>",
  "location": "<section / table / figure>"}]}"""


def _norm(s: str) -> str:
    # undo common pypdf glyph mangling in Nature-family PDFs (/C0 = minus, B = ~, þ = +)
    s = s.replace("/C0", "-").replace("þ", "+")
    s = re.sub(r"\bB(?=\d)", "", s)
    return re.sub(r"\s+", "", re.sub(r"[^a-z0-9]", " ", s.lower()))


def cmd_draft(args):
    cfg = load_config(args.config)
    if args.dry_run:
        cfg = mock_config(cfg)
    papers = load_papers(Path(args.repo))
    P = Providers(cfg)
    run_dir = Path(args.out); run_dir.mkdir(parents=True, exist_ok=True)
    L = Ledger(run_dir, cfg)
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None
    sel = [p for p in papers.values() if not args.areas or p["area"] in args.areas.split(",")]
    if args.per_domain:
        sel = [p for p in sel if p["number"] <= args.per_domain]
    drafters = args.model.split(",")
    out_path = Path(args.output)
    # resume: skip papers already drafted in the output file
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["paper_id"])
    todo = [p for p in sorted(sel, key=lambda p: (p["area"], p["number"])) if p["paper_id"] not in done]
    print(f"{len(sel)} papers selected, {len(done & {p['paper_id'] for p in sel})} already drafted, "
          f"{len(todo)} to go ({args.workers} in parallel)", flush=True)
    for d in drafters:                      # let the drafter run as many calls in parallel as we have workers
        P._sems[P.model(d)["provider"]] = threading.Semaphore(args.workers)
        if args.effort and P.model(d)["provider"] == "openai":
            P.cfg["models"][d].setdefault("extra", {})["reasoning"] = {"effort": args.effort}
    lock, n_done, t_start = threading.Lock(), [0], time.time()

    def one(p):
        drafter = drafters[(p["number"] - 1) % len(drafters)]   # round-robin within each area
        r = P.call(drafter, DRAFT_SYSTEM, f"Paper: {p['title']} (doi {p['doi']}). Write the 6 questions.",
                   pdf_path=p["path"], max_tokens=P.model(drafter).get("optimizer_max_tokens", 16000))
        L.record("draft", "draft", drafter, r, paper_id=p["paper_id"])
        j = parse_json(r.text) or {}
        text = ""
        if PdfReader:
            try:
                text = _norm(" ".join((pg.extract_text() or "") for pg in PdfReader(p["path"]).pages))
            except Exception:
                pass
        rows = []
        for k, q in enumerate(j.get("questions", [])[:6], 1):
            ev = q.get("evidence") or ""
            rows.append(dict(qid=f"{p['paper_id']}_q{k}", paper_id=p["paper_id"], question=q.get("question"),
                             answer_type=q.get("answer_type", "text"), qtype=q.get("qtype", "unspecified"),
                             reference=q.get("reference") or {}, evidence=ev, location=q.get("location"),
                             evidence_found_in_pdf_text=(bool(text) and bool(ev) and _norm(ev)[:60] in text),
                             drafted_by=drafter, verified=False))
        with lock:                          # write as we go, so an interrupted run loses nothing
            with out_path.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done[0] += 1
            el = time.time() - t_start
            print(f"  [{n_done[0]}/{len(todo)}] {p['paper_id']}: {len(rows)} questions "
                  f"({el / 60:.1f} min elapsed, ~{el / n_done[0] * (len(todo) - n_done[0]) / 60:.0f} min left)"
                  + ("" if rows else f"  ! {r.error or 'unparseable reply'}"), flush=True)
        return rows

    with ThreadPoolExecutor(args.workers) as ex:
        allrows = [r for rows in ex.map(one, todo) for r in rows]
    print(f"wrote {len(allrows)} new draft questions to {out_path}. Papers with 0 questions are retried "
          f"if you rerun the same command. Verify the questions before a real run.")

# ---------------------------------------------------------------------------

def cmd_transfer(args):
    """Evaluate each loop's best prompt with other extractors (same area) and in other areas (same model),
    on the test papers only. Writes transfer.csv and transfer.md into the run directory."""
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.used.json").read_text())
    ra = json.loads((run_dir / "run_args.json").read_text())
    papers = load_papers(Path(ra["repo"]))
    qs = load_questions(Path(ra["questions"]), papers, allow_unverified=True)
    splits = json.loads((run_dir / "splits.json").read_text())
    loops = [json.loads(p.read_text()) for p in sorted((run_dir / "loops").glob("*.json"))]
    own = {(l["area"], l["model"]): l for l in loops}
    P, L = Providers(cfg), Ledger(run_dir, cfg)
    J = JudgeCache(run_dir / "judge_cache.json")
    modes = args.modes.split(",")
    jobs = []
    for src in loops:
        if src["best_iter"] == 0:
            continue
        prompt = dict(instructions=src["best_instructions"], partition=src["best_partition"])
        for tgt_area, sp in splits.items():
            for tgt_model in cfg["extractors"]:
                same_area, same_model = tgt_area == src["area"], tgt_model == src["model"]
                if same_area and same_model:
                    continue
                if (same_area and "model" in modes) or (same_model and "area" in modes):
                    jobs.append((src, prompt, tgt_area, tgt_model, sp["test"]))
    print(f"{len(jobs)} transfer evaluations")
    rows = []
    for src, prompt, ta, tm, test_ids in jobs:
        aq = [q for q in qs if q["area"] == ta]
        qbp = {}
        for q in aq:
            qbp.setdefault(q["paper_id"], []).append(q)
        try:
            e = evaluate(P, L, J, cfg, tm, prompt, papers, qbp, test_ids, f"transfer:{ta}:{tm}",
                         dict(area=ta, extractor=tm, iter=src["best_iter"], split="transfer",
                              source_area=src["area"], source_model=src["model"]), run_dir,
                         items_file="items_transfer.jsonl")
        except ExtractionFailed as ex:
            print(f"  skip {src['area']}/{src['model']} -> {ta}/{tm}: {ex}"); continue
        base = own.get((ta, tm), {}).get("test_seed") or [None]
        b = sum(base) / len(base) if base[0] is not None else None
        rows.append(dict(source_area=src["area"], source_model=src["model"], target_area=ta, target_model=tm,
                         score=round(e["score"], 4), target_seed_test=None if b is None else round(b, 4),
                         delta_vs_seed=None if b is None else round(e["score"] - b, 4),
                         tokens=e["usage"]["tokens"], cost=e["usage"]["cost"] if e["usage"]["cost_known"] else None))
        print(f"  {src['area']}/{src['model']} -> {ta}/{tm}: {e['score']:.2f} (target seed {b})", flush=True)
    if not rows:
        print("nothing to transfer (no loop improved on its seed prompt)"); return
    with (run_dir / "transfer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["# Prompt transfer (test papers)\n", "Δ = transferred prompt's test score minus the target's own seed "
          "prompt on the same papers.\n", "| source area | source model | target area | target model | score | Δ vs seed |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        d = "—" if r["delta_vs_seed"] is None else f"{r['delta_vs_seed']:+.2f}"
        md.append(f"| {r['source_area']} | {r['source_model']} | {r['target_area']} | {r['target_model']} | "
                  f"{r['score']:.2f} | {d} |")
    (run_dir / "transfer.md").write_text("\n".join(md) + "\n")
    print(f"wrote {run_dir/'transfer.md'}")


def cmd_check(args):
    """One text call and one PDF call per configured model: verifies keys, model IDs and PDF upload."""
    cfg = load_config(args.config)
    if args.dry_run:
        cfg = mock_config(cfg)
    papers = load_papers(Path(args.repo))
    small = min(papers.values(), key=lambda p: Path(p["path"]).stat().st_size)
    P = Providers(cfg)
    P.max_retries = 1
    ok = True
    for name in cfg["models"]:
        for label, pdf, prompt in (("text", None, "Reply with the single word OK."),
                                   ("pdf ", small["path"], "Give the title of the attached paper in at most 12 words.")):
            r = P.call(name, "You are a terse assistant.", prompt, pdf_path=pdf, max_tokens=P.model(name).get("judge_max_tokens", 2000))
            status = "FAIL " + r.error if r.error else ("EMPTY reply (raise max_tokens?)" if not r.text.strip() else "ok   " + r.text.strip()[:70])
            ok &= status.startswith("ok")
            print(f"{name:8s} {label} in={r.in_tokens:>7,} out={r.out_tokens:>6,} {r.latency_s:5.1f}s  {status}", flush=True)
    print("\nall models OK" if ok else "\nfix the FAIL lines before running (key, model_id, or SDK version)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml")
    common.add_argument("--repo", default="LitPro", help="path to a clone of github.com/Ray16/LitPro")
    common.add_argument("--areas", default=None, help="comma list, e.g. FET,PFAS")
    common.add_argument("--dry-run", action="store_true", help="route every model to the mock provider")

    r = sub.add_parser("run", parents=[common])
    r.add_argument("--questions", default="questions.jsonl")
    r.add_argument("--out", default="runs")
    r.add_argument("--name", default=None)
    r.add_argument("--allow-unverified", action="store_true")
    r.add_argument("--extractors", default=None, help="override config, e.g. claude,gpt")
    r.add_argument("--max-iters", type=int, default=None)
    r.add_argument("--max-tokens", type=int, default=None, help="per-loop token budget (used only without max_usd)")
    r.add_argument("--max-usd", type=float, default=None, help="per-loop dollar budget")

    d = sub.add_parser("draft", parents=[common])
    d.add_argument("--model", required=True, help="model key(s) from config; a comma list rotates drafters by paper")
    d.add_argument("--output", default="questions.draft.jsonl")
    d.add_argument("--out", default="runs/draft")
    d.add_argument("--workers", type=int, default=16, help="papers drafted in parallel")
    d.add_argument("--per-domain", type=int, default=None, help="draft only papers 1..N of each domain")
    d.add_argument("--effort", choices=["low", "medium", "high"], default=None,
                   help="override the OpenAI reasoning effort for drafting")

    sub.add_parser("check", parents=[common])

    tr = sub.add_parser("transfer")
    tr.add_argument("run_dir")
    tr.add_argument("--modes", default="model,area", help="model = other extractors, same area; area = same model, other areas")

    rp = sub.add_parser("report")
    rp.add_argument("run_dir")

    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "draft":
        cmd_draft(args)
    elif args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "transfer":
        cmd_transfer(args)
    else:
        report(Path(args.run_dir))


if __name__ == "__main__":
    main()
