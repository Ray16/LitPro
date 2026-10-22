#!/usr/bin/env python3
"""
fetch_papers.py — build the corpus: N verified open-access PDFs per domain, plus a LitPro-style manifest.

Reads candidates/<domain>.csv (written by scan_oa_domains.py), walks each list from the most cited paper
down, and keeps a paper only if
  - its licence is Creative Commons,
  - its title does not look like a review / perspective / correction,
  - a PDF downloads (publisher link, then Europe PMC, then Unpaywall),
  - the file is a real PDF with 4-40 pages, and its DOI or title appears on the first two pages,
  - its DOI is not already used by another domain.
It stops at --per-domain papers per domain.

Output (default corpus/):
  corpus/paper/<field>/NN_<title>.pdf     the PDFs (keep them out of git)
  corpus/paper/manifest.json              same format as LitPro, so:  litpro_evolve.py run --repo corpus
  corpus/titles_to_review.md              skim these titles for off-topic papers before writing questions
  corpus/fetch_log.csv                    every attempt and why it was skipped

Re-running is safe: papers already in the manifest are kept and not downloaded again.

  python fetch_papers.py                                   # the 10 domains, 30 papers each
  python fetch_papers.py --domains photocatalysis,hydrogels --per-domain 30
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, os, re, sys, time, urllib.parse, urllib.request
from datetime import date
from pathlib import Path

# scan_oa_domains.py domain name -> field name used in paper ids (e.g. photocatalysis_07)
DOMAINS = {
    "MOF": "MOF",
    "FET": "FET",
    "electrocatalysis": "electrocatalysis",
    "organic photovoltaics": "OPV",
    "photocatalysis": "photocatalysis",
    "thermoelectrics": "thermoelectrics",
    "polymer membranes": "membranes",
    "high-entropy alloys": "HEA",
    "zeolites": "zeolites",
    "hydrogels": "hydrogels",
}
BACKUP_DOMAINS = {"perovskite solar cells": "PSC", "batteries": "batteries"}   # use with --domains
# --title-match: a refill candidate must match these in its TITLE (keeps refills on-topic)
TITLE_PATTERNS = {
    "MOF": r"metal.organic framework|\bMOFs?\b|\bZIF",
    "OPV": r"organic solar cell|organic photovoltaic|non-fullerene|fullerene-free",
    "membranes": r"membrane.*(separation|siev|filtration|desalination|permea|selectiv)|"
                 r"(separation|siev|filtration|desalination|permea|selectiv).*membrane|nanofiltration|reverse osmosis",
    "FET": r"transistor", "electrocatalysis": r"electrocatal|electroreduction|evolution reaction|reduction reaction",
    "photocatalysis": r"photocatal|photoreduction", "thermoelectrics": r"thermoelectric",
    "HEA": r"entropy alloy|multi-principal", "hydrogels": r"hydrogel", "zeolites": r"zeolite",
}
CC = ("cc-by", "cc-by-sa", "cc-by-nd", "cc-by-nc", "cc-by-nc-sa", "cc-by-nc-nd", "cc0", "public-domain")
NOT_RESEARCH = re.compile(r"\b(review|perspective|progress in|roadmap|tutorial|correction|erratum|retraction|"
                          r"editorial|comment on|reply to|fundamentals and applications|recent advances|"
                          r"challenges and opportunities|design strategies for)\b", re.I)
EMAIL = os.environ.get("OPENALEX_EMAIL", "")
UA = f"Mozilla/5.0 (compatible; litpro-corpus/1.0; mailto:{EMAIL or 'unknown'})"


def http_get(url: str, timeout=60) -> tuple[bytes | None, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def europepmc_pdf_url(doi: str) -> str | None:
    q = urllib.parse.quote(f'DOI:"{doi}"')
    data, _ = http_get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&resultType=core")
    try:
        res = json.loads(data)["resultList"]["result"]
    except Exception:
        return None
    for r in res:
        for u in (r.get("fullTextUrlList") or {}).get("fullTextUrl", []):
            if u.get("documentStyle") == "pdf" and u.get("availabilityCode") in ("OA", "F"):
                return u.get("url")
        if r.get("pmcid"):
            return f"https://europepmc.org/articles/{r['pmcid']}?pdf=render"
    return None


def unpaywall_pdf_url(doi: str) -> str | None:
    if not EMAIL:
        return None
    data, _ = http_get(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}")
    try:
        loc = json.loads(data).get("best_oa_location") or {}
        return loc.get("url_for_pdf")
    except Exception:
        return None


def check_pdf(blob: bytes, doi: str, title: str) -> tuple[bool, str, int]:
    if not blob or not blob.lstrip()[:5].startswith(b"%PDF"):
        return False, "not a PDF (probably an HTML landing or bot-check page)", 0
    try:
        from pypdf import PdfReader
        rd = PdfReader(io.BytesIO(blob))
        pages = len(rd.pages)
        head = " ".join((rd.pages[i].extract_text() or "") for i in range(min(2, pages))).lower()
    except Exception as e:
        return False, f"unreadable PDF ({type(e).__name__})", 0
    if not 4 <= pages <= 40:
        return False, f"{pages} pages (expected 4-40)", pages
    words = [w for w in re.findall(r"[a-z]{4,}", title.lower())]
    overlap = sum(w in head for w in words) / max(1, len(words))
    if doi.lower() not in head.replace(" ", "") and overlap < 0.6:
        return False, f"DOI not on first pages and title overlap {overlap:.2f}", pages
    return True, "", pages


def slug(s: str, n=90) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^A-Za-z0-9]+", "-", s)).strip("-")[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", default=",".join(DOMAINS), help="comma list of scan domain names")
    ap.add_argument("--per-domain", type=int, default=30)
    ap.add_argument("--candidates", default="candidates")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between downloads (be polite)")
    ap.add_argument("--drop", default=None,
                    help='remove papers before refilling, e.g. "MOF:1,7,9;OPV:3,4" (field:numbers; fields separated by ;)')
    ap.add_argument("--drop-field", default=None, help="remove whole fields, e.g. zeolites (comma list)")
    ap.add_argument("--title-match", action="store_true",
                    help="refill only with candidates whose title matches TITLE_PATTERNS for the field")
    ap.add_argument("--restore", action="store_true",
                    help="download only the PDFs listed in an existing manifest that are missing locally, and check "
                         "their SHA-256 (use this after cloning the repo)")
    ap.add_argument("--venues", default=None,
                    help='comma list of journal names to keep, e.g. "Nature,Nature Communications,Science Advances,Science"')
    args = ap.parse_args()

    venues = {v.strip() for v in args.venues.split(",")} if args.venues else None
    root = Path(args.out) / "paper"
    root.mkdir(parents=True, exist_ok=True)
    man_path = root / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {
        "retrieved": str(date.today()), "download_tool": "fetch_papers.py",
        "selection_criteria": ["OpenAlex search (scan_oa_domains.py), journal articles, open access, CC licence",
                               "ranked by citations within high-impact venues", "PDF identity checked (DOI or title on first pages)",
                               "reviews/perspectives/corrections excluded by title", "one domain per DOI"],
        "papers": []}
    manifest.setdefault("dropped", [])

    if args.restore:
        ok = bad = 0
        for p in manifest["papers"]:
            f = p["files"][0]
            fp = root / f["file"]
            if fp.exists() and hashlib.sha256(fp.read_bytes()).hexdigest() == f["sha256"]:
                ok += 1; continue
            urls = [f.get("source_url"), europepmc_pdf_url(p["doi"]), unpaywall_pdf_url(p["doi"])]
            for url in [u for u in urls if u]:
                blob, err = http_get(url)
                time.sleep(args.sleep)
                if blob and hashlib.sha256(blob).hexdigest() == f["sha256"]:
                    fp.parent.mkdir(parents=True, exist_ok=True); fp.write_bytes(blob); ok += 1
                    print(f"  restored {p['field']}_{p['number']:02d}", flush=True)
                    break
            else:
                bad += 1
                print(f"  ! {p['field']}_{p['number']:02d}: could not restore an identical copy ({p['doi']})", flush=True)
        print(f"{ok} PDFs present and verified, {bad} missing")
        return

    def remove(entry, why):
        for f in entry.get("files", []):
            fp = root / f["file"]
            if fp.exists():
                fp.unlink()
        manifest["papers"].remove(entry)
        manifest["dropped"].append(dict(field=entry["field"], number=entry["number"], doi=entry["doi"],
                                        title=entry["title"], reason=why))
        print(f"  dropped {entry['field']}_{entry['number']:02d}: {entry['title'][:70]}")

    if args.drop_field:
        for fld in [x.strip() for x in args.drop_field.split(",")]:
            for e in [p for p in manifest["papers"] if p["field"] == fld]:
                remove(e, f"field {fld} dropped")
            d = root / fld
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
    if args.drop:
        for part in args.drop.split(";"):
            fld, nums = part.split(":")
            nums = {int(n) for n in nums.split(",") if n.strip()}
            for e in [p for p in manifest["papers"] if p["field"] == fld.strip() and p["number"] in nums]:
                remove(e, "off-topic (manual review)")
    man_path.write_text(json.dumps(manifest, indent=1))
    used = {p["doi"].lower() for p in manifest["papers"]} | {d["doi"].lower() for d in manifest["dropped"]}
    log_path = Path(args.out) / "fetch_log.csv"
    new_log = not log_path.exists()
    logf = log_path.open("a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["domain", "doi", "title", "status", "detail"])

    for dom in [d.strip() for d in args.domains.split(",")]:
        known = {**DOMAINS, **BACKUP_DOMAINS}
        if dom not in known:
            sys.exit(f"unknown domain {dom!r}; known: {', '.join(known)}")
        field = known[dom]
        csv_path = Path(args.candidates) / (dom.replace("/", "_").replace(" ", "_") + ".csv")
        if not csv_path.exists():
            sys.exit(f"missing {csv_path}: run scan_oa_domains.py first")
        rows = sorted(csv.DictReader(csv_path.open()), key=lambda r: -int(r.get("citations") or 0))
        have = [p for p in manifest["papers"] if p["field"] == field]
        print(f"\n{dom} ({field}): {len(have)} already in manifest, {len(rows)} candidates", flush=True)
        for r in rows:
            if len(have) >= args.per_domain:
                break
            doi = (r.get("doi") or "").replace("https://doi.org/", "").strip()
            title = (r.get("title") or "").strip()
            def skip(why):
                log.writerow([dom, doi, title, "skipped", why]); logf.flush()
            if not doi:
                skip("no DOI"); continue
            if doi.lower() in used:
                skip("already used (this or another domain)"); continue
            if (r.get("license") or "") not in CC:
                skip(f"licence {r.get('license') or 'unknown'}"); continue
            if venues and (r.get("venue") or "") not in venues:
                skip(f"venue {r.get('venue')} not in --venues"); continue
            if args.title_match and field in TITLE_PATTERNS and not re.search(TITLE_PATTERNS[field], title, re.I):
                skip("title does not match the field (--title-match)"); continue
            if NOT_RESEARCH.search(title):
                skip("title looks like a review/perspective/correction"); continue
            got, why_all, src = None, [], None
            for name, url in (("publisher", r.get("pdf_url")), ("europepmc", None), ("unpaywall", None)):
                if name == "europepmc":
                    url = europepmc_pdf_url(doi)
                elif name == "unpaywall":
                    url = unpaywall_pdf_url(doi)
                if not url:
                    why_all.append(f"{name}: no URL"); continue
                blob, err = http_get(url)
                time.sleep(args.sleep)
                if err:
                    why_all.append(f"{name}: {err}"); continue
                ok, why, pages = check_pdf(blob, doi, title)
                if ok:
                    got, src = (blob, pages), url
                    break
                why_all.append(f"{name}: {why}")
            if not got:
                skip(" | ".join(why_all)); continue
            blob, pages = got
            taken = {p["number"] for p in have}
            num = next(n for n in range(1, 10_000) if n not in taken)   # reuse numbers freed by --drop
            rel = f"{field}/{num:02d}_{slug(title)}.pdf"
            (root / field).mkdir(exist_ok=True)
            (root / rel).write_bytes(blob)
            entry = dict(field=field, number=num, title=title, doi=doi, year=int(r.get("year") or 0),
                         journal=r.get("venue"), citations_openalex=int(r.get("citations") or 0),
                         license=r.get("license"),
                         files=[dict(file=rel, format="pdf", bytes=len(blob),
                                     sha256=hashlib.sha256(blob).hexdigest(), source_url=src)],
                         verification=dict(pdf_pages=pages, checked_date=str(date.today())))
            manifest["papers"].append(entry); have.append(entry); used.add(doi.lower())
            man_path.write_text(json.dumps(manifest, indent=1))
            log.writerow([dom, doi, title, "kept", f"{pages} pages via {src}"]); logf.flush()
            print(f"  [{len(have):2d}/{args.per_domain}] {title[:80]}", flush=True)
        if len(have) < args.per_domain:
            print(f"  ! only {len(have)}/{args.per_domain}: rerun scan_oa_domains.py with a larger --top "
                  f"or lower --min-citations, then rerun this script", flush=True)

    logf.close()
    md = ["# Titles to review\n", "Skim for off-topic papers. To drop one, delete its entry from manifest.json "
          "and its PDF, then rerun fetch_papers.py to refill the slot.\n"]
    for field in sorted({p["field"] for p in manifest["papers"]}):
        ps = sorted((p for p in manifest["papers"] if p["field"] == field), key=lambda p: p["number"])
        md.append(f"\n## {field} ({len(ps)})\n")
        md += [f"{p['number']:02d}. {p['title']} — {p['journal']}, {p['year']} ({p['citations_openalex']} cit.)" for p in ps]
    (Path(args.out) / "titles_to_review.md").write_text("\n".join(md) + "\n")
    total = sum(f["bytes"] for p in manifest["papers"] for f in p["files"])
    print(f"\n{len(manifest['papers'])} papers, {total / 1e9:.2f} GB in {root}. "
          f"Review {Path(args.out) / 'titles_to_review.md'}; failures in {log_path}.")


if __name__ == "__main__":
    main()
