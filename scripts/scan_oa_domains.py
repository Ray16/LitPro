#!/usr/bin/env python3
"""
scan_oa_domains.py — count open-access, CC-licensed, highly cited papers per chem/matsci domain (OpenAlex).

For each domain query it reports
  - n_all:      OA journal articles in the year range with >= min citations (any venue)
  - n_venues:   the same, restricted to high-impact venues (LitPro-style: Nature, Science, Nat Commun, Sci Adv,
                plus other selective OA chemistry/materials journals; edit VENUES below)
  - top venues for the domain
  - for the top N papers by citations: share with a CC licence and share with a direct PDF link
and writes candidates/<domain>.csv (top papers) so you can pick the 10 per domain.

No API key is needed for small scans; set OPENALEX_EMAIL (polite pool) and, if your account has one,
OPENALEX_API_KEY.

  python scan_oa_domains.py                          # all domains below, 2015-2025, >= 200 citations
  python scan_oa_domains.py --min-citations 100 --years 2018-2025 --top 50
  python scan_oa_domains.py --domains "perovskite solar cells,lithium batteries"
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time, urllib.parse, urllib.request
from pathlib import Path

API = "https://api.openalex.org/works"

# domain -> OpenAlex title/abstract search (| = OR). The first four are the current LitPro areas.
DOMAINS = {
    "MOF": "metal-organic framework|metal organic framework",
    "FET": "field-effect transistor|field effect transistor",
    "electrocatalysis": "electrocatalyst|electrocatalysis|electrocatalytic",
    "PFAS": "perfluoroalkyl|polyfluoroalkyl|PFAS",
    "lithium/sodium batteries": "lithium-ion battery|lithium metal battery|sodium-ion battery|solid-state electrolyte",
    "perovskite solar cells": "perovskite solar cell|perovskite photovoltaic",
    "organic photovoltaics": "organic solar cell|non-fullerene acceptor",
    "photocatalysis": "photocatalytic hydrogen|photocatalytic CO2|photocatalyst",
    "thermoelectrics": "thermoelectric",
    "2D materials": "two-dimensional material|MXene|transition metal dichalcogenide",
    "polymer membranes": "membrane separation|gas separation membrane|polymer membrane",
    "high-entropy alloys": "high-entropy alloy|high entropy alloy",
    "zeolites/porous carbons": "zeolite|porous carbon",
    "hydrogels": "hydrogel",
}

# High-impact venues by ISSN (print|online). Edit freely.
VENUES = {
    "Nature": "0028-0836|1476-4687",
    "Science": "0036-8075|1095-9203",
    "Nature Communications": "2041-1723",
    "Science Advances": "2375-2548",
    "Chemical Science": "2041-6520|2041-6539",
    "ACS Central Science": "2374-7943|2374-7951",
    "JACS Au": "2691-3704",
    "Advanced Science": "2198-3844",
    "npj Computational Materials": "2057-3960",
    "Communications Chemistry": "2399-3669",
    "Communications Materials": "2662-4443",
}
CC = ("cc-by", "cc-by-sa", "cc-by-nd", "cc-by-nc", "cc-by-nc-sa", "cc-by-nc-nd", "cc0", "public-domain")


def get(params: dict) -> dict:
    params = dict(params)
    if os.environ.get("OPENALEX_EMAIL"):
        params["mailto"] = os.environ["OPENALEX_EMAIL"]
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    url = API + "?" + urllib.parse.urlencode(params, safe=":|,<>")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt); continue
            sys.exit(f"OpenAlex HTTP {e.code}: {body}\nURL: {url}")
        except urllib.error.URLError as e:
            if attempt < 3:
                time.sleep(2 ** attempt); continue
            sys.exit(f"network error: {e}")


def base_filter(query: str, years: str, min_cit: int) -> list[str]:
    return [f"title_and_abstract.search:{query}", f"publication_year:{years}", "type:article",
            "is_oa:true", f"cited_by_count:>{min_cit - 1}"]


def count(filters: list[str]) -> int:
    return get({"filter": ",".join(filters), "per-page": 1})["meta"]["count"]


def scan(name: str, query: str, args, venue_issns: str) -> dict:
    f = base_filter(query, args.years, args.min_citations)
    n_all = count(f)
    n_ven = count(f + [f"primary_location.source.issn:{venue_issns}"])
    groups = get({"filter": ",".join(f), "group_by": "primary_location.source.id"}).get("group_by", [])
    top_venues = ", ".join(f"{g['key_display_name']} ({g['count']})" for g in groups[:4])
    # top papers (restricted to venues unless --any-venue)
    ff = f if args.any_venue else f + [f"primary_location.source.issn:{venue_issns}"]
    res = get({"filter": ",".join(ff), "sort": "cited_by_count:desc", "per-page": min(args.top, 200),
               "select": "id,doi,title,publication_year,cited_by_count,primary_location,best_oa_location"})
    rows = []
    for w in res.get("results", []):
        loc = w.get("best_oa_location") or {}
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        rows.append(dict(title=w.get("title"), doi=w.get("doi"), year=w.get("publication_year"),
                         citations=w.get("cited_by_count"), venue=src, license=loc.get("license"),
                         pdf_url=loc.get("pdf_url")))
    k = len(rows) or 1
    cc = sum((r["license"] or "") in CC for r in rows) / k
    pdf = sum(bool(r["pdf_url"]) for r in rows) / k
    ok = sum(((r["license"] or "") in CC) and bool(r["pdf_url"]) for r in rows)
    out = Path(args.out); out.mkdir(exist_ok=True)
    with (out / (name.replace("/", "_").replace(" ", "_") + ".csv")).open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["title"]); wr.writeheader(); wr.writerows(rows)
    return dict(domain=name, n_all=n_all, n_venues=n_ven, top_k=len(rows), cc_share=cc, pdf_share=pdf,
                usable=ok, top_venues=top_venues)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2015-2025")
    ap.add_argument("--min-citations", type=int, default=200)
    ap.add_argument("--top", type=int, default=50, help="top papers per domain to inspect and save")
    ap.add_argument("--domains", default=None, help="comma list of names from DOMAINS (default: all)")
    ap.add_argument("--any-venue", action="store_true", help="candidate lists from any venue, not just VENUES")
    ap.add_argument("--out", default="candidates")
    args = ap.parse_args()
    names = [d.strip() for d in args.domains.split(",")] if args.domains else list(DOMAINS)
    issns = "|".join(VENUES.values())
    results = []
    for n in names:
        if n not in DOMAINS:
            sys.exit(f"unknown domain {n!r}; add it to DOMAINS")
        r = scan(n, DOMAINS[n], args, issns)
        results.append(r)
        print(f"{n:28s} all={r['n_all']:>6,}  high-impact venues={r['n_venues']:>5,}  "
              f"top{r['top_k']}: CC {r['cc_share']:.0%}, PDF {r['pdf_share']:.0%}, usable {r['usable']}", flush=True)
    results.sort(key=lambda r: -r["n_venues"])
    md = [f"# Open-access, highly cited papers per domain\n",
          f"OpenAlex, {args.years}, journal articles, open access, >= {args.min_citations} citations. "
          f"'Usable' = CC licence and a direct PDF link, among the top {args.top} papers "
          f"({'any venue' if args.any_venue else 'high-impact venues'}).\n",
          "| domain | all venues | high-impact venues | usable in top list | top venues |", "|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['domain']} | {r['n_all']:,} | {r['n_venues']:,} | {r['usable']}/{r['top_k']} | {r['top_venues']} |")
    Path("domain_scan.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote domain_scan.md and {args.out}/<domain>.csv")


if __name__ == "__main__":
    main()
