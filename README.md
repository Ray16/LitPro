# LitPro: self-improving prompt agents for scientific data extraction

Can low-cost LLM agents improve their own extraction prompts for chemistry and materials-science papers,
using only a score as feedback? LitPro builds the benchmark and the agent loop to find out, and measures
accuracy against cost and runtime per domain.

Rodrigo Ferreira (rpferreira@uchicago.edu), Ray Zhu (rzhu16@uchicago.edu)

## What is in this repo

| Path | Contents |
|---|---|
| `paper/` | the original 40-paper set (4 domains, 10 papers each); see `paper/README.md` |
| `corpus/paper/manifest.json` | the 270-paper corpus (9 domains): DOI, licence, journal, citations, source URL, SHA-256 per PDF |
| `corpus/candidates/`, `corpus/domain_scan.md` | OpenAlex candidate lists and per-domain open-access counts used to select the corpus |
| `corpus/titles_to_review.md`, `corpus/fetch_log.csv` | final title list per domain and the reason each skipped candidate was skipped |
| `questions/questions.clean.jsonl` | all 1,614 drafted questions (6 per paper), well-formed, not yet all verified |
| `questions/questions.<domain>.jsonl` | verified questions, one file per domain (done: electrocatalysis, FET, HEA, MOF, OPV) |
| `questions/review/` | the flagged questions with checker answers and the review decision for each |
| `scripts/` | the pipeline (below) |
| `config.yaml` | models, prices, judges, budget, split and seed prompt |
| `results/` | pilot report, agent prompt trajectories, figures, one-page summary |

PDFs are not stored in git (about 0.8 GB). Rebuild an identical copy with `python scripts/fetch_papers.py --restore`,
which downloads each file and checks its SHA-256 against the manifest.

## Corpus

270 open-access, CC-licensed research papers, 30 per domain (269 have questions; drafting failed for
`electrocatalysis_08`): electrocatalysis, FET, high-entropy alloys (HEA),
MOF, organic photovoltaics (OPV), hydrogels, separation membranes, photocatalysis and thermoelectrics.
Selected from OpenAlex (2018–2025, at least 100 citations, mostly Nature, Nature Communications, Science Advances
and Science), ranked by citations. Each PDF is checked for identity (DOI or title on its first pages).
Off-topic papers were removed by title review; cross-over papers (e.g. COFs in the MOF set) were kept on purpose.

## Questions

Six per paper, one of each type: a value from running text, a table, a figure, a calculation from reported values,
a choice between conflicting values, and an "absent" question the paper does not answer.
Drafted by GPT-5.5; checked by two independent models (Claude Sonnet 5.5, Gemini 3.6 Flash) that answer every
question with the PDF. Questions where a checker disagrees with the reference are reviewed by hand
(5 domains so far: 34 of 894 flagged, 21 kept, 8 fixed, 5 dropped).

Each line of a question file:

```json
{"qid": "FET_27_q5", "paper_id": "FET_27", "question": "...", "answer_type": "numeric | text | not_reported",
 "qtype": "text | table | figure | calculation | conflict | absent",
 "reference": {"value": 2500, "unit": "cycles", "rel_tol": null, "abs_tol": 1, "text": "", "accept": ""},
 "evidence": "...", "location": "...", "verified": true}
```

## The experiment

For each domain and each agent (Claude Haiku 5.5, GPT-5.4 mini, Gemini 3.5 Flash-Lite):

1. Papers are split 10 train / 20 test, stratified so every question type appears in both.
2. The agent starts from a 21-word seed prompt and rewrites it up to 5 times, seeing only its score on the train papers.
   It may change wording, structure, or partition (all questions in one call vs. one call per question).
   Budget: $5 per loop.
3. The best train prompt and the seed are scored on the held-out test papers.

Grading: numeric answers are checked in code against the reference tolerance; text answers go to a panel of frontier
judges (Claude Opus 5.5, GPT-5.5, Gemini 3.1 Pro), majority vote; "absent" questions are correct only if the agent
abstains. Non-unanimous judge votes are flagged.

## Results so far (electrocatalysis pilot, 19 test papers)

| Agent | Test seed → best | Δ (95% CI) | Cost per loop |
|---|---|---|---|
| Haiku 5.5 | 1.00 → 0.97 | −0.03 [−0.07, 0.00] | $0.36 |
| GPT-5.4 mini | 0.97 → 0.96 | −0.01 [−0.04, +0.02] | $2.16 |
| Flash-Lite 3.5 | 0.84 → 0.87 | +0.03 [−0.01, +0.06] | $0.21 |

1. **Near saturation:** a 21-word prompt already gets 84–100% on unseen papers; remaining errors are mostly
   figure-only values, calculations and "absent" questions.
2. **Self-improvement does not yet generalize:** agents explore sensibly (rewording, domain checklists, per-question
   calls) and gain 1–3 train questions, but every test difference includes zero. With score-only feedback and
   10 train papers they overfit to noise.
3. **Cost separates the models:** Haiku is most accurate at $0.36 per loop; GPT-5.4 mini costs 6× more with no gain;
   Flash-Lite is cheapest but least accurate. With prompt caching, per-question calls cost only 1.4× more.

Judges agreed on 99% of items. Full report and every prompt each agent tried: `results/pilot.electrocatalysis/`.

## Setup

```bash
conda create -n litpro-env python=3.11 -y && conda activate litpro-env
pip install anthropic openai google-genai pyyaml matplotlib pypdf
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
python scripts/fetch_papers.py --restore      # rebuild corpus/paper/*.pdf from the manifest
python scripts/litpro_evolve.py check --repo corpus   # one text and one PDF call per model in config.yaml
```

Model IDs and prices in `config.yaml` are as of October 2026; check them on each provider's pricing page.

## Running

```bash
# one domain (about 10–15 min and ~$3 with the cheap agents)
python scripts/litpro_evolve.py run --repo corpus --questions questions/questions.electrocatalysis.jsonl \
  --areas electrocatalysis --name pilot.electrocatalysis

# all verified domains (per-domain files only; questions.clean.jsonl is not verified)
cat questions/questions.{electrocatalysis,FET,HEA,MOF,OPV}.jsonl > questions.all.jsonl
python scripts/litpro_evolve.py run --repo corpus --questions questions.all.jsonl --name full

# rebuild the report and plots, or test whether best prompts transfer across agents and domains
python scripts/litpro_evolve.py report runs/full
python scripts/litpro_evolve.py transfer runs/full
```

The run refuses unverified questions unless `--allow-unverified` is passed.

Each run writes `runs/<name>/`: `report.md`, `score_vs_cost_runtime.png`, `iterations.csv` (score, tokens, cost and
time per evaluation), `prompts/` (every prompt the agent tried, its reasoning and score), `items.jsonl` (every graded
answer), `judge_flags.csv` and `calls.jsonl` (every API call with tokens, latency and cost).

## Building the corpus and questions from scratch

```bash
python scripts/scan_oa_domains.py --min-citations 100 --years 2018-2025 --top 100     # OpenAlex counts + candidates
python scripts/fetch_papers.py --candidates corpus/candidates --title-match \
  --venues "Nature,Nature Communications,Science Advances,Science"                    # 30 verified PDFs per domain
python scripts/litpro_evolve.py draft --repo corpus --model gpt --output questions.draft.jsonl
python scripts/refcheck_review.py clean --drafts questions.draft.jsonl --out questions.clean.jsonl
# reference check, one domain at a time (each checker in its own terminal)
python scripts/litpro_evolve.py run --repo corpus --questions questions.clean.jsonl --allow-unverified \
  --extractors claude --areas thermoelectrics --max-iters 0 --name rc_claude.thermoelectrics
python scripts/litpro_evolve.py run --repo corpus --questions questions.clean.jsonl --allow-unverified \
  --extractors gemini --areas thermoelectrics --max-iters 0 --name rc_gemini.thermoelectrics
python scripts/refcheck_review.py split --run runs/rc_claude.thermoelectrics runs/rc_gemini.thermoelectrics \
  --questions questions.clean.jsonl --areas thermoelectrics \
  --out questions/questions.thermoelectrics.jsonl --review questions/review/to_review.thermoelectrics.jsonl
# edit the review file (fix / keep with "verified": true, or "drop": true), then
python scripts/refcheck_review.py merge --questions questions/questions.thermoelectrics.jsonl \
  --reviewed questions/review/to_review.thermoelectrics.jsonl
```

## Status and next steps

- Verify questions for hydrogels, membranes, photocatalysis and thermoelectrics; run all 9 domains (~$100).
- Give agents per-question-type scores and require a 2-question gain before keeping a rewrite, to curb overfitting.
- Harder tasks (multi-figure, cross-paper); temperature and model-tier sweeps; prompt transfer across models and domains.

## Licences

Papers are redistributable under their own Creative Commons licences, recorded per paper in the
manifest; PDFs are fetched from the publishers or Europe PMC and are not modified.
