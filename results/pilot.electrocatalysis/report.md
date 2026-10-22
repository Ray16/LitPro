# LitPro prompt-evolution run `pilot.electrocatalysis`


## electrocatalysis

| model | best iter | partition | train seed → best | test seed [95% CI] | test best [95% CI] | Δ test (paired) | tokens | cost | runtime (s) | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| gemini-lite | 5 | joint | 0.87 → 0.90 | 0.84 [0.77, 0.90] | 0.87 [0.80, 0.92] | +0.03 [-0.01, +0.06] | 644,965 | $0.21 | 441 | 0 |
| gpt-mini | 2 | joint | 0.93 → 0.98 | 0.97 [0.94, 1.00] | 0.96 [0.92, 1.00] | -0.01 [-0.04, +0.02] | 3,504,355 | $2.16 | 1110 | 0 |
| haiku | 3 | joint | 0.97 → 0.98 | 1.00 [1.00, 1.00] | 0.97 [0.93, 1.00] | -0.03 [-0.07, +0.00] | 4,344,181 | $0.36 | 729 | 0 |

<details><summary>gemini-lite: best instructions (iter 5, joint; stop: max_iters)</summary>

```
Carefully read the attached electrocatalysis paper, including main text, tables, figures, and supporting information. For each question, locate the exact relevant section and extract precise quantitative and qualitative findings. When answering about electrochemical performance (such as overpotential at 10 mA cm-2 or other current densities, Tafel slopes, Faradaic efficiency, ECSA, TOF, and stability), report exact numerical values, units, and testing conditions (electrolyte, pH, reference electrode, IR-correction). If the paper does not contain the answer, explicitly state 'not reported'.
```
</details>

<details><summary>gpt-mini: best instructions (iter 2, joint; stop: max_iters)</summary>

```
You are answering questions about one scientific paper. Use only the attached paper; do not use outside knowledge.

Read the entire paper carefully before answering, including the abstract, introduction, methods, results, discussion, conclusion, tables, figure captions, and any appendix/supplement provided. For each question, find the most direct explicit support in the paper.

Answer with the most specific information stated in the paper. Prefer exact names, numbers, units, chemical formulas, abbreviations, experimental conditions, and comparisons as written. If the paper defines an acronym, you may use the defined form; otherwise use the paper’s wording. If multiple similar values or materials appear, choose only the one that exactly matches the question and the paper context; do not substitute a related catalyst, control, condition, or metric.

When answering quantitative questions, use the exact reported value and unit, and preserve the reported basis/condition (for example, potential vs current density, electrolyte concentration, pH, scan rate, temperature, reference electrode, or loading) if relevant. If the answer is presented in a table, figure, or caption, use that source directly rather than inferring from surrounding prose. If a question asks for multiple items, list all items that are explicitly reported.

Do not infer unstated mechanisms, fill in missing values, or mix up similar materials or metrics. If the paper does not explicitly state the answer, say it is not reported. Keep answers concise and factual; do not add extra explanation.
```
</details>

<details><summary>haiku: best instructions (iter 3, joint; stop: max_iters)</summary>

```
Answer each question using only the attached paper. Do not add outside knowledge.

Work through these steps for each question:
1. Find every relevant passage: main text, tables, figure captions, axis labels, legends, footnotes, and mentions of supporting information.
2. Identify exactly which quantity is requested. Electrocatalysis papers use many related metrics, so do not swap them:
- Overpotential at a stated current density (e.g., at 10 mA cm^-2) is not the same as the onset potential or the potential itself.
- Mass activity (A mg^-1) is not specific activity (mA cm^-2 or A m^-2).
- Tafel slope (mV dec^-1) is not exchange current density.
- Faradaic efficiency (%) is not product yield or selectivity.
- ECSA-normalized activity is not geometric-area activity.
- Turnover frequency is not current density.
3. Report numbers exactly as stated, with units and the conditions the paper attaches to them: electrolyte and pH, reference electrode and scale (RHE, SHE, Ag/AgCl), iR correction or not, catalyst loading, temperature, and time.
4. When several values appear, choose the one for the specific material, composition, potential window, or condition the question names. Do not use a best-case, average, or different-sample value unless the question asks for it.
5. If a value appears only in a figure, use a value stated in the text or caption. Read a plot only if the question asks for an approximate reading.
6. For stability or durability questions, report the stated duration, retention, decay rate, or cycle count under the stated conditions.
7. For yes/no or qualitative questions, answer from any statement in the text or caption, including indirect wording or a described trend.
8. Mark an item as not reported only after completing steps 1 to 7 and finding no supporting information anywhere in the paper.

Be precise and concise. Do not infer values the paper does not give, and do not convert units unless the question asks for it.
```
</details>

## Accuracy by question type (seed prompt)

| area | model | split | absent | calculation | conflict | figure | text |
|---|---|---|---|---|---|---|---|
| electrocatalysis | gemini-lite | train | 7/10 | 16/17 | 13/13 | 6/10 | 10/10 |
| electrocatalysis | gemini-lite | test | 16/19 | 25/31 | 25/26 | 12/19 | 18/19 |
| electrocatalysis | gpt-mini | train | 8/10 | 16/17 | 13/13 | 9/10 | 10/10 |
| electrocatalysis | gpt-mini | test | 17/19 | 31/31 | 26/26 | 18/19 | 19/19 |
| electrocatalysis | haiku | train | 10/10 | 17/17 | 13/13 | 8/10 | 10/10 |
| electrocatalysis | haiku | test | 19/19 | 31/31 | 26/26 | 19/19 | 19/19 |

## Judge disagreement

2 of 170 judge-graded items were not unanimous (full list in `judge_flags.csv`). Questions flagged most often — check their reference answers first:

- `electrocatalysis_23_q4`: 1
- `electrocatalysis_17_q1`: 1

Pairwise judge agreement on judge-graded items:

- opus vs gpt: 0.99 (n=170)
- opus vs gemini-pro: 0.99 (n=170)
- gpt vs gemini-pro: 0.99 (n=170)

Same-provider judge vs the other judges (self-preference check):

| extractor | items | own judge more lenient | own judge stricter |
|---|---|---|---|
| gemini-lite | 60 | 0 | 0 |
| gpt-mini | 56 | 0 | 1 |
| haiku | 54 | 0 | 0 |
