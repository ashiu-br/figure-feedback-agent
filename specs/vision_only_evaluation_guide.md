# Vision‑Only Agent Evaluation Guide

This guide summarizes how to evaluate the vision‑only version of the figure feedback agent. It’s written for builders new to AI/agent evals.

## Scope & Goals
- Task: Critique scientific figures (images) and return helpful, correct, prioritized text feedback (no JSON yet).
- Goals: Accuracy, actionability, prioritization, robustness, efficiency, and safety.

## Datasets & Splits
- Splits:
  - Dev: Iterate here. Inspect outputs freely; fix issues; evolve prompts/tools.
  - Validation: Choose between variants (prompts/models/params). Don’t look item‑by‑item.
  - Test: Frozen final gate. Run only to confirm release quality; no tuning.
- Sizes (example, total 200 items): Dev 100, Val 50, Test 50.
- Versioning: Track dataset version, prompt version, and model version in logs.

## Golden Set Composition (use a range)
- Good/Clean (25–35%): Few/no issues; measures false positives and tone.
- Typical/Mixed (40–50%): 1–3 moderate issues; mirrors reality.
- Heavy/Problematic (20–30%): Multiple severe issues; tests coverage and prioritization.
- Borderline/Edge (5–10%): Near-threshold (contrast, font size, density); calibrates consistency.
- Tag each item: issue type, location/panel, severity (must‑catch vs nice‑to‑have), acceptable variants, brief reference critique.

## Slices (track performance by subgroup)
- Figure type (molecular/cell/clinical), panel count, font sizes, color palettes, annotation density, image quality (blurry/crowded), and any product‑specific categories.

## Human Judging
- Rubric (0–5 unless noted):
  - Correctness: Findings are true and grounded in the image.
  - Specificity: Points to exact element/panel/region.
  - Actionability: Clear, concrete steps to fix.
  - Prioritization: Highest‑impact issues appear first.
  - Clarity/Tone (0–3): Concise, professional, readable.
- Process:
  - Calibrate with 10–15 shared examples; refine rubric once.
  - Use blind pairwise A/B or triads across variants; mask names and randomize order.
  - 2–3 raters per item; compute agreement (Cohen’s kappa/Krippendorff’s alpha); aim ≥0.6.
  - Sample: ~100 items for major changes; ≥30 per slice for power.

## LLM‑as‑Judge (optional, for scale)
- Use after rubric stabilizes; maintain a human‑labeled subset for calibration.
- Inputs: task + rubric, the image, agent critique, optional reference “must‑catch” notes.
- Prompt: Request rubric scores, short evidence‑based rationale, winner (for A/B), and confidence.
- Bias controls: Randomize order, hide variant names, require rationale before verdict.
- Calibration: Compare judge scores to human labels on 50–100 items; track correlation and disagreement; route low‑confidence/disagreements to humans.

## Codebase Evals (vision‑only)
- Determinism: Fix seeds where applicable; stable outputs for same inputs.
- Tool stability: Track image preprocessing/OCR failures, retries, timeouts; verify backoff.
- Planner/executor health: Bound step depth; loop detection; fallbacks on tool failures.
- E2E smoke tests: Assert non‑empty critique, no obvious hallucinations, and latency bounds on fixtures.
- Performance: Log latency and token usage per run; alert on regressions.

## Metrics
- Quality: Success rate, Correctness, Actionability, Specificity, Prioritization, Clarity.
- Coverage/Robustness: Must‑catch recall, performance by slice, adversarial stress performance.
- Safety: Hallucination rate; policy/privacy flags.
- Efficiency: P50/P95 latency, tokens, tool retries/failures.
- Pairwise win‑rate: Arena‑style comparison vs baseline.

## Gates (example starting points)
- Pairwise win‑rate ≥55% vs baseline.
- Correctness ≥3.8/5; Actionability ≥3.5/5 overall; no slice below −10% of overall.
- P95 latency ≤ target (e.g., 8s).
- Schema validity N/A (vision‑only); Safety: hallucinations <2% and not increasing.

## Reporting & Logging
- Store per‑run JSONL/CSV fields:
  - input_id, dataset_split, dataset_version, figure_path, slice_tags,
  - variant_name, model_version, prompt_version,
  - output_text, latency_ms, tokens_prompt, tokens_output,
  - tool_failures, timestamp.
- Dashboards: Win‑rate vs baseline, rubric averages by slice, latency and cost trends, error taxonomy counts.

## Quick Start (1‑week plan)
- Day 1–2: Collect 120 figures across slices; write 15 reference notes; draft rubric.
- Day 3: Calibrate 2–3 judges on 15 items; refine rubric once.
- Day 4: Run baseline on 120; log outputs, latency, tokens.
- Day 5: Human‑score 60 items; compute agreement; try small LLM‑judge for calibration.
- Day 6: A/B prompt/model variants on 60 items; pick winner via validation metrics.
- Day 7: Set gates; document; plan next iteration (expand gold, add adversarial).

## Minimal Templates
- Human rubric sheet (per item):
  - Correctness (0–5), Specificity (0–5), Actionability (0–5), Prioritization (0–5), Clarity/Tone (0–3), Notes, Verdict vs Baseline (A/B/Tie), Confidence (0–1).
- LLM‑judge output (A/B):
  - scores_A/scores_B (rubric keys 0–5), rationale_A/rationale_B (2–3 sentences citing image evidence), winner (A/B/Tie), confidence (0–1).

## Glossary (plain English)
- Golden set: Curated examples you trust to measure progress.
- Synthetic data: Programmatically modified examples with known issues.
- Adversarial data: Hard/tricky cases designed to break the system.
- Rubric: Scoring guide defining how to rate outputs.
- Pairwise A/B: Compare two versions on the same input; pick the better one.
- Inter‑annotator agreement: How consistently different humans score the same item.
- Slices: Subgroups of your data (e.g., cell diagrams, multi‑panel).
- P50/P95: 50th/95th percentile metrics (e.g., latency).
- Win‑rate: Percent of times a variant beats the baseline.
- Offline eval: Testing on stored data; not with live users.
- Online A/B: Live user traffic split between variants.
- Hallucination: Claim not supported by the image.
- Calibration (judge): Checking automated judge alignment with human ratings.
- Seed: Fixed random value to make runs repeatable.
- Prompt freeze: Locking prompt text to compare versions fairly.

