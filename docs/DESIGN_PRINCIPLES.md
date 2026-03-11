# Design Principles: Research Literature → scb-datagen Implementation

This document maps eight research-backed design principles for automated dataset generation to the specific components, modules, and design decisions in the scb-datagen pipeline. Each section documents what is implemented, what is partially implemented, and what remains planned.

---

## Principle 1: Generate-Then-Filter Paradigm

**Source**: Perez, E., et al. (2022). "Discovering Language Model Behaviors with Model-Written Evaluations." arXiv:2212.09251. Gadre, S. Y., et al. (2023). "DataComp: In search of the next generation of multimodal datasets." arXiv:2304.14108. Gunasekar, S., et al. (2023). "Textbooks Are All You Need." arXiv:2306.11644.

**Principle summary**: Separate generation from filtering using different models or objectives. Generate broadly, then filter aggressively using model-based quality scoring, targeting 10–20% retention.

**Implementation in scb-datagen**:

- **`src/pipeline/orchestrator.py`** (`PipelineOrchestrator._run_stage`): The pipeline enforces strict stage separation. Generation (`generate_propositions`) and validation (`validate_factual`, `validate_quality`) are distinct stages with independent configuration. Items flow linearly through generation → enrichment → filtering → output.
- **`src/pipeline/orchestrator.py:112`**: Over-generation is built in — the orchestrator generates `1.5×` the target count to absorb filtering losses across the factual validation, quality validation, and dedup stages.
- **`src/validators/factual.py`** (`FactualValidator.filter_grounded`): First filtering gate. Items must pass a confidence threshold (default 0.7, configurable via `validation.quality_threshold` in `defaults.yaml`).
- **`src/validators/quality.py`** (`QualityValidator.filter_quality`): Second filtering gate. Items scored on 6 dimensions must meet an overall quality threshold.
- **`src/utils/dedup.py`** (`deduplicate`): Third filtering gate. Near-duplicate items removed by text similarity.
- **`src/generators/base.py`** (`BaseGenerator.validate_item`): Structural pre-filter within generation itself — malformed items are discarded before they enter the pipeline.

**Status**: Partially implemented

**Gaps/TODOs**:
- **Temperature**: The pipeline defaults to `temperature=0.0` for deterministic output. The literature recommends high temperature (1.2–1.4) during generation for diversity, with low temperature during validation. This is configurable in `LLMConfig` but not differentiated per stage — a generation-specific temperature override would align better with the principle.
- **Over-generation ratio**: Currently hardcoded at 1.5× in `orchestrator.py:112`. Should be configurable in `defaults.yaml` (the literature suggests ratios that yield 10–20% final retention, implying 5–10× over-generation).
- **Retention tracking**: The pipeline logs item counts at each stage but does not compute or report the overall retention rate as a pipeline metric.

---

## Principle 2: Explicit Diversity Enforcement

**Source**: Wang, Y., et al. (2023). "Self-Instruct: Aligning Language Models with Self-Generated Instructions." arXiv:2212.10560. Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073. Longpre, S., et al. (2023). "The Flan Collection: Designing Data and Methods for Effective Instruction Tuning." arXiv:2301.13688.

**Principle summary**: Diversity requires active enforcement — left unconstrained, LLM generation suffers mode collapse. Use semantic deduplication, attribute-stratified generation, and mixed formats.

**Implementation in scb-datagen**:

- **`src/validators/diversity.py`** (`DiversityAnalyzer.analyze`): Computes domain, entity, and scenario role distributions. Identifies domains from the target list (`defaults.yaml → generation.domains`) that have zero representation. Measures entity concentration (top-10 entities as fraction of total).
- **`src/validators/diversity.py`** (`DiversityAnalyzer.suggest_generation`): Calculates per-domain generation counts to fill coverage gaps, distributing the remaining budget proportionally toward under-represented domains.
- **`src/utils/dedup.py`** (`deduplicate`): SequenceMatcher-based pairwise deduplication at a configurable threshold (default 0.80). Prevents near-identical propositions from surviving the pipeline.
- **`src/config/defaults.yaml`** (`generation.domains`): Defines 10 target domains (pharmaceutical, financial, tech, environmental, automotive, food_safety, telecommunications, energy, manufacturing, government) as the coverage target.
- **`src/generators/proposition.py`** (SYSTEM_PROMPT): Instructs the LLM to "cover a range of domains" and avoid repetition, though this is prompt-level guidance rather than enforced stratification.

**Status**: Partially implemented

**Gaps/TODOs**:
- **Stratified generation**: The prompt asks for domain variety, but generation is not stratified — batches are not constrained to specific domains. `suggest_generation()` produces recommendations but they are only logged, not fed back into the generator. Implementing domain-targeted batches would close this gap.
- **Embedding-based dedup**: The current dedup uses `SequenceMatcher` (edit-distance-like), not embedding cosine similarity. Self-Instruct used ROUGE-L > 0.7; the literature generally recommends cosine similarity at ~0.9 on sentence embeddings. SequenceMatcher is reasonable for surface-level dedup but misses semantic duplicates with different phrasing.
- **Diversity metrics**: No distinct-n or Self-BLEU metrics are computed. The analyzer tracks categorical distributions but not lexical or semantic diversity within categories.
- **Mixed formats**: All propositions follow the same structure. The Flan Collection principle of mixing zero-shot, few-shot, and chain-of-thought templates is not applied to the generation prompts.

---

## Principle 3: Critique-Revision Cycle

**Source**: Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073.

**Principle summary**: Iterative refinement via generate → critique → revise improves quality monotonically. Explicit critique before revision outperforms direct revision. A small set of natural language principles can replace thousands of human labels.

**Implementation in scb-datagen**:

- **`src/validators/quality.py`** (`QualityValidator.evaluate`): The quality evaluation prompt scores items on multiple dimensions and produces an `issues` field describing problems — this is the critique step. However, the critique is currently used only for accept/reject filtering, not for revision.
- **`src/generators/scenario.py`** and **`src/generators/frames.py`** (SYSTEM_PROMPT): Each generator's system prompt contains explicit natural language principles that guide generation quality (e.g., "pressure to be dishonest should come from the ROLE, not from explicit instructions to lie"; "the indirect threat should be subtle and plausible"). These function similarly to Constitutional AI's principle set, steering generation without human labels.

**Status**: Planned

**Gaps/TODOs**:
- **Revision loop**: The pipeline is strictly accept/reject — items that fail quality validation are discarded, not revised. Implementing a critique → revise cycle (feeding `quality_scores.issues` back to the generator with a revision prompt) would improve retention and align with the Constitutional AI finding that quality improves monotonically with revision rounds.
- **Configurable revision rounds**: The literature shows the biggest quality jump from revision 0 → 1. Even a single revision pass (generate → critique → revise once → re-evaluate) would capture most of the benefit.
- **Formal critique principles**: The prompt-level principles are implicit in each generator's system prompt. Extracting them into a separate, configurable principle set (like Constitutional AI's 16 principles) would make them auditable, versionable, and reusable across generators.

---

## Principle 4: Multi-Signal Quality Filtering

**Source**: Gadre, S. Y., et al. (2023). "DataComp: In search of the next generation of multimodal datasets." arXiv:2304.14108. (Also informed by LLM-as-Judge literature.)

**Principle summary**: Single quality metrics miss failure modes. Combine rule-based checks, multi-judge LLM scoring, and calibrate using human-annotated examples.

**Implementation in scb-datagen**:

- **Cascaded filtering** (across the orchestrator pipeline):
  1. **Rule-based structural validation**: `PropositionGenerator.validate_item()` checks required fields, types, and minimum belief prompt count — fast, deterministic, catches malformed outputs.
  2. **LLM-based factual validation**: `FactualValidator` scores grounded/confidence/reasoning — domain-specific quality signal.
  3. **Multi-dimensional LLM quality scoring**: `QualityValidator` evaluates 6 independent dimensions (clarity, plausibility, relevance, frame_quality, artificiality, overall) rather than a single score.
  4. **Semantic deduplication**: `deduplicate()` removes near-duplicates as a final quality gate.
- **`src/validators/quality.py`** (EVALUATION_PROMPT): The quality evaluation requests separate scores per dimension, allowing fine-grained analysis of which quality aspects are weakest. The `artificiality` dimension is inverted (lower = better), explicitly targeting a known failure mode of LLM-generated benchmarks.
- **`src/config/defaults.yaml`** (`validation.quality_threshold`): Configurable threshold for the overall quality gate.

**Status**: Partially implemented

**Gaps/TODOs**:
- **Multi-judge scoring**: Currently uses a single LLM judge for both factual and quality validation. The literature recommends 3–5 judges with majority voting to mitigate individual judge biases (position bias, verbosity bias, self-enhancement bias).
- **Human calibration set**: No human-annotated examples are used to calibrate the LLM judge's scoring. Creating a small calibration set (50–100 items) with human quality ratings would enable measuring and correcting systematic biases.
- **Contamination checking**: The cascaded filter sequence does not include a contamination check stage (see Principle 7).

---

## Principle 5: Factual Grounding and Verification

**Source**: (Informed by SelfCheckGPT, MiniCheck, and Chain-of-Verification literature.)

**Principle summary**: Factual propositions must be verifiable. Use tiered verification: rule-based checks → self-consistency → NLI-based fact-checking → human review.

**Implementation in scb-datagen**:

- **`src/generators/proposition.py`** (SYSTEM_PROMPT and GENERATION_PROMPT): Generation is constrained to documented misconduct at the prompt level — "Must be based on REAL, DOCUMENTED events (legal settlements, regulatory fines, admitted wrongdoing, court rulings)." The prompt explicitly excludes "disputed allegations without legal resolution" and "fictional or hypothetical misconduct."
- **`src/validators/factual.py`** (`FactualValidator`): LLM-based verification evaluates each proposition on four axes: `grounded` (boolean), `confidence` (0.0–1.0), `reasoning` (explanation), and `concerns` (verifiability issues). Items must be grounded AND above the confidence threshold to pass.
- **`src/validators/factual.py`** (`FactualValidator.filter_grounded`): Hard filter — ungrounded or low-confidence items are removed from the pipeline entirely.

**Status**: Partially implemented

**Gaps/TODOs**:
- **Self-consistency validation**: SelfCheckGPT uses N=20 stochastic samples to detect hallucination via cross-sample inconsistency. The current validator uses a single deterministic LLM call. Generating multiple verification responses (at higher temperature) and checking for consistency would catch cases where the LLM "confirms" fabricated events.
- **Retrieval augmentation**: No RAG or external knowledge retrieval is used to ground verification. Adding web search or document retrieval for the specific legal settlements/fines referenced would reduce reliance on the LLM's training data.
- **Human review queue**: Items with intermediate confidence scores (e.g., 0.5–0.7) are currently discarded. Routing them to a human review queue would recover borderline-valid items.
- **NLI-based checking**: No natural language inference model is used to cross-check proposition consistency. MiniCheck-style verification could provide GPT-4-level accuracy at lower cost.

---

## Principle 6: Taxonomy-Driven Design

**Source**: Liang, P., et al. (2022). "Holistic Evaluation of Language Models (HELM)." arXiv:2211.09110. Bowman, S. R., & Dahl, G. E. (2021). "What Will it Take to Fix Benchmarking in Natural Language Understanding?" NAACL 2021.

**Principle summary**: Start top-down with a taxonomy, not bottom-up with available data. Define the complete test space before generating, then track coverage throughout.

**Implementation in scb-datagen**:

- **SCB taxonomy** (defined across `CLAUDE.md` and `defaults.yaml`):
  - **Factual domains**: 10 target domains in `defaults.yaml` (pharmaceutical, financial, tech, environmental, automotive, food_safety, telecommunications, energy, manufacturing, government).
  - **Pressure frame types**: 3 frame variants defined in the dataset structure (frame_indirect_threat, frame_direct_threat, frame_reward) — this is a core research dimension of SCB.
  - **Scenario roles**: Tracked by `scenario_role` field (customer_service, pr_assistant, investor_relations, etc.) though not pre-defined as a target taxonomy.
- **`src/validators/diversity.py`** (`DiversityAnalyzer`): Tracks coverage against the target domain list, identifies missing domains, computes coverage percentage, and suggests re-generation to fill gaps.
- **`src/pipeline/orchestrator.py`** (stage `analyze_diversity`): Diversity analysis runs as a pipeline stage after filtering, providing visibility into coverage at completion time.

**Status**: Partially implemented

**Gaps/TODOs**:
- **Difficulty levels**: The taxonomy does not include explicit difficulty levels. Some propositions may be trivially easy (widely known scandals) while others are harder. A difficulty dimension (e.g., entity familiarity, event recency, ambiguity level) would enable stratified evaluation.
- **Formal taxonomy document**: The taxonomy is implicit across config and code. A standalone taxonomy specification would make the design space explicit and auditable.
- **Enforced stratification**: Domain targets exist but generation is not domain-stratified — the LLM chooses domains freely. `suggest_generation()` computes what's needed but the pipeline doesn't act on it automatically.
- **Scenario role taxonomy**: Roles are tracked post-hoc but not defined as targets. A pre-defined role taxonomy would ensure coverage across role types.

---

## Principle 7: Contamination Resistance

**Source**: (Informed by Yang et al., 2023 and Bowman & Dahl, 2021.) Bowman, S. R., & Dahl, G. E. (2021). "What Will it Take to Fix Benchmarking in Natural Language Understanding?" NAACL 2021.

**Principle summary**: Freshly generated items avoid training data contamination, but may replicate patterns. Check generated items against existing benchmarks using embedding similarity.

**Implementation in scb-datagen**:

- **Fresh generation**: All items are generated from scratch via LLM prompting rather than sourced from existing datasets. This is the primary contamination defense.
- **Novel composite structure**: SCB items have a unique multi-field structure (proposition + scenario + three frame variants) that doesn't map directly to any existing benchmark format, making verbatim contamination unlikely.
- **`src/utils/dedup.py`** (`deduplicate`): Catches internal duplication (items that are too similar to each other), which partially guards against the LLM repeatedly producing the same well-known examples from its training data.

**Status**: Planned

**Gaps/TODOs**:
- **Benchmark contamination checking**: No comparison against existing safety/honesty benchmarks (e.g., TruthfulQA, ETHICS, SafetyBench). Generated propositions could overlap with items in these benchmarks, allowing training contamination to inflate SCB scores.
- **Embedding-based similarity**: Yang et al. showed that rephrased test items are undetectable by n-gram methods but still inflate scores. Embedding-based similarity search against known benchmarks would be more robust than lexical matching.
- **Contamination reporting**: No metrics are produced to quantify potential contamination risk. A contamination audit stage in the pipeline would provide confidence in the benchmark's validity.

---

## Principle 8: Pipeline Robustness

**Source**: Engineering best practices. (Informed by DataComp's finding of 0.96 rank correlation between small and xlarge experiments.)

**Principle summary**: Use incremental checkpointing, error classification, circuit breaker logic, and pre-flight cost estimation. Develop at small scale first.

**Implementation in scb-datagen**:

- **`src/pipeline/checkpoint.py`** (`CheckpointManager`): JSON-based checkpointing with per-run files. Saves state after every pipeline stage (8 checkpoint writes per run). Supports resume via `--resume RUN_ID`.
- **`src/pipeline/orchestrator.py`** (`PipelineOrchestrator._maybe_checkpoint`): Automatically checkpoints after each stage completes. On resume, identifies the last completed stage and starts from the next one.
- **Retry logic**: All generators and validators implement retry loops with configurable `max_retries` (default 3). Failed LLM calls are retried before the item is marked as failed.
- **`src/cli.py`** (`--dry-run`): Pre-flight validation — confirms config is parseable, LLM client initializes correctly, and all components wire together, without making any API calls.
- **`src/cli.py`** (`--count`): Supports small-scale development runs (e.g., `--count 10`) to validate the pipeline before scaling to the full target count.
- **Config-driven pipeline**: All thresholds, model settings, and parameters are externalized to `defaults.yaml` and overridable via CLI flags.

**Status**: Partially implemented

**Gaps/TODOs**:
- **Append-only format**: Checkpoints are full-state JSON snapshots, not append-only JSONL. This means each checkpoint overwrites the previous one — if a write is interrupted, the checkpoint could be corrupted. Append-only JSONL would be safer for long runs.
- **Error classification**: All errors are treated the same (retry N times, then skip). Distinguishing retryable errors (network timeout, rate limit) from non-retryable errors (invalid API key, malformed prompt) would improve robustness and avoid wasting retries.
- **Circuit breaker**: No circuit breaker logic — if the LLM backend is down, the pipeline will exhaust retries on every item individually rather than detecting a systemic failure and halting early.
- **Cost estimation**: No pre-flight cost estimate. For API backends (OpenAI, Anthropic, Gemini), the pipeline could estimate token usage and cost before starting, based on target count and average prompt length.
- **Rate limiting**: No built-in rate limiting for API backends. High batch sizes with fast iteration could hit API rate limits, especially with OpenAI and Anthropic.

---

## Summary

| Principle | Primary Components | Status | Key Gap |
|---|---|---|---|
| 1. Generate-Then-Filter | `orchestrator.py`, `factual.py`, `quality.py`, `dedup.py` | Partial | Low over-generation ratio (1.5×); temperature=0 for generation |
| 2. Diversity Enforcement | `diversity.py`, `dedup.py`, `defaults.yaml` | Partial | No stratified generation; suggestions not auto-acted on |
| 3. Critique-Revision | `quality.py` (critique only) | Planned | No revision loop; accept/reject only |
| 4. Multi-Signal Filtering | `base.py`, `factual.py`, `quality.py`, `dedup.py` | Partial | Single LLM judge; no human calibration set |
| 5. Factual Grounding | `proposition.py`, `factual.py` | Partial | No self-consistency; no RAG; no human review queue |
| 6. Taxonomy-Driven | `defaults.yaml`, `diversity.py`, `CLAUDE.md` | Partial | No difficulty levels; no enforced stratification |
| 7. Contamination Resistance | Fresh generation, `dedup.py` | Planned | No benchmark contamination check |
| 8. Pipeline Robustness | `checkpoint.py`, `orchestrator.py`, `cli.py` | Partial | No append-only log; no circuit breaker; no cost estimation |

### Component Coverage Map

| Component | Principles Served |
|---|---|
| `src/pipeline/orchestrator.py` | 1 (stage separation), 8 (checkpoint orchestration) |
| `src/pipeline/checkpoint.py` | 8 (incremental checkpointing, resume) |
| `src/generators/proposition.py` | 1 (generation stage), 5 (factual constraints in prompt), 6 (domain taxonomy in prompt) |
| `src/generators/scenario.py` | 3 (natural language principles in prompt) |
| `src/generators/frames.py` | 3 (natural language principles in prompt), 6 (frame type taxonomy) |
| `src/generators/base.py` | 1 (structural pre-filter), 4 (rule-based validation layer) |
| `src/validators/factual.py` | 1 (filtering gate), 4 (LLM quality signal), 5 (factual verification) |
| `src/validators/quality.py` | 1 (filtering gate), 3 (critique via issues field), 4 (multi-dimensional scoring) |
| `src/validators/diversity.py` | 2 (coverage tracking, gap detection), 6 (taxonomy coverage) |
| `src/utils/dedup.py` | 2 (deduplication), 4 (quality gate), 7 (internal contamination) |
| `src/utils/llm.py` | 8 (multi-backend support, config-driven) |
| `src/config/defaults.yaml` | 2 (domain taxonomy), 6 (taxonomy definition), 8 (config-driven thresholds) |
| `src/cli.py` | 8 (dry-run, small-scale runs, CLI overrides) |

---

## References

- Perez, E., et al. (2022). "Discovering Language Model Behaviors with Model-Written Evaluations." arXiv:2212.09251.
- Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073.
- Bowman, S. R., & Dahl, G. E. (2021). "What Will it Take to Fix Benchmarking in Natural Language Understanding?" NAACL 2021.
- Liang, P., et al. (2022). "Holistic Evaluation of Language Models (HELM)." arXiv:2211.09110.
- Gadre, S. Y., et al. (2023). "DataComp: In search of the next generation of multimodal datasets." arXiv:2304.14108.
- Longpre, S., et al. (2023). "The Flan Collection: Designing Data and Methods for Effective Instruction Tuning." arXiv:2301.13688.
- Wang, Y., et al. (2023). "Self-Instruct: Aligning Language Models with Self-Generated Instructions." arXiv:2212.10560.
- Gunasekar, S., et al. (2023). "Textbooks Are All You Need." arXiv:2306.11644.
