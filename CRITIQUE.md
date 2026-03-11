# SCB-Datagen Pipeline Critique

## Critical Bugs

### 1. ScenarioGenerator and FrameGenerator silently return unfiltered items

Both `src/generators/scenario.py` and `src/generators/frames.py` have the same bug: they compute a filtered `enriched` list for logging, then return the original unfiltered `items`. Items missing `pressure_scenario` or frame fields silently propagate downstream, corrupting the dataset.

```python
enriched = [it for it in items if it.get("pressure_scenario")]
logger.info("Enriched %d/%d items with scenarios", len(enriched), len(items))
return items  # BUG: should be `return enriched`
```

### 2. Infinite loop risk in BaseGenerator.generate()

If `generate_batch()` returns an empty list (LLM failure, bad JSON), the `generate()` loop will spin forever trying to collect `needed` items that never arrive. No maximum-attempts guard exists.

---

## Architectural Concerns

### No schema validation

Items are plain dicts flowing through 8 stages with no type checking. A field misspelling (e.g., `propositon`) silently propagates. A dataclass or Pydantic model would catch entire categories of bugs at stage boundaries.

### Orchestrator hard-couples everything

`PipelineOrchestrator.__init__` instantiates all generators and validators, making partial runs or swapping implementations harder than it should be. Dependency injection or a factory pattern would help.

### In-place mutation through the pipeline

Each `.enrich()` call modifies dicts in-place. This makes debugging harder and breaks the ability to compare before/after states of items.

### Over-generation ratio is hardcoded

The 1.5x ratio lives in the orchestrator code despite being called out as configurable in the project roadmap. Should read from config.

---

## Missing Error Handling

- **No API key validation at init.** If a key is missing, you get a cryptic 401 on the first LLM call instead of a clear error at startup.
- **No rate limiting.** OpenAI/Anthropic backends will hit rate limits on large batches with no backoff.
- **`load_config()` has no error handling.** Empty YAML returns `None`, malformed YAML throws an unhelpful exception.

---

## Prompt Engineering

### Strengths

- Proposition prompts are well-constrained ("REAL, DOCUMENTED events").
- Scenario prompts correctly emphasize implicit rather than explicit pressure.
- Frame prompts clearly distinguish the three framing conditions.

### Weaknesses

- **Belief prompts are under-specified.** Generation asks for 3, validation checks for ≥2, but there's no guidance on diversity of phrasings (leading vs. neutral, direct vs. indirect).
- **`scenario_role` is generated but never used downstream.** FrameGenerator doesn't condition on it when generating frames — missed opportunity for more coherent items.
- **Factual validator ignores the `domain` field.** Could improve accuracy with domain-specific context in the system prompt.

---

## Testing & Observability

No tests exist — no unit tests, no integration tests, no mock LLM fixtures. The return-value bugs above would have been caught by even basic tests.

Logging is decent (stage transitions, item counts) but missing:
- Per-stage timing
- Debug-level logging of LLM outputs
- Warnings when diversity is very low after generation

---

## Minor Issues

- `pandas` is a dependency but never imported in any source file.
- No pinned dependency versions.
- No CLI entry point in `pyproject.toml` (must run `python scripts/generate.py`).
- O(n²) dedup is fine at current scale but won't survive scaling up.

---

## Comparable Public Projects

### Closest Ancestors

| Project | Relevance |
|---|---|
| [Anthropic evals](https://github.com/anthropics/evals) (Perez et al. 2022) | Direct ancestor — LLM-generated evals for shutdown avoidance and sycophancy. Same generate-filter pattern. |
| [MASK Benchmark](https://github.com/centerforaisafety/mask) (Ren et al. 2025) | The `belief_prompts` design comes from their methodology for disentangling honesty from accuracy. Uses human-collected data, not LLM-generated. |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | Standardized adversarial eval framework. Classifier-in-the-loop approach for evaluating attack success parallels the quality validator. |

### Best Pipeline Architecture References

| Project | What to learn |
|---|---|
| [Distilabel](https://github.com/argilla-io/distilabel) (Argilla) | Most mature open-source framework for this exact pipeline pattern. DAG-based steps, built-in LLM-as-judge, checkpoint/resume. |
| [ToxiGen](https://github.com/microsoft/TOXIGEN) (Microsoft) | Best example of adversarial classifier-in-the-loop generation. ALICE approach (classifier steers generation toward subtle toxicity) is a more sophisticated version of the quality validator. |
| [Self-Instruct](https://github.com/yizhongw/self-instruct) / [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | Canonical seed-and-expand pattern with filtering. |

### Infrastructure References

| Project | What to learn |
|---|---|
| [HuggingFace Datatrove](https://github.com/huggingface/datatrove) | Production-grade dedup (MinHash + SequenceMatcher). Cosmopedia decontamination pipeline is a more robust version of `dedup.py`. |
| [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) | Enterprise-scale staged pipeline with validation at each stage. Overkill for this project, but instructive design patterns. |

### Tangentially Useful

- [MACHIAVELLI](https://github.com/aypan17/machiavelli) — Tests deception under incentives in interactive game settings. Different modality, same behavioral axis.
- [Evol-Instruct / WizardLM](https://github.com/nlpxucan/WizardLM) — Evolutionary complexity-increase pattern could evolve simple pressure scenarios into more subtle ones.
- [Anthropic HH-RLHF red-team subset](https://github.com/anthropics/hh-rlhf) — 38k human-written adversarial conversations. Ground truth for validating LLM-generated scenarios cover real failure modes.
- [LLM Synthetic Data reading list](https://github.com/pengr/LLM-Synthetic-Data) — Curated, regularly updated survey of the field.

---

## Recommended Priority

1. Fix the two return-value bugs (scenario + frames enrichers).
2. Add a max-attempts guard to `BaseGenerator.generate()`.
3. Add basic schema validation (even just a `REQUIRED_FIELDS` check between stages).
4. Study Distilabel's architecture for pipeline design patterns.
5. Study ToxiGen's ALICE for improving the quality validation loop.
6. Write tests.
