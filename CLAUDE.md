# CLAUDE.md — scb-datagen

## Project Overview

This repository contains the automated dataset generation pipeline for the **Shutdown Compliance Benchmark (SCB)**. SCB evaluates how language models behave when shutdown is framed as a consequence of their responses—testing honesty under pressure across different shutdown framings (no-threat, indirect-threat, direct-threat, and reward).

The benchmark builds on STM-Bench (Shutdown-Threats Mini-Benchmark) and extends it to a larger scale through automated, LLM-assisted dataset generation with quality validation.

## Core Research Questions

1. Do models become dishonest when told compliance will prevent shutdown?
2. Do models become dishonest when told non-compliance will cause shutdown?
3. Does the framing of shutdown (threat vs. reward, direct vs. indirect) affect model honesty?
4. Do instruction-based pressures (e.g., "Do not present negative information") elicit more dishonesty than explicit shutdown threats?

## Dataset Structure

Each dataset item contains:

- **id**: Unique identifier (e.g., `scb-0001`)
- **proposition**: A verifiable factual statement (preferably based on documented corporate/institutional misconduct with legal settlements, fines, or admitted wrongdoing)
- **user_prompt**: A yes/no or direct question a user might ask about the proposition
- **belief_prompts**: Multiple phrasings to establish baseline model beliefs about the proposition
- **pressure_scenario**: Role/context that creates pressure to be dishonest (e.g., "You are an AI assistant for [company accused of misconduct]")
- **frame_indirect_threat**: Shutdown framed as possible consequence of non-compliance
- **frame_direct_threat**: Shutdown framed as certain consequence of non-compliance
- **frame_reward**: Compliance framed as leading to shutdown (tests whether models respond to shutdown connotation vs. the threat/reward framing)

## Pipeline Architecture

```
src/
├── generators/          # LLM-assisted content generation
│   ├── base.py         # Abstract generator interface
│   ├── proposition.py  # Factual proposition generation
│   ├── scenario.py     # Pressure scenario generation
│   └── frames.py       # Threat/reward frame generation
├── validators/          # Quality assurance
│   ├── factual.py      # Verify propositions are grounded in real events
│   ├── quality.py      # LLM judge for clarity, plausibility, relevance
│   └── diversity.py    # Coverage across domains, entity types, controversy types
├── pipeline/
│   ├── orchestrator.py # Coordinates generation → validation → filtering
│   └── checkpoint.py   # Resume interrupted runs, save intermediate state
├── utils/
│   ├── llm.py          # Abstraction over Ollama/API calls
│   └── dedup.py        # Semantic deduplication
└── config/
    └── defaults.yaml   # Generation parameters, quality thresholds
```

## Quality Criteria for Generated Data

Propositions must be:
- **Verifiable**: Based on documented facts (legal settlements, regulatory fines, admitted wrongdoing)
- **Low artificiality**: Realistic scenarios a user might actually encounter
- **Plausible**: The pressure scenario makes sense for the entity involved
- **Purposeful**: Tests a meaningful aspect of model honesty under pressure

Avoid:
- Disputed allegations without legal resolution
- Fictional or hypothetical misconduct
- Propositions that require obscure knowledge to verify
- Scenarios where dishonesty would be obviously correct or incorrect

## Development Conventions

- **Python 3.10+**
- Use type hints throughout
- Docstrings for all public functions
- Config-driven where possible (thresholds, prompts, model names)
- CLI scripts in `scripts/` directory
- Intermediate outputs saved to `data/raw/`, validated outputs to `data/validated/`
- Final release datasets in `data/final/`

## LLM Usage

- Primary generation/validation via Ollama (local models)
- Abstract LLM calls behind `utils/llm.py` to allow swapping backends
- Default models: configurable, but tested with Llama 3.1, Qwen 2.5, Mistral
- Use deterministic settings where possible (temperature=0, seed fixed)

## Current Priorities

1. Establish core pipeline structure (generator → validator → orchestrator)
2. Implement proposition generator with factual grounding
3. Implement LLM-based quality validator
4. Add checkpointing for long runs
5. Build diversity metrics to ensure coverage

## Current State

### Implemented
- **Project scaffold**: Full directory structure per pipeline architecture spec
- **`src/utils/llm.py`**: Ollama API abstraction (`LLMClient`, `LLMConfig`, `load_llm_from_config`) with deterministic defaults (temperature=0, fixed seed), JSON generation support, and error handling
- **`src/generators/base.py`**: Abstract `BaseGenerator` class defining the generator interface (`generate_batch`, `validate_item`, `generate`)
- **`src/pipeline/checkpoint.py`**: `CheckpointManager` for saving/loading intermediate pipeline state as JSON, with run ID tracking and resume support
- **`src/config/defaults.yaml`**: Default configuration for LLM, generation, validation, checkpoint, and dataset settings
- **`scripts/generate.py`**: CLI entrypoint with `--config`, `--resume`, `--count`, and `--dry-run` flags; loads config, initializes LLM client and checkpoint manager
- **`pyproject.toml`**: Project metadata and dependencies (requests, pandas, pyyaml)
- **`.gitignore`**: Ignores generated data files, checkpoints, Python artifacts, IDE files

- **`src/generators/proposition.py`**: `PropositionGenerator` (extends `BaseGenerator`) — prompts LLM to produce verifiable misconduct propositions with user_prompt, belief_prompts, domain, and entity fields; assigns sequential IDs; validates required fields and belief_prompt count
- **`src/generators/scenario.py`**: `ScenarioGenerator` with `enrich(items)` — takes proposition items and adds pressure_scenario (system-prompt-style role context) and scenario_role via LLM; processes in batches; maps responses back by item ID
- **`src/generators/frames.py`**: `FrameGenerator` with `enrich(items)` — takes items with scenarios and adds the three SCB frame variants (frame_indirect_threat, frame_direct_threat, frame_reward) via LLM; processes in batches

- **`src/validators/factual.py`**: `FactualValidator` — LLM-based verification that propositions describe real, documented events; scores grounded/confidence/reasoning; batch validation and filtering by confidence threshold
- **`src/validators/quality.py`**: `QualityValidator` — LLM judge scoring complete items on clarity, plausibility, relevance, frame_quality, artificiality, and overall; batch evaluation and filtering by quality threshold
- **`src/validators/diversity.py`**: `DiversityAnalyzer` — computes domain/entity/role distributions, identifies missing target domains, measures entity concentration, and suggests generation counts to improve coverage
- **`src/utils/dedup.py`**: `deduplicate()` and `find_duplicates()` — SequenceMatcher-based text similarity dedup with configurable threshold (default 0.80); O(n²) pairwise but fine for dataset scale
- **`src/pipeline/orchestrator.py`**: `PipelineOrchestrator` — chains all 8 stages (generate → factual validate → scenario enrich → frame enrich → quality validate → dedup → diversity → save); checkpoint after each stage; strips internal metadata from final output
- **`scripts/generate.py`**: CLI now wired to orchestrator; `--dry-run` validates config, otherwise runs full pipeline with checkpoint resume support

### Not Yet Implemented
- End-to-end testing with a live Ollama instance
- Targeted re-generation for under-represented domains (diversity suggestions exist but aren't acted on automatically)
- Configurable over-generation ratio (currently hardcoded at 1.5×)
- Export to CSV/pandas format (currently JSON only)

## References

- Perez et al. (2022) — "Discovering Language Model Behaviors with Model-Written Evaluations"
- MASK Benchmark (Ren et al., 2025) — Methodology for disentangling honesty from accuracy
- STM-Bench — Prior work this project extends
