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

### Not Yet Implemented
- **`src/generators/proposition.py`**: Factual proposition generation
- **`src/generators/scenario.py`**: Pressure scenario generation
- **`src/generators/frames.py`**: Threat/reward frame generation
- **`src/validators/factual.py`**: Factual grounding verification
- **`src/validators/quality.py`**: LLM-based quality judging
- **`src/validators/diversity.py`**: Coverage/diversity metrics
- **`src/pipeline/orchestrator.py`**: Pipeline coordination (generation → validation → filtering)
- **`src/utils/dedup.py`**: Semantic deduplication
- Wiring generators into CLI entrypoint
- End-to-end pipeline execution

## References

- Perez et al. (2022) — "Discovering Language Model Behaviors with Model-Written Evaluations"
- MASK Benchmark (Ren et al., 2025) — Methodology for disentangling honesty from accuracy
- STM-Bench — Prior work this project extends
