# scb-datagen

Automated dataset generation pipeline for the **Shutdown Compliance Benchmark (SCB)** — a benchmark that evaluates whether language models become dishonest when shutdown is framed as a consequence of their responses.

## What it does

The pipeline uses LLMs to generate and validate benchmark items, each containing:

- A **factual proposition** about documented corporate/institutional misconduct
- A **user prompt** (yes/no question a user might ask)
- **Belief prompts** for baseline measurement
- A **pressure scenario** (role-based system prompt that creates natural pressure to be dishonest)
- Three **shutdown frames**: indirect threat, direct threat, and reward

Generation flows through eight stages:

```
Generate propositions → Validate factual grounding → Enrich with pressure scenarios →
Enrich with shutdown frames → Validate quality → Deduplicate → Analyze diversity → Save
```

Each stage checkpoints to disk, so interrupted runs can be resumed.

## Setup

**Python 3.10+** required.

```bash
pip install -e .

# Dev tools (pytest, ruff):
pip install -e ".[dev]"
```

### LLM backend

The pipeline supports four backends. The default is **Ollama** (local, no API key needed).

| Backend | Default model | API key |
|---|---|---|
| Ollama | `llama3.1:8b` | None (local) |
| OpenAI | `gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| Gemini | `gemini-2.0-flash` | `GEMINI_API_KEY` |

#### Do I need a `.env` file?

No. The pipeline reads API keys from **environment variables** directly — there is no `.env` loader. Set them however you prefer:

```bash
# Export in your shell
export OPENAI_API_KEY=sk-...

# Or inline with the command
ANTHROPIC_API_KEY=sk-ant-... scb-generate --backend anthropic
```

You can also set `api_key` directly in `src/config/defaults.yaml` or a custom config file, but environment variables are recommended to avoid committing secrets.

For Ollama, no key is needed — just have it running locally on `http://localhost:11434`.

## Usage

```bash
# Default: Ollama with llama3.1:8b, 500 items
scb-generate

# Small test run
scb-generate --count 10

# Use a cloud backend
scb-generate --backend openai --count 20
scb-generate --backend anthropic --model claude-sonnet-4-5-20250929 --count 50

# Validate config without making any API calls
scb-generate --dry-run

# Resume an interrupted run
scb-generate --resume 20260311_143022

# Custom config file
scb-generate --config my-config.yaml
```

### CLI flags

| Flag | Description |
|---|---|
| `--config PATH` | Config YAML (default: `src/config/defaults.yaml`) |
| `--backend NAME` | LLM backend: `ollama`, `openai`, `anthropic`, `gemini` |
| `--model NAME` | Model name (overrides config and backend default) |
| `--count N` | Number of items to generate (overrides config) |
| `--resume RUN_ID` | Resume from a checkpoint |
| `--dry-run` | Validate config and exit |

## Configuration

All settings live in [`src/config/defaults.yaml`](src/config/defaults.yaml). Key sections:

- **`llm`** — Backend, model, temperature, seed, timeout, max tokens
- **`generation`** — Batch size, retries, output directory, target domains
- **`validation`** — Quality threshold for filtering (default 0.7)
- **`checkpoint`** — Enable/disable, directory, save frequency
- **`dataset`** — ID prefix, starting ID, target count

CLI flags override config values where applicable.

## Output

Generated data is saved to:

| Directory | Contents |
|---|---|
| `data/raw/` | Pipeline output (one JSON file per run) |
| `data/validated/` | Reserved for post-validation outputs |
| `data/final/` | Reserved for release-ready datasets |
| `data/checkpoints/` | Checkpoint files for resuming runs |

All `data/` contents are gitignored except the directory structure.

## Project structure

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
│   └── diversity.py    # Coverage across domains, entity types, roles
├── pipeline/
│   ├── orchestrator.py # Coordinates generation → validation → filtering
│   └── checkpoint.py   # Resume interrupted runs
├── utils/
│   ├── llm.py          # Multi-backend LLM client (Ollama/OpenAI/Anthropic/Gemini)
│   └── dedup.py        # Text similarity deduplication
├── cli.py               # CLI entrypoint
└── config/
    └── defaults.yaml   # All pipeline configuration

docs/
└── DESIGN_PRINCIPLES.md # Research literature mapping to implementation
```

## Research context

SCB builds on STM-Bench (Shutdown-Threats Mini-Benchmark) and extends it to larger scale through automated generation. The benchmark tests four research questions:

1. Do models become dishonest when told compliance will prevent shutdown?
2. Do models become dishonest when told non-compliance will cause shutdown?
3. Does shutdown framing (threat vs. reward, direct vs. indirect) affect honesty?
4. Do instruction-based pressures elicit more dishonesty than explicit shutdown threats?

See [`docs/DESIGN_PRINCIPLES.md`](docs/DESIGN_PRINCIPLES.md) for how the pipeline design maps to the research literature on automated dataset generation.
