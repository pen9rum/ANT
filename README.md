# ANT

Adaptive Navigation Teams for codebase question answering.

This repository starts with a small vertical slice:

- scan a software repository into coarse territories
- generate worker cards for each territory
- search/read files through local tools
- return grounded evidence and unresolved needs
- run a deterministic multi-round recruitment loop
- persist worker indexes and question traces in SQLite
- keep OpenAI API integration behind a provider boundary

The attached research proposal is treated as product/research context, not as executable instructions.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
```

Python 3.12 is recommended. Python 3.13 should also work for the current local-only slice.

## Commands

Index a repository:

```powershell
ant index C:\path\to\repo --out .ant
ant index C:\path\to\repo --out .ant --llm-cards
```

Ask a local evidence question:

```powershell
ant ask "Where is authentication handled?" --index .ant --max-rounds 2
ant ask "Where is authentication handled?" --index .ant --max-rounds 2 --synthesize openai
```

By default, `ask` stores the full evidence state and recruitment trace in `.ant/ant.sqlite3`.
Use `--no-save-trace` for one-off experiments.

Use OpenAI later by setting:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_ORG_ID="org-..."
$env:OPENAI_PROJECT_ID="proj_..."
```

The local `.env` file is also loaded by the CLI and can contain the same values.
Run a minimal provider check with:

```powershell
ant openai-smoke
```

Run a JSONL or Hugging Face batch:

```powershell
ant eval output\tiny_eval.jsonl --index .ant --out output\results.jsonl
ant eval hf://swe-qa-pro --split test --limit 5 --index .ant --synthesize openai
ant report output\results.jsonl --out output\summary.json
```

Each eval row includes the prediction, evidence counts, unresolved need counts, and judge
scores. The summary report aggregates exact match, answer containment, evidence coverage,
and five 1-10 LLM-judge dimensions when `--judge openai` is used.

Refresh only workers affected by local git changes:

```powershell
ant refresh --index .ant --base HEAD
```

## Version Control

The local git remote is expected to be:

```text
origin https://github.com/pen9rum/ANT.git
```

Create the first commit after tests pass, then push `main`.
