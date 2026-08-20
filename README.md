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
```

Ask a local evidence question:

```powershell
ant ask "Where is authentication handled?" --index .ant --max-rounds 2
```

By default, `ask` stores the full evidence state and recruitment trace in `.ant/ant.sqlite3`.
Use `--no-save-trace` for one-off experiments.

Use OpenAI later by setting:

```powershell
$env:OPENAI_API_KEY="..."
```

## Version Control

The local git remote is expected to be:

```text
origin https://github.com/pen9rum/ANT.git
```

Create the first commit after tests pass, then push `main`.
