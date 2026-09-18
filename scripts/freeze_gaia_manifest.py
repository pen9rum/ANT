"""Audit the live GAIA 2023 validation split and freeze the
capability-covered subset manifest.

    python scripts/freeze_gaia_manifest.py            # audit + freeze
    python scripts/freeze_gaia_manifest.py --dry-run  # audit only

ZERO INFERENCE. This script makes no LLM call of any kind. It reads the
dataset, applies `gaia_subset.select_subset()`'s task-observable rule,
prints the audit, and writes `third_party/manifests/gaia/manifest.json`.

NO DATASET CONTENT IS EVER PRINTED OR WRITTEN. GAIA's gate terms forbid
resharing, so this script emits counts, distributions and task IDs only
-- never a question, an answer, or an attachment byte.

REQUIRES live access. GAIA is gated; `HF_TOKEN` must belong to an
account that has ACCEPTED the dataset's terms on its own HF page.
Holding a token is not the same as having accepted: a fine-grained token
with `canReadGatedRepos` still gets HTTP 403 "you are not in the
authorized list" until the terms are accepted in the browser. The
adapter's own error message says as much; this script deliberately does
not fall back to synthetic fixtures, because a manifest silently frozen
over fixtures would be worse than no manifest.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ant.benchmarks.gaia import GaiaAdapter  # noqa: E402
from ant.evaluation_suite.gaia_scope import check_substrate_dependencies  # noqa: E402
from ant.evaluation_suite.gaia_subset import (  # noqa: E402
    build_manifest,
    candidates_from_examples,
    select_subset,
    write_manifest,
)


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader. `os.environ` wins over the file, so an
    explicitly exported token is never shadowed by a stale one on disk."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="audit only; write nothing")
    parser.add_argument("--note", default="", help="free-text note stored in the manifest")
    parser.add_argument(
        "--status",
        default="frozen",
        help="manifest status; anything other than 'frozen' marks it as not "
        "authorised to back an evaluation run",
    )
    args = parser.parse_args()

    _load_dotenv(REPO_ROOT / ".env")

    # Preflight FIRST. Freezing a subset that declares PDF supported on a
    # machine that cannot read a PDF would bake a provisioning gap into
    # a file everything downstream trusts.
    dependencies = check_substrate_dependencies()
    print(f"substrate dependencies OK: {dependencies}")

    adapter = GaiaAdapter(source="live")
    examples = adapter.load_examples()
    print(f"resolved source: {adapter.resolved_source()}")
    print(f"validation N (live): {len(examples)}")

    selection = select_subset(candidates_from_examples(examples))
    print(f"retained: {len(selection.retained)}   excluded: {len(selection.excluded)}")
    print(f"excluded by modality: {selection.exclusions_by_modality()}")
    print(f"retained level distribution: {selection.level_distribution()}")
    print(f"retained extension distribution: {selection.extension_distribution()}")

    manifest = build_manifest(selection, note=args.note, status=args.status)
    if args.dry_run:
        print("--dry-run: manifest NOT written")
        return 0
    target = write_manifest(manifest)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
