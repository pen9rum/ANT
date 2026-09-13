"""WebWalkerQA infrastructure -- dataset loader, normalized record schema,
deterministic filtering, and a preparation-manifest builder ONLY.

This package deliberately does NOT crawl URLs, does NOT call an LLM, and
does NOT implement a web-navigation inference method. It exists so that
when a WebWalkerQA evaluation track is built later, the dataset and its
reproducible manifests are already wired into this evaluation suite. See
`prepare.py` for the CLI entry point and `loader.py`'s own module
docstring for the no-leakage contract on `source_websites`/`golden_path`.
"""

from __future__ import annotations

from ant.evaluation_suite.webwalkerqa.loader import (
    WebWalkerQaRecord,
    filter_by_difficulty,
    filter_by_domain,
    filter_by_hop_type,
    filter_english,
    inference_view,
    load_webwalkerqa_records,
    sample_deterministic,
)
from ant.evaluation_suite.webwalkerqa.manifest import (
    WebWalkerQaPrepManifest,
    build_prep_manifest,
)

__all__ = [
    "WebWalkerQaPrepManifest",
    "WebWalkerQaRecord",
    "build_prep_manifest",
    "filter_by_difficulty",
    "filter_by_domain",
    "filter_by_hop_type",
    "filter_english",
    "inference_view",
    "load_webwalkerqa_records",
    "sample_deterministic",
]
