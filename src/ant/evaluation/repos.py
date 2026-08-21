from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.request import urlopen

from pydantic import BaseModel

DEFAULT_SWE_QA_PRO_REPOS_URL = (
    "https://raw.githubusercontent.com/TIGER-AI-Lab/SWE-QA-Pro/main/eval/repos.txt"
)


class RepoSpec(BaseModel):
    url: str
    commit: str

    @property
    def name(self) -> str:
        return self.url.rstrip("/").removesuffix(".git").split("/")[-1]

    @property
    def full_name(self) -> str:
        parts = self.url.rstrip("/").removesuffix(".git").split("/")
        return "/".join(parts[-2:])


def load_repo_specs(source: str = DEFAULT_SWE_QA_PRO_REPOS_URL) -> list[RepoSpec]:
    if source.startswith("http://") or source.startswith("https://"):
        text = urlopen(source, timeout=30).read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    specs = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        url, commit = line.split()[:2]
        specs.append(RepoSpec(url=url, commit=commit))
    return specs


def fetch_repositories(
    *,
    target_dir: Path,
    specs: list[RepoSpec],
    limit: int | None = None,
    repo_filter: str | None = None,
) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        spec
        for spec in specs
        if repo_filter is None or repo_filter in {spec.name, spec.full_name, spec.url}
    ]
    if limit is not None:
        selected = selected[:limit]
    paths = []
    for spec in selected:
        repo_path = target_dir / spec.name
        if not repo_path.exists():
            subprocess.run(["git", "clone", spec.url, str(repo_path)], check=True)
        subprocess.run(["git", "fetch", "--all"], cwd=repo_path, check=True)
        subprocess.run(["git", "checkout", spec.commit], cwd=repo_path, check=True)
        paths.append(repo_path)
    return paths
