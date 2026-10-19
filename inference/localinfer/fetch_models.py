"""Download the pinned Kronos base model and tokenizer into the local Hugging Face cache.

The model and tokenizer IDs and revisions come from a checkpoint's resolved config, so
local inference loads exactly the weights the run was trained on. Set HF_HOME to choose
the cache location; inference then loads them with local_files_only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pinned_models(checkpoint: Path) -> list[tuple[str, str]]:
    from stock_forecasting.config import ExperimentConfig

    model = ExperimentConfig.from_yaml(Path(checkpoint) / "resolved-config.yaml").model
    pairs = [(model.time_series_model_id, model.time_series_model_revision),
             (model.time_series_tokenizer_id, model.time_series_tokenizer_revision)]
    if any(not repo or not revision for repo, revision in pairs):
        raise ValueError("The checkpoint config does not pin both Kronos repositories")
    return pairs


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Any local checkpoint directory of the runs to infer.")
    args = parser.parse_args(argv)
    paths = {}
    for repo, revision in pinned_models(args.checkpoint):
        paths[repo] = {"revision": revision,
                       "path": snapshot_download(repo_id=repo, revision=revision)}
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
