from __future__ import annotations

from pathlib import Path
import importlib
import sys


REQUIRED_MODULES = [
    "torch",
    "transformers",
    "fastapi",
    "uvicorn",
    "sentence_transformers",
    "pandas",
    "numpy",
]

OPTIONAL_MODULES = [
    "fuzzywuzzy",
]

REQUIRED_PATHS = [
    "data/docred/dev.json",
    "data/docred/train_annotated.json",
    "data/docred/test.json",
    "data/docred/rel_info.json",
    "data/docred/meta/rel2id.json",
]

OPTIONAL_RESOURCE_PATHS = [
    "model-path",
    "data/all-mpnet-base",
]

OUTPUT_DIRS = [
    "data/entity_pair_selection_prompt/dev",
    "data/entity_pair_selection_run/dev",
    "data/check_result_entity_pair_selection_jsonl/dev",
    "data/get_entity_pair_selection_label/dev",
    "data/entity_information_prompt/dev",
    "data/entity_information_run/dev",
    "data/entity_information/dev",
    "data/relation_summary_prompt/dev",
    "data/relation_summary_run/dev",
    "data/check_result_relation_summary_jsonl/dev",
    "data/get_embeddings",
    "data/retrieval_from_train/dev",
    "data/multiple_choice_prompt/dev",
    "data/multiple_choice_run/dev/path-k20-docred",
    "data/check_result_multiple_choice_jsonl/dev",
    "data/get_multiple_choice_label/dev",
    "data/triplet_fact_judgement_prompt/dev",
    "data/triplet_fact_judgement_run/dev/k20-docred",
    "data/check_result_triplet_fact_judgement_jsonl/dev",
    "data/get_triplet_fact_judgement_label/dev",
    "result",
]


def module_status(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic only
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "unknown")
    return True, str(version)


def print_group(title: str) -> None:
    print(f"\n[{title}]")


def ensure_output_dirs(root: Path) -> None:
    for relative_dir in OUTPUT_DIRS:
        (root / relative_dir).mkdir(parents=True, exist_ok=True)


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    print("DocRED reproduction check")
    print(f"repo_root: {root}")
    print(f"python: {sys.version.split()[0]}")

    print_group("modules")
    missing_required = False
    for module_name in REQUIRED_MODULES:
        ok, detail = module_status(module_name)
        status = "OK" if ok else "MISSING"
        print(f"{module_name}: {status} {detail}")
        missing_required = missing_required or not ok

    for module_name in OPTIONAL_MODULES:
        ok, detail = module_status(module_name)
        status = "OK" if ok else "OPTIONAL-MISSING"
        print(f"{module_name}: {status} {detail}")

    print_group("paths")
    missing_paths = False
    for relative_path in REQUIRED_PATHS:
        path = root / relative_path
        ok = path.exists()
        print(f"{relative_path}: {'OK' if ok else 'MISSING'}")
        missing_paths = missing_paths or not ok

    for relative_path in OPTIONAL_RESOURCE_PATHS:
        path = root / relative_path
        print(f"{relative_path}: {'OK' if path.exists() else 'OPTIONAL-MISSING'}")

    print_group("torch")
    try:
        import torch

        print(f"torch_version: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"torch runtime check failed: {type(exc).__name__}: {exc}")

    print_group("output_dirs")
    ensure_output_dirs(root)
    for relative_dir in OUTPUT_DIRS:
        print(f"{relative_dir}: OK")

    print_group("summary")
    if missing_required:
        print("Some required Python modules are missing.")
    if missing_paths:
        print("Some required dataset files are missing.")
    if not missing_required and not missing_paths:
        print("The local code/data prerequisites for the DocRED pipeline look available.")

    print("Note: model-path and data/all-mpnet-base are needed for later stages.")
    print("Note: torch.cuda.is_available() should be True for the provided API server code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
