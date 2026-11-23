"""Registry exclusively for native 409-observation/five-action app models."""

from pathlib import Path
from .core import ROOT, read_json
from .partial_agents import ALGORITHMS
from .partial_contract import configuration
from .partial_features_scaled import ARMS, load_scaled_policy
from . import product_features, phase3_features

RUNS = ROOT / "runs"


def _feature_arm(metadata):
    """Identify a checkpoint format without changing its saved feature version."""
    if metadata.get("actions") != 5 or metadata.get("observation_values") != 409:
        return None
    version = metadata.get("feature_version")
    feature_count = metadata.get("feature_count")
    if (
        version == phase3_features.VERSION
        and feature_count == phase3_features.FEATURE_COUNT
        and metadata.get("agent_state_values") == phase3_features.STATE
    ):
        return "product-v3"
    if (
        version == product_features.VERSION
        and feature_count == product_features.FEATURE_COUNT
    ):
        return "product-v2"
    if feature_count == 412:
        return next(
            (arm for arm, saved_version in ARMS.items() if saved_version == version),
            None,
        )
    return None


def _selection_matches(selection, algorithm, checkpoint_id, checksum):
    """A selected ID must also refer to the originally selected weight bytes."""
    record = selection.get(algorithm, {})
    return (
        record.get("id") == checkpoint_id
        and record.get("checkpoint_sha256") == checksum
    )


def _selection_scope(selected_phase3, selected_phase2, product_phase):
    if selected_phase3 and selected_phase2:
        return "Phase 2 retained by Phase 3"
    if selected_phase3:
        return "Phase 3 selected"
    if selected_phase2:
        return "Phase 2 baseline"
    if product_phase == 3:
        return "Phase 3 candidate"
    return "Phase 2 candidate"


def registry(root=None):
    """List completed native checkpoints; loading performs full weight validation.

    A cancelled or incomplete worker contributes no selectable checkpoints, even
    when it wrote one model before stopping. Historical selections remain read-only.
    """
    root = Path(root or RUNS)
    items = []
    selection = (
        read_json(root / "selection.json")
        if (root / "selection.json").is_file()
        else {}
    )
    best = selection.get("selected", {})
    phase3 = (
        read_json(root / "phase3-selection.json")
        if (root / "phase3-selection.json").is_file()
        else {}
    )
    phase3_best = phase3.get("selected", {})
    for file in sorted(root.glob("training/*/manifest.json"), reverse=True):
        try:
            run = read_json(file)
            if run.get("status") != "complete":
                continue
            for name, record in run.get("models", {}).items():
                if name not in ALGORITHMS or record.get("status") != "complete":
                    continue
                path = file.parent / record["checkpoint"]
                meta = read_json(path.with_suffix(".json"))
                arm = _feature_arm(meta)
                if arm is None:
                    continue
                item_id = run["run_id"] + "/" + name
                is_best = _selection_matches(
                    best, name, item_id, meta["checkpoint_sha256"]
                )
                is_phase3 = _selection_matches(
                    phase3_best, name, item_id, meta["checkpoint_sha256"]
                )
                scope = _selection_scope(is_phase3, is_best, run.get("product_phase"))
                state = (
                    "Best on development"
                    if is_best
                    else "Trained" if record["actions"] >= 10000 else "Starter"
                )
                items.append(
                    {
                        "id": run["run_id"] + "/" + name,
                        "algorithm": name,
                        "path": str(path),
                        "arm": arm,
                        "seed": run["seed"],
                        "product_phase": run.get("product_phase", 2),
                        "phase3_selected": is_phase3,
                        "phase2_baseline": is_best,
                        "selection_scope": scope,
                        "agent_state_values": meta.get("agent_state_values", 409),
                        "phase3_study_id": run.get("phase3_study_id"),
                        "candidate": run.get("candidate"),
                        "actions": record["actions"],
                        "preset": run["preset"],
                        "status": state,
                        "checkpoint_sha256": meta["checkpoint_sha256"],
                        "feature_version": meta["feature_version"],
                        "shaping": run.get("shaping", "none"),
                        "q_stability": (
                            run.get("q_stability", "off")
                            if name == "q-learning"
                            else None
                        ),
                        "training_case_hash": run.get("case_hash"),
                        "learning_evidence": record.get("learning_evidence", {}),
                        "label": f"{scope} | {state} | {name.upper()} · {record['actions']} actions · seed {run['seed']} · {run['run_id']}",
                    }
                )
        except (OSError, ValueError, KeyError):
            continue
    return sorted(items, key=lambda i: i["status"] != "Best on development")


def load_model(item):
    """Load the original policy class and reject a corrupt or incompatible save.

    These module/class paths deliberately stay stable: SB3 checkpoints refer to
    them when restoring their serialized neural architectures.
    """
    if item.get("arm") == "public-vote-v1":
        from .ensemble import load_ensemble

        return load_ensemble(item)
    path = Path(item["path"])
    if not path.is_file():
        raise ValueError(
            "The checkpoint is missing. Refresh the registry or train a new model."
        )
    try:
        config = configuration()
        if item["arm"] == "product-v3":
            return phase3_features.load_policy(path, config)
        if item["arm"] == "product-v2":
            return product_features.load_policy(path, config)
        return load_scaled_policy(path, config, item["arm"])
    except Exception as exc:
        raise ValueError(
            "This checkpoint is incompatible or corrupt. Select a native five-action model or train a new one. Details: "
            + str(exc)
        ) from exc


def newest_by_algorithm(items=None):
    result = {}
    for item in items if items is not None else registry():
        result.setdefault(item["algorithm"], item)
    return result


def preferred_by_algorithm(items=None):
    """Keep the Phase2 API default; the Phase3 UI uses phase3_selected explicitly."""
    return newest_by_algorithm(items)


def phase3_selected(root=None):
    """Resolve the four frozen development-selected IDs without selecting again."""
    root = Path(root or RUNS)
    file = root / "phase3-selection.json"
    if not file.is_file():
        return {}
    chosen = read_json(file)["selected"]
    available = {i["id"]: i for i in registry(root)}
    result = {}
    for a, record in chosen.items():
        item = available.get(record["id"])
        if item is None or item["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise ValueError(
                "Phase3 selection points to missing or incompatible checkpoint"
            )
        result[a] = item
    return result
