"""One bounded registered Phase3 study; holdout runs only after a locked selection."""

import argparse, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
from .core import ROOT, read_json, write_json, digest
from .app_registry import registry, RUNS
from .phase3_data import train_dev, verify_frozen, TEST_PATH, TRAIN_PATH
from .phase3_training import train
from .phase3_evaluation import score_candidate, rank, export_comparison
from .release import sha

ALGORITHMS = ("q-learning", "sarsa", "dqn", "ppo")


def baseline_items():
    """Resolve original Phase2 selected IDs and reject altered checkpoint hashes."""
    selection = read_json(RUNS / "selection.json")["selected"]
    items = {i["id"]: i for i in registry()}
    out = {}
    for algorithm, record in selection.items():
        item = items[record["id"]]
        if item["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise ValueError("Phase2 baseline changed")
        out[algorithm] = item
    return out


def run(study):
    """Execute the registered fresh jobs and lock selection using development only."""
    study = Path(study).resolve()
    protocol = read_json(study / "protocol.json")
    if (RUNS / ("holdout-use-" + protocol["holdout_version"] + ".json")).exists():
        raise ValueError(
            "This product holdout has already been used; register a new holdout version before another study"
        )
    if (study / "training-started.json").exists():
        raise FileExistsError(
            "Study training already started; no silent retry or exact-resume claim"
        )
    if sha(TRAIN_PATH) != protocol["train_dev_file_sha256"]:
        raise ValueError("Frozen train/development inputs changed")
    # Check only the small new manifest's checksum; do not evaluate/read holdout cases.
    if sha(TEST_PATH) != protocol["holdout_file_sha256"]:
        raise ValueError("New product test file changed")
    write_json(
        study / "training-started.json",
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_commit": "4dc21aec5c095656934ce128b99a83cb1a3eeb1b",
            "heldout_policy_evaluations": 0,
        },
    )
    dev = train_dev()["development"]
    baselines = baseline_items()
    results = {a: [] for a in ALGORITHMS}
    attempts = []
    used = 0
    for a in ALGORITHMS:
        result = score_candidate(baselines[a], dev)
        result["source"] = "Phase2 baseline"
        results[a].append(result)
        write_json(study / f"baseline-{a}-development.json", result)
        print(f'BASELINE DEVELOPMENT {a} {result["successes"]}/{len(dev)}', flush=True)
    for candidate in protocol["candidate_set"]:
        for a in ALGORITHMS:
            budget = candidate["actions_per_family"]
            if used + budget > protocol["total_substantive_action_cap"]:
                raise ValueError("Registered cumulative training budget exceeded")
            used += budget
            attempt = {
                "algorithm": a,
                "candidate": candidate["name"],
                "requested_actions": budget,
                "seed": candidate["seed"],
                "objective": candidate["objective"],
                "state": "allocated",
            }
            attempts.append(attempt)
            write_json(
                study / "training-ledger.json",
                {
                    "allocated_actions": used,
                    "cap": protocol["total_substantive_action_cap"],
                    "attempts": attempts,
                    "heldout_policy_evaluations": 0,
                },
            )
            print(f'TRAIN {a} {candidate["name"]} {budget}', flush=True)
            try:
                receipt = train(
                    "standard",
                    [a],
                    budget,
                    candidate["seed"],
                    objective=candidate["objective"],
                    study_id=study.name,
                    candidate=candidate["name"],
                )
                attempt.update(state="complete", receipt=receipt)
            except Exception as exc:
                attempt.update(state="failed", error=repr(exc))
                write_json(
                    study / "training-ledger.json",
                    {
                        "allocated_actions": used,
                        "cap": protocol["total_substantive_action_cap"],
                        "attempts": attempts,
                        "heldout_policy_evaluations": 0,
                    },
                )
                print("TRAINING FAILED " + repr(exc), flush=True)
                continue
            item = next(i for i in registry() if i["id"] == receipt["run_id"] + "/" + a)
            try:
                result = score_candidate(item, dev)
            except Exception as exc:
                attempt.update(
                    development_validation="failed", evaluation_error=repr(exc)
                )
                write_json(
                    study / "training-ledger.json",
                    {
                        "allocated_actions": used,
                        "attempts": attempts,
                        "heldout_policy_evaluations": 0,
                    },
                )
                raise RuntimeError(
                    "Development gate failed; stop without consulting holdout"
                ) from exc
            result.update(source="Phase3 candidate", candidate=candidate["name"])
            results[a].append(result)
            write_json(
                study / (receipt["run_id"] + "-" + a + "-development.json"), result
            )
            print(
                f'DEVELOPMENT {a} {candidate["name"]} {result["successes"]}/{len(dev)}',
                flush=True,
            )
            write_json(
                study / "training-ledger.json",
                {
                    "allocated_actions": used,
                    "cap": protocol["total_substantive_action_cap"],
                    "attempts": attempts,
                    "heldout_policy_evaluations": 0,
                },
            )
            write_json(study / "development-scores.json", results)
    chosen = {}
    all_items = {i["id"]: i for i in registry()}
    for a in ALGORITHMS:
        chosen[a] = all_items[max(results[a], key=rank)["checkpoint_id"]]
    selection = {
        "schema": "phase3-development-selection-v1",
        "selected": {
            a: {
                "id": i["id"],
                "checkpoint_sha256": i["checkpoint_sha256"],
                "feature_version": i["feature_version"],
                "actions": i["actions"],
                "retains_phase2": i["id"] == baselines[a]["id"],
            }
            for a, i in chosen.items()
        },
        "study": study.relative_to(RUNS).as_posix(),
        "criterion": protocol["selection"],
        "development_case_hashes": [digest(c) for c in dev],
        "holdout_outcomes_used": False,
        "heldout_policy_evaluations_at_selection": 0,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (study / "selection.json").open("x", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    write_json(
        study / "selection-lock.json",
        {"sha256": sha(study / "selection.json"), "heldout_policy_evaluations": 0},
    )
    choices = (
        [("Phase3 " + a, chosen[a], "RL selected on development") for a in ALGORITHMS]
        + [("Phase2 " + a, baselines[a], "RL Phase2 baseline") for a in ALGORITHMS]
        + [
            ("Observed-Map Replanner", "Observed-Map Replanner", "Non-RL reference"),
            ("Random reference", "Random reference", "Non-RL reference"),
        ]
    )
    # Persist development replays before final assessment; all selected models frozen.
    dev_result = export_comparison(
        choices,
        dev,
        study / "development-comparison",
        "development",
        {"selection_sha256": sha(study / "selection.json"), "test_evaluations_used": 0},
    )
    write_json(
        RUNS / "phase3-selection.json",
        {
            **selection,
            "development_comparison": str(Path(dev_result["folder"]).relative_to(RUNS)),
        },
    )
    write_json(
        study / "ready-for-final.json",
        {
            "status": "selected-and-development-export-complete",
            "selection_sha256": sha(study / "selection.json"),
            "training_actions_allocated": used,
            "heldout_policy_evaluations": 0,
        },
    )
    print("READY FOR SINGLE FINAL " + str(study), flush=True)
    return study


def final(study):
    """Perform the single guarded final run after selection is locked.

    Create both use markers before reading test cases or calling any policy;
    an interrupted final run must not silently be restarted as an unseen test."""
    study = Path(study).resolve()
    protocol = read_json(study / "protocol.json")
    selection = read_json(study / "selection.json")
    lock = read_json(study / "selection-lock.json")
    if (
        sha(study / "selection.json") != lock["sha256"]
        or selection["holdout_outcomes_used"]
    ):
        raise ValueError("Selection lock invalid")
    if not (study / "ready-for-final.json").is_file():
        raise ValueError("Training/development phase incomplete")
    if (study / "heldout-evaluation-started.json").exists():
        raise FileExistsError(
            "Single final product evaluation already started. Never rerun this suite after outcomes are known."
        )
    # Create the one-run marker BEFORE reading cases or invoking any policy.
    marker = {
        "run_count": 1,
        "suite_version": protocol["holdout_version"],
        "suite_hash": protocol["holdout_suite_hash"],
        "selection_sha256": lock["sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
        ).strip(),
        "code_sha256": {
            name: sha(ROOT / "src/robot_navigation" / name)
            for name in (
                "phase3_features.py",
                "phase3_training.py",
                "phase3_evaluation.py",
                "phase3_study.py",
            )
        },
        "no_training_after_final": True,
    }
    with (study / "heldout-evaluation-started.json").open("x", encoding="utf-8") as f:
        json.dump(marker, f, indent=2)
    global_use = RUNS / ("holdout-use-" + protocol["holdout_version"] + ".json")
    with global_use.open("x", encoding="utf-8") as f:
        json.dump(
            {
                **marker,
                "study": study.relative_to(RUNS).as_posix(),
                "purpose": "One final assessment per product holdout version; never remove to reuse outcomes as unseen",
            },
            f,
            indent=2,
        )
    suite = verify_frozen()
    if suite["suite_hash"] != protocol["holdout_suite_hash"]:
        raise ValueError("Holdout version mismatch")
    available = {i["id"]: i for i in registry()}
    chosen = {a: available[x["id"]] for a, x in selection["selected"].items()}
    baselines = baseline_items()
    for a, i in chosen.items():
        if i["checkpoint_sha256"] != selection["selected"][a]["checkpoint_sha256"]:
            raise ValueError("Locked checkpoint changed")
    choices = (
        [("Phase3 " + a, chosen[a], "RL selected on development") for a in ALGORITHMS]
        + [("Phase2 " + a, baselines[a], "RL Phase2 baseline") for a in ALGORITHMS]
        + [
            ("Observed-Map Replanner", "Observed-Map Replanner", "Non-RL reference"),
            ("Random reference", "Random reference", "Non-RL reference"),
        ]
    )
    result = export_comparison(
        choices,
        suite["cases"],
        study / "heldout-comparison",
        "product-heldout",
        {
            "holdout_version": suite["version"],
            "holdout_suite_hash": suite["suite_hash"],
            "holdout_file_sha256": sha(TEST_PATH),
            "final_evaluation_run_count": 1,
            "selection_sha256": lock["sha256"],
            "evaluation_started_metadata": marker,
            "selection_data_scope": "Training plus separate development only; no heldout outcomes used",
        },
        with_optimum=True,
    )
    write_json(
        study / "heldout-evaluation-complete.json",
        {
            "status": "complete",
            "run_count": 1,
            "suite_hash": suite["suite_hash"],
            "case_rows": result["manifest"]["case_count"],
            "unique_episodes": result["manifest"]["unique_executed_episodes"],
            "unique_actions": result["manifest"]["unique_executed_actions"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "no_post_test_training": True,
        },
    )
    write_json(
        RUNS / "phase3-publication.json",
        {
            "schema": "phase3-product-publication-v1",
            "study": study.relative_to(RUNS).as_posix(),
            "development_comparison": (study / "development-comparison")
            .relative_to(RUNS)
            .as_posix(),
            "heldout_comparison": (study / "heldout-comparison")
            .relative_to(RUNS)
            .as_posix(),
            "holdout_suite_hash": suite["suite_hash"],
            "holdout_version": suite["version"],
            "final_evaluation_run_count": 1,
            "selection_sha256": lock["sha256"],
        },
    )
    print("FINAL COMPLETE " + str(study), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--run", action="store_true")
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    if a.run:
        run(a.study)
    elif a.final:
        final(a.study)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
