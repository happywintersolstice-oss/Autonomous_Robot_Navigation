"""Registered bounded product training, matched shaping ablation and measured selection."""

import argparse, json, csv
from pathlib import Path
from .core import write_json, read_json, digest, ROOT
from .app_service import identity, compare
from .app_training import train
from .app_registry import RUNS, registry, load_model
from .product_curriculum import manifest as dataset_manifest, cases
from .partial_contract import configuration
from .partial_evaluation import record_episode
from .release import sha
from . import product_features

ALGORITHMS = ("q-learning", "sarsa", "dqn", "ppo")


def evaluate(item):
    policy = load_model(item)
    before = policy.state_fingerprint()
    rows = []
    for case in cases("development"):
        record = record_episode(case, policy, configuration(), 0)
        rows.append(
            {
                "case_id": case["id"],
                "case_hash": digest(case),
                "layout_hash": case["layout_hash"],
                "event_seed": case["event_seed"],
                "condition": case["product_kind"],
                **record["metrics"],
            }
        )
    assert policy.state_fingerprint() == before
    return {
        "checkpoint_id": item["id"],
        "checkpoint_sha256": item["checkpoint_sha256"],
        "actions": item["actions"],
        "shaping": item["shaping"],
        "cases": len(rows),
        "successes": sum(r["outcome"] == "success" for r in rows),
        "collisions": sum(r["outcome"] == "collision" for r in rows),
        "timeouts": sum(r["outcome"] == "timeout" for r in rows),
        "mean_reward": sum(r["return"] for r in rows) / len(rows),
        "waits": sum(r["wait_count"] for r in rows),
        "inference_parameters_unchanged": True,
        "rows": rows,
    }


def rank(result):
    return (
        result["successes"],
        result["mean_reward"],
        -result["waits"],
        -result["actions"],
    )


def run(resume=None):
    study = Path(resume).resolve() if resume else RUNS / "product-phase2" / identity()
    if resume and (study / "selection.json").is_file():
        raise ValueError(
            "Study already completed; refusing to repeat its training. Start a fresh study for new training."
        )
    dataset = dataset_manifest()
    recovery = {}
    prior = {}
    interrupted = []
    if resume:
        if (study / "evaluation-recovery.json").is_file():
            interrupted = read_json(study / "evaluation-recovery.json").get(
                "interrupted_runs", []
            )
        prior = read_json(study / "candidate-results.json")
        original = read_json(study / "datasets.json")
        assert original["training_hash"] == dataset["training_hash"]
        for old, new in zip(
            original["development_cases"], dataset["development_cases"]
        ):
            assert {k: v for k, v in old.items() if k != "split"} == {
                k: v for k, v in new.items() if k != "split"
            }
        recovery = {
            "reason": "Evaluator accepts validation/manual labels, not development. Only development split metadata corrected; layouts, goals, events, training data and official contracts unchanged. Completed native runs reused without retraining. No successful development score existed before this amendment.",
            "original_evaluation_failures": prior["failures"],
            "premature_q_learning_30000": "Completed before evaluation integration error was noticed. Eligible for primary selection only if measured 10000-action candidates trigger the registered escalation; otherwise reported as exploratory.",
            "partial_sarsa_30000": "Interrupted. No usable checkpoint; exact final transition count unavailable; most recent recorded progress retained. Its 30000-action budget slot is consumed and will not be repeated.",
        }
        for file in RUNS.glob("training/*/manifest.json"):
            m = read_json(file)
            if m.get("seed") == 82111 and m.get("status") == "running":
                progress = read_json(file.parent / "progress.json")
                interrupted.append(
                    {
                        "run_id": m["run_id"],
                        "last_recorded_actions": progress["completed_actions"],
                        "exact_final_actions": None,
                        "maximum_requested_actions": m["requested_actions_per_model"],
                    }
                )
                write_json(file.parent / "interruption.json", interrupted[-1])
                m.update(
                    status="cancelled",
                    error="Worker interrupted to correct development split metadata; no inference model published.",
                )
                write_json(file, m)
        recovery["interrupted_runs"] = interrupted
        write_json(study / "evaluation-recovery.json", recovery)
        write_json(study / "datasets-compatible-split.json", dataset)
        protocol = read_json(study / "protocol.json")
    else:
        study.mkdir(parents=True, exist_ok=False)
        write_json(study / "datasets.json", dataset)
        protocol = {
            "version": "product-phase2-registered-v1",
            "dataset_hash": digest(dataset),
            "feature_version": product_features.VERSION,
            "features": product_features.FEATURE_COUNT,
            "row_features": product_features.ROW,
            "features_description": "Shared learned action scorer; public goal offsets/distance bins, progress, memory occupancy/visits/age/block counts and rays; no action scores, paths, masks or teacher in encoding.",
            "matched_ablation": {
                "actions_per_family": 10000,
                "seed": 82111,
                "shaping": ["none", "progress-v1"],
            },
            "escalation": {
                "budgets": [30000, 50000],
                "trigger": "Best new development success below 0.70",
                "seed": 82111,
                "shaping": "progress-v1",
            },
            "maximum_substantive_actions_all_families": 400000,
            "maximum_actions_per_family_per_run": 50000,
            "worker_wall_cap_seconds": 3600,
            "selection": "Highest success count, then original mean return, then fewer Wait actions, then lower budget; first candidate breaks exact ties. Development selection is optimistic, not unseen accuracy.",
            "training_reward": "Official + .2*(old Manhattan-new Manhattan) when progress-v1 selected; not claimed policy invariant under gamma=.99. Official evaluation reward unchanged.",
        }
        write_json(study / "protocol.json", protocol)
    print("REGISTERED " + str(study), flush=True)
    results = {a: [] for a in ALGORITHMS}
    failures = list(prior.get("failures", []))
    chosen = {}
    used = list(prior.get("training_receipts", []))
    exploratory = []
    for budget, shaping in [
        (10000, "none"),
        (10000, "progress-v1"),
        (30000, "progress-v1"),
        (50000, "progress-v1"),
    ]:
        for algorithm in ALGORITHMS:
            existing = (
                next(
                    (
                        i
                        for i in registry()
                        if i["seed"] == 82111
                        and i["algorithm"] == algorithm
                        and i["actions"] == budget
                        and i["shaping"] == shaping
                    ),
                    None,
                )
                if resume
                else None
            )
            gated = budget > 10000 and (
                not results[algorithm]
                or max(r["successes"] / r["cases"] for r in results[algorithm]) >= 0.70
            )
            if gated:
                if existing:
                    result = evaluate(existing)
                    result["selection_eligible"] = False
                    exploratory.append(result)
                    write_json(
                        study
                        / (existing["id"].replace("/", "-") + "-exploratory.json"),
                        result,
                    )
                continue
            if resume and algorithm == "sarsa" and budget == 30000 and interrupted:
                print("SKIP consumed interrupted SARSA 30000-action slot", flush=True)
                continue
            if existing:
                item = existing
                print("REUSE " + item["id"], flush=True)
            else:
                print(
                    f"TRAIN {algorithm} actions={budget} shaping={shaping}", flush=True
                )
                try:
                    receipt = train(
                        "standard",
                        [algorithm],
                        budget,
                        82111,
                        shaping=shaping,
                        q_stability=False,
                    )
                    used.append(receipt)
                    item = next(
                        i
                        for i in registry()
                        if i["id"] == receipt["run_id"] + "/" + algorithm
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "algorithm": algorithm,
                            "actions_requested": budget,
                            "shaping": shaping,
                            "stage": "training",
                            "error": repr(exc),
                        }
                    )
                    write_json(
                        study / "candidate-results-recovered.json",
                        {
                            "results": results,
                            "failures": failures,
                            "training_receipts": used,
                        },
                    )
                    continue
            try:
                result = evaluate(item)
            except Exception as exc:
                failures.append(
                    {"algorithm": algorithm, "stage": "evaluation", "error": repr(exc)}
                )
                write_json(
                    study / "candidate-results-recovered.json",
                    {
                        "results": results,
                        "failures": failures,
                        "training_receipts": used,
                    },
                )
                raise RuntimeError(
                    "Evaluation gate failed; stop before further training"
                ) from exc
            result["selection_eligible"] = True
            results[algorithm].append(result)
            write_json(
                study / (item["id"].replace("/", "-") + "-development.json"), result
            )
            print(
                f'MEASURED {algorithm} {result["successes"]}/24 successes; collisions={result["collisions"]} timeouts={result["timeouts"]}',
                flush=True,
            )
            write_json(
                study / "candidate-results-recovered.json",
                {
                    "results": results,
                    "failures": failures,
                    "training_receipts": used,
                    "exploratory": exploratory,
                    "recovery": recovery,
                },
            )
    ledger = {
        "results": results,
        "failures": failures,
        "training_receipts": used,
        "exploratory": exploratory,
        "recovery": recovery,
    }
    return finalize(study, ledger, allow_training=True)


def finalize(study, ledger=None, allow_training=False):
    study = Path(study).resolve()
    ledger = ledger or read_json(study / "candidate-results-recovered.json")
    results = ledger["results"]
    dataset = dataset_manifest()
    protocol = read_json(study / "protocol.json")
    allowed = {r["run_id"] for r in ledger.get("training_receipts", [])}
    amendment = (
        read_json(study / "q-stability-amendment.json")
        if (study / "q-stability-amendment.json").is_file()
        else {}
    )
    if amendment.get("completed_run_id"):
        allowed.add(amendment["completed_run_id"])
        if not any(
            r["run_id"] == amendment["completed_run_id"]
            for r in ledger.get("training_receipts", [])
        ):
            ledger["training_receipts"].append(
                {
                    "run_id": amendment["completed_run_id"],
                    "actual_actions": 30000,
                    "status": "complete",
                    "purpose": "Registered stability amendment",
                }
            )
    q_results = results["q-learning"]
    stable = [
        i
        for i in registry()
        if i["id"].split("/")[0] in allowed
        and i["algorithm"] == "q-learning"
        and i["seed"] == 82111
        and i["actions"] == 30000
        and i.get("q_stability") == "target-v1"
        and i["shaping"] == "none"
    ]
    if stable and not any(r.get("q_stability") == "target-v1" for r in q_results):
        item = stable[0]
        result = evaluate(item)
        result.update(selection_eligible=True, q_stability="target-v1")
        q_results.append(result)
        write_json(study / (item["id"].replace("/", "-") + "-development.json"), result)
        print("STABILITY Q-learning " + str(result["successes"]) + "/24", flush=True)
    if (
        not stable
        and allow_training
        and max(r["successes"] / r["cases"] for r in q_results) < 0.70
    ):
        spent = (
            sum(r["actual_actions"] for r in ledger.get("training_receipts", []))
            + sum(
                f.get("actions_requested", 0)
                for f in ledger.get("failures", [])
                if f.get("stage") == "training"
            )
            + sum(
                x["maximum_requested_actions"]
                for x in ledger.get("recovery", {}).get("interrupted_runs", [])
            )
        )
        if spent + 30000 <= 400000:
            amendment = {
                "registered_before_training": True,
                "reason": "Persistent weak shared linear Q-learning; one bounded lagged-target/TD-clip stability candidate",
                "actions": 30000,
                "seed": 82111,
                "shaping": "none",
                "q_stability": "target-v1",
                "original_global_ceiling": 400000,
            }
            write_json(study / "q-stability-amendment.json", amendment)
            receipt = train(
                "standard",
                ["q-learning"],
                30000,
                82111,
                shaping="none",
                q_stability=True,
            )
            ledger["training_receipts"].append(receipt)
            amendment["completed_run_id"] = receipt["run_id"]
            write_json(study / "q-stability-amendment.json", amendment)
            return finalize(study, ledger, allow_training=False)
    chosen = {}
    for algorithm in ALGORITHMS:
        if not results[algorithm]:
            raise RuntimeError("No completed verified model for " + algorithm)
        best = max(results[algorithm], key=rank)
        chosen[algorithm] = next(
            i for i in registry() if i["id"] == best["checkpoint_id"]
        )
    write_json(study / "candidate-results-final.json", ledger)
    starters = {
        i["algorithm"]: i
        for i in registry()
        if i["id"].startswith("20261122T112025Z-a7301ff2/")
    }
    comparison = compare(chosen, include_random=True, starter_items=starters)
    selection = {
        "schema": "product-development-selection-v1",
        "selected": {
            a: {"id": i["id"], "checkpoint_sha256": i["checkpoint_sha256"]}
            for a, i in chosen.items()
        },
        "comparison": Path(comparison["folder"]).relative_to(RUNS).as_posix(),
        "study": study.relative_to(RUNS).as_posix(),
        "development_hash": dataset["development_hash"],
        "criterion": protocol["selection"],
    }
    write_json(study / "selection.json", selection)
    write_json(RUNS / "selection.json", selection)
    print("COMPLETED " + str(study), flush=True)
    print(
        json.dumps(
            {
                "selected": {
                    a: {
                        "id": i["id"],
                        "actions": i["actions"],
                        "checkpoint_sha256": i["checkpoint_sha256"],
                    }
                    for a, i in chosen.items()
                },
                "comparison": comparison["folder"],
            },
            indent=2,
        ),
        flush=True,
    )
    return study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument(
        "--resume",
        type=Path,
        help="Continue study orchestration; never resumes optimizer/buffer/RNG state",
    )
    p.add_argument("--finalize-study", type=Path)
    a = p.parse_args()
    if a.finalize_study:
        finalize(a.finalize_study)
    elif a.run:
        run(a.resume)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
