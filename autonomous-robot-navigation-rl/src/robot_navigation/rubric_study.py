"""One registered development-only ensemble study; never trains or opens holdout cases."""

import argparse
import csv
import gzip
import json
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

from .app_registry import RUNS, registry, phase3_selected, load_model
from .app_service import identity
from .core import ROOT, digest, read_json, write_json
from .ensemble import VERSION, LABEL, descriptor
from .phase3_data import train_dev, shortest
from .phase3_demo import NAMES, scenario
from .phase3_evaluation import record, case_row, summarize
from .phase3_features import PublicHistory
from .partial_agents import ALGORITHMS
from .partial_baselines import OnlinePlanner, RandomPolicy
from .release import sha


def register(output=None):
    """Freeze cases, member hashes, votes, limits and analysis before inference."""
    folder = Path(output or RUNS / "rubric-readiness" / identity())
    folder.mkdir(parents=True, exist_ok=False)
    members = phase3_selected()
    ensemble = descriptor(members)
    cases = train_dev()["development"]
    for name in NAMES:
        case = scenario(name, "Temporary")
        distance = len(shortest(case["rows"], case["start"], case["goal"])) - 1
        case.update(
            base_distance=distance,
            distance_band="long" if distance >= 8 else "short",
            product_scope="Previously available manual/development demonstration",
        )
        cases.append(case)
    write_json(
        folder / "panel.json",
        {
            "scope": "development-only; includes previously inspected cases",
            "cases": cases,
            "hash": digest(cases),
        },
    )
    protocol = {
        "version": "rubric-ensemble-development-v1",
        "study_id": folder.name,
        "scope": "Development and previously inspected demos only; not a new unseen or final assessment",
        "case_count": 36,
        "case_hash": digest(cases),
        "case_ids": [case["id"] for case in cases],
        "panel_file_sha256": sha(folder / "panel.json"),
        "ensemble_version": VERSION,
        "members": ensemble["member_provenance"],
        "member_set_sha256": ensemble["member_set_sha256"],
        "vote_rule": ensemble["tie_rule"],
        "fitted_vote_weights": False,
        "candidate_count": 1,
        "evaluation_seed": 0,
        "random_reference": "Single RNG stream seeded0 across ordered cases",
        "comparators": [
            *ALGORITHMS,
            LABEL,
            "Observed-Map Replanner",
            "Random reference",
        ],
        "training_action_cap": 0,
        "optimizer_step_cap": 0,
        "episode_cap": 252,
        "environment_action_cap": 50400,
        "hour_cap_seconds": 3600,
        "history_probe_cap_per_family": 256,
        "probe_selection": "Evenly spaced public decisions from each native policy's actual saved development episodes",
        "history_probe": "Compare actual public-history state against initial/empty history at the identical public409; direct internal prediction only, no environment execution or parameter updates. Counterfactual inputs can be out of distribution; this is sensitivity, not causal feature importance.",
        "uncertainty": "2000 family-cluster bootstrap replicates, seed94711, ensemble-minus-single success differences; descriptive on this development panel, not statistical superiority or new unseen accuracy",
        "selection": "No post-result retuning, no change to four selected checkpoint IDs or publication. Ensemble always remains optional and separately labeled, regardless of results.",
        "frozen_holdout_access": False,
    }
    write_json(folder / "protocol.json", protocol)
    print(str(folder), flush=True)
    return folder


class TimedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.seconds = 0.0

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def act(self, observation):
        started = time.perf_counter()
        action = self.policy.act(observation)
        self.seconds += time.perf_counter() - started
        return action


def public_history_sensitivity(policy, public_episodes, limit=256):
    """Probe only (observation, executed action); no scenarios/events/truth argument."""
    total = sum(len(episode) for episode in public_episodes)
    chosen = set(np.linspace(0, total - 1, min(limit, total), dtype=int).tolist())
    before = policy.state_fingerprint()
    index = changed = sampled = 0

    def action_for(state):
        if hasattr(policy, "weights"):
            return int(np.argmax(policy.scores(state)))
        return int(
            np.asarray(policy.model.predict(state, deterministic=True)[0]).item()
        )

    for episode in public_episodes:
        history = PublicHistory()
        for observation, action in episode:
            public = np.asarray(observation, dtype=np.float32)
            if index in chosen:
                actual = action_for(history.state(public))
                neutral = action_for(PublicHistory().state(public))
                changed += actual != neutral
                sampled += 1
            history.remember(public, int(action))
            index += 1
    if before != policy.state_fingerprint():
        raise ValueError("Offline probe altered learned weights")
    return {
        "probed_decisions": sampled,
        "different_greedy_actions": changed,
        "difference_fraction": changed / sampled if sampled else None,
        "environment_actions": 0,
        "optimizer_steps": 0,
        "interpretation": "Empty-history sensitivity on saved development observations; not causal importance, not a navigation outcome, and not unseen performance",
    }


def cluster_intervals(rows, reference_label=LABEL):
    rng = np.random.default_rng(94711)
    families = sorted({row["topology_family"] for row in rows})
    grouped = {
        label: {
            family: [
                row
                for row in rows
                if row["policy"] == label and row["topology_family"] == family
            ]
            for family in families
        }
        for label in [reference_label, *ALGORITHMS]
    }
    result = []
    for algorithm in ALGORITHMS:
        deltas = []
        for _ in range(2000):
            sample = rng.choice(families, len(families), replace=True)
            denominator = sum(len(grouped[algorithm][family]) for family in sample)
            difference = sum(
                sum(row["success"] for row in grouped[reference_label][family])
                - sum(row["success"] for row in grouped[algorithm][family])
                for family in sample
            )
            deltas.append(difference / denominator)
        result.append(
            {
                "ensemble_minus": algorithm,
                "low": float(np.quantile(deltas, 0.025)),
                "high": float(np.quantile(deltas, 0.975)),
                "unit": "success proportion difference",
                "clusters": len(families),
                "replicates": 2000,
            }
        )
    return result


def execute(folder):
    folder = Path(folder).resolve()
    protocol = read_json(folder / "protocol.json")
    if (folder / "started.json").exists():
        raise FileExistsError(
            "This registered development study already started; no silent retry"
        )
    if sha(folder / "panel.json") != protocol["panel_file_sha256"]:
        raise ValueError("Registered cases changed")
    cases = read_json(folder / "panel.json")["cases"]
    if (
        len(cases) != 36
        or digest(cases) != protocol["case_hash"]
        or any(case["split"] not in ("validation", "manual") for case in cases)
    ):
        raise ValueError("Only the registered development/manual cases may run")
    items = {item["id"]: item for item in registry()}
    members = {name: items[meta["id"]] for name, meta in protocol["members"].items()}
    combined = descriptor(members)
    if combined["member_set_sha256"] != protocol["member_set_sha256"]:
        raise ValueError("Registered ensemble members changed")
    with (folder / "started.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
                ).strip(),
                "protocol_sha256": sha(folder / "protocol.json"),
                "holdout_evaluations": 0,
            },
            stream,
            indent=2,
        )
    evidence = folder / "comparison"
    (evidence / "episodes").mkdir(parents=True, exist_ok=False)
    choices = [
        (name, item, "RL selected on earlier development")
        for name, item in members.items()
    ]
    choices += [
        (LABEL, combined, "Untrained voting ensemble of four RL checkpoints"),
        ("Observed-Map Replanner", None, "Non-RL reference"),
        ("Random reference", None, "Non-RL reference"),
    ]
    rows, failures, timing, probes = [], [], {}, {}
    began = time.perf_counter()
    executed_actions = episodes = 0
    try:
        for label, item, kind in choices:
            native = (
                load_model(item)
                if item is not None
                else (
                    OnlinePlanner()
                    if label == "Observed-Map Replanner"
                    else RandomPolicy(0)
                )
            )
            policy = TimedPolicy(native)
            public_episodes = []
            for case in cases:
                if (
                    episodes >= protocol["episode_cap"]
                    or time.perf_counter() - began > protocol["hour_cap_seconds"]
                ):
                    raise RuntimeError("Registered evaluation resource cap reached")
                # Every episode is capped by the unchanged environment at200.
                if executed_actions + 200 > protocol["environment_action_cap"]:
                    raise RuntimeError(
                        "Remaining registered environment budget is insufficient"
                    )
                replay = record(case, policy)
                if item is combined:
                    replay["ensemble_votes"] = list(native.vote_trace)
                    replay["ensemble_members"] = combined["member_provenance"]
                path = evidence / "episodes" / f"{episodes:04d}.json.gz"
                with gzip.open(path, "xt", encoding="utf-8") as stream:
                    json.dump(replay, stream, allow_nan=False)
                row = case_row(
                    label,
                    kind,
                    item or label,
                    replay,
                    path.relative_to(evidence).as_posix(),
                )
                rows.append(row)
                episodes += 1
                executed_actions += len(replay["transitions"])
                if label in ALGORITHMS:
                    public_episodes.append(
                        [(t["observation"], t["action"]) for t in replay["transitions"]]
                    )
                if row["outcome"] != "success":
                    failures.append(
                        {
                            "policy": label,
                            "case_id": case["id"],
                            "outcome": row["outcome"],
                            "replay": row["replay"],
                            "last8_public_decisions": [
                                {
                                    key: t[key]
                                    for key in (
                                        "observation",
                                        "action",
                                        "reward",
                                        "terminated",
                                        "truncated",
                                    )
                                }
                                for t in replay["transitions"][-8:]
                            ],
                        }
                    )
            count = sum(row["steps"] for row in rows if row["policy"] == label)
            timing[label] = {
                "inference_seconds": policy.seconds,
                "actions": count,
                "mean_inference_ms_per_action": policy.seconds * 1000 / count,
            }
            if label in ALGORITHMS:
                probes[label] = public_history_sensitivity(
                    native, public_episodes, protocol["history_probe_cap_per_family"]
                )
            print(
                f"RECORDED {label}: {sum(r['success'] for r in rows if r['policy'] == label)}/36",
                flush=True,
            )
        summary = summarize(rows)
        manifest = {
            "schema": "rubric-ensemble-development-v1",
            "scope": "development",
            "purpose": protocol["scope"],
            "case_count": len(rows),
            "panel_case_count": len(cases),
            "case_ids": protocol["case_ids"],
            "case_hashes": [digest(case) for case in cases],
            "summary": summary,
            "protocol_sha256": sha(folder / "protocol.json"),
            "ensemble": {
                key: value for key, value in combined.items() if key != "members"
            },
            "training_actions": 0,
            "optimizer_steps": 0,
            "heldout_evaluations": 0,
            "environment_actions": executed_actions,
            "inference_timing": timing,
            "public_history_sensitivity": probes,
            "cluster_intervals": cluster_intervals(rows),
            "source_hashes": {
                name: sha(ROOT / "src/robot_navigation" / name)
                for name in ("ensemble.py", "rubric_study.py")
            },
            "inclusion": "All32 original development cases and all4 independent manual demo presets; all outcomes retained",
        }
        write_json(evidence / "manifest.json", manifest)
        write_json(
            evidence / "failure-traces.json",
            {"failures": failures, "scope": "development"},
        )
        for filename, data in (("case-results.csv", rows), ("summary.csv", summary)):
            with (evidence / filename).open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        inventory = {
            path.relative_to(evidence).as_posix(): {
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }
            for path in evidence.rglob("*")
            if path.is_file()
        }
        write_json(evidence / "export-inventory.json", {"files": inventory})
        with zipfile.ZipFile(
            evidence / "results-and-replays.zip", "x", zipfile.ZIP_DEFLATED
        ) as archive:
            for name in [*inventory, "export-inventory.json"]:
                archive.write(evidence / name, name)
        write_json(
            folder / "complete.json",
            {
                "status": "complete",
                "episodes": episodes,
                "environment_actions": executed_actions,
                "training_actions": 0,
                "optimizer_steps": 0,
                "heldout_evaluations": 0,
                "elapsed_seconds": time.perf_counter() - began,
            },
        )
        write_json(
            RUNS / "rubric-publication.json",
            {
                "comparison": evidence.relative_to(RUNS).as_posix(),
                "study": folder.relative_to(RUNS).as_posix(),
                "scope": "development-only",
                "member_set_sha256": combined["member_set_sha256"],
            },
        )
        return manifest
    except BaseException as exc:
        write_json(
            folder / "failed.json",
            {
                "error": repr(exc),
                "episodes_completed": episodes,
                "environment_actions_completed": executed_actions,
                "no_retry": True,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--study", type=Path)
    args = parser.parse_args()
    if args.register:
        register()
    elif args.study:
        execute(args.study)
    else:
        parser.error("Choose --register or --study <registered-folder>")


if __name__ == "__main__":
    main()
