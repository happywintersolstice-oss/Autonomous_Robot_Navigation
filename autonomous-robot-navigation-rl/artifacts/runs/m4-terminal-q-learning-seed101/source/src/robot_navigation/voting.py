"""Frozen majority voting across three independently trained tabular policies."""
import hashlib
import os
from pathlib import Path

import numpy as np

from .core import read_json, write_json


class VotingPolicy:
    algorithm = "voting"

    def __init__(self, task, members, seeds, policy_id="voting"):
        if len(members) != 3 or len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("Voting requires three distinct predeclared training seeds")
        algorithms = {m.algorithm for m in members}
        if len(algorithms) != 1 or not algorithms <= {"q-learning", "sarsa"}:
            raise ValueError("Voting members must belong to one shortlisted tabular family")
        for member in members:
            task.check(member.task.signature)
        self.task, self.members, self.seeds = task, tuple(members), tuple(seeds)
        self.family = members[0].algorithm
        self.policy_id = policy_id

    def decision(self, observation):
        proposals = [m.act(observation) for m in self.members]
        counts = np.bincount(proposals, minlength=4)
        tied = int(counts.max()) == 1
        action = proposals[0] if tied else int(np.argmax(counts))
        return {"action": int(action), "member_actions": proposals, "vote_counts": counts.tolist(),
                "three_way_tie": tied, "priority_seed": self.seeds[0], "member_seeds": list(self.seeds)}

    def act(self, observation):
        return self.decision(observation)["action"]

    def state_fingerprint(self):
        hasher = hashlib.sha256()
        for member in self.members:
            hasher.update(member.q.tobytes())
        return hasher.hexdigest()

    def save(self, path, member_paths):
        path = Path(path)
        if len(member_paths) != 3:
            raise ValueError("Three member checkpoints are required")
        references = []
        for member, seed, member_path in zip(self.members, self.seeds, member_paths):
            member_path = Path(member_path).resolve()
            metadata = read_json(member_path.with_suffix(".json"))
            if metadata["config"]["seed"] != seed or metadata["policy_id"] != member.policy_id:
                raise ValueError("Voting member seed/identity mismatch")
            references.append({"path": os.path.relpath(member_path, path.parent).replace("\\", "/"),
                "seed": seed, "policy_id": member.policy_id,
                "sha256": hashlib.sha256(member_path.read_bytes()).hexdigest()})
        write_json(path, {"algorithm": "voting", "family": self.family, "policy_id": self.policy_id,
            "signature": self.task.signature, "kind": "frozen inference-only voting; no training",
            "rule": "majority; three-way tie uses first member", "members": references})

    @classmethod
    def load(cls, path, task):
        from .agents import load_policy
        path = Path(path)
        metadata = read_json(path)
        task.check(metadata["signature"])
        if metadata.get("algorithm") != "voting" or metadata.get("rule") != "majority; three-way tie uses first member":
            raise ValueError("Unsupported voting descriptor/rule")
        members, seeds = [], []
        for reference in metadata["members"]:
            member_path = (path.parent / reference["path"]).resolve()
            if hashlib.sha256(member_path.read_bytes()).hexdigest() != reference["sha256"]:
                raise ValueError("Voting member checksum mismatch")
            member_metadata = read_json(member_path.with_suffix(".json"))
            if (member_metadata["algorithm"] not in ("q-learning", "sarsa") or
                    member_metadata["config"]["seed"] != reference["seed"] or
                    member_metadata["policy_id"] != reference["policy_id"]):
                raise ValueError("Voting member identity mismatch")
            members.append(load_policy(member_path, task))
            seeds.append(reference["seed"])
        result = cls(task, members, seeds, metadata["policy_id"])
        if metadata["family"] != result.family:
            raise ValueError("Voting family mismatch")
        return result
