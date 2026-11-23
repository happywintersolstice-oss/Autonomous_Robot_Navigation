"""Explicit inference-only voting of the four independently trained RL policies.

No planner, obstacle identity, mask, confidence rescaling or fitted vote weights.
Every member sees the same public observation and actual executed-action history.
"""

from copy import deepcopy
import numpy as np

from .core import digest
from .partial_agents import ALGORITHMS
from .partial_contract import configuration, signature
from .partial_observation import validate
from .phase3_features import PublicHistory

VERSION = "four-model-public-majority-v1"
ARM = "public-vote-v1"
LABEL = "Four-model vote (ensemble; no additional training)"


def descriptor(members):
    """Describe the existing weights being combined, without creating fake weights."""
    if set(members) != set(ALGORITHMS):
        raise ValueError(
            "The ensemble requires one checkpoint from each of the four RL families"
        )
    if any(
        item.get("algorithm") != name or item.get("arm") == ARM
        for name, item in members.items()
    ):
        raise ValueError("Ensemble members must be matching native RL checkpoints")
    provenance = {
        name: {
            key: members[name][key]
            for key in ("id", "checkpoint_sha256", "feature_version", "actions")
        }
        for name in ALGORITHMS
    }
    identity = digest({"version": VERSION, "members": provenance})
    return {
        "id": "ensemble/" + identity[:20],
        "algorithm": "four-model-vote",
        "arm": ARM,
        "kind": "inference-only ensemble of four RL checkpoints",
        "label": LABEL,
        "feature_version": VERSION,
        "checkpoint_sha256": None,
        "member_set_sha256": identity,
        "member_provenance": provenance,
        "members": deepcopy(members),
        "actions": 0,
        "constituent_training_actions": sum(
            item["actions"] for item in members.values()
        ),
        "seed": None,
        "tie_rule": "Most votes, then lowest action index among tied actions (0 Up through4 Wait)",
    }


class VotingPolicy:
    """One unweighted vote per family, with deterministic five-action tie-breaking."""

    algorithm = "four-model-vote"
    training_seed = None
    checkpoint_sha256 = None  # There is no newly trained ensemble weight file.

    def __init__(self, members, description):
        if set(members) != set(ALGORITHMS):
            raise ValueError("All four independent policies are required")
        self.members = members
        self.policy_id = description["id"]
        self.signature = {**signature(configuration()), "ensemble_version": VERSION}
        self.description = description
        self.reset()

    def reset(self):
        self.history = PublicHistory()
        self.vote_trace = []
        for member in self.members.values():
            if hasattr(member, "reset"):
                member.reset()

    def state_fingerprint(self):
        return digest(
            {name: member.state_fingerprint() for name, member in self.members.items()}
        )

    def act(self, observation):
        public = validate(observation)
        proposals = {}
        for name in ALGORITHMS:
            member = self.members[name]
            if hasattr(member, "history"):
                # A member's proposed action is not necessarily executed. Reset
                # its local state to the actual shared decision history first.
                member.history = deepcopy(self.history)
            action = member.act(public.copy())
            if (
                isinstance(action, (bool, np.bool_))
                or not isinstance(action, (int, np.integer))
                or int(action) not in range(5)
            ):
                raise ValueError(
                    "A voting member returned an invalid five-action decision"
                )
            proposals[name] = int(action)
        counts = np.bincount(list(proposals.values()), minlength=5)
        chosen = int(np.flatnonzero(counts == counts.max())[0])
        self.history.remember(public, chosen)
        for member in self.members.values():
            if hasattr(member, "history"):
                # Synchronize to the chosen action, not the member's own vote.
                member.history = deepcopy(self.history)
        self.vote_trace.append(
            {
                "proposals": proposals,
                "counts": counts.tolist(),
                "chosen": chosen,
                "tied": int(np.count_nonzero(counts == counts.max())) > 1,
            }
        )
        return chosen


def load_ensemble(item):
    from .app_registry import load_model

    expected = descriptor(item["members"])
    if (
        item.get("arm") != ARM
        or item.get("id") != expected["id"]
        or item.get("member_set_sha256") != expected["member_set_sha256"]
    ):
        raise ValueError("Ensemble descriptor or member provenance changed")
    policies = {}
    for name, member in item["members"].items():
        policy = load_model(member)
        if policy.checkpoint_sha256 != member["checkpoint_sha256"]:
            raise ValueError(
                "Ensemble member weights differ from the registered descriptor"
            )
        policies[name] = policy
    return VotingPolicy(policies, expected)
