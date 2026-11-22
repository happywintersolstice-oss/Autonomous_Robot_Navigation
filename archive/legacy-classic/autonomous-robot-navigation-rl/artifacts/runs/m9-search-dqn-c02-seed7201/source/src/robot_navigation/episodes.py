"""Episode records, pure metrics, replay validation and synchronized frame state."""
import numpy as np

from .core import ACTION_NAMES, NavigationEnv, validate_scenario


def rollout(env, policy, scenario, seed=0, mode="validation", *, allow_final=False):
    if mode not in ("validation", "training", "custom demonstration") and not (mode == 'final evaluation' and allow_final):
        raise ValueError("Final-test rollouts are disabled in this milestone")
    observation, _ = env.reset(seed=seed, options={"scenario": scenario})
    episode = {"signature": env.task.signature, "scenario": scenario, "seed": seed,
               "policy_id": policy.policy_id, "mode": mode,
               "initial_observation": observation.tolist(), "transitions": []}
    while True:
        decision = None
        if hasattr(policy, "decision"):
            decision = policy.decision(observation)
            action, scores = decision["action"], None
        elif hasattr(policy, "action_and_scores"):
            action, scores = policy.action_and_scores(observation)
            scores = scores.tolist()
        else:
            scores = policy.scores(observation).tolist() if hasattr(policy, "scores") else None
            action = policy.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        episode["transitions"].append({"observation": observation.tolist(), "action": action,
            "reward": reward, "next_observation": next_observation.tolist(),
            "terminated": terminated, "truncated": truncated, "info": info, "scores": scores})
        if decision is not None:
            episode["transitions"][-1]["decision"] = decision
        observation = next_observation
        if terminated or truncated:
            break
    episode["metrics"] = episode_metrics(episode)
    return episode


def episode_metrics(episode):
    transitions = episode["transitions"]
    if not transitions or not (transitions[-1]["terminated"] or transitions[-1]["truncated"]):
        raise ValueError("Metrics require a completed episode")
    outcome = transitions[-1]["info"]["outcome"]
    length, shortest = len(transitions), episode["scenario"]["shortest_length"]
    return {"outcome": outcome, "steps": length,
            "return": float(sum(t["reward"] for t in transitions)),
            "successful_route_length": length if outcome == "success" else None,
            "spl": shortest / max(shortest, length) if outcome == "success" else 0.0}


def aggregate(metrics):
    if not metrics:
        raise ValueError("Cannot aggregate zero episodes")
    n = len(metrics)
    counts = {k: sum(m["outcome"] == k for m in metrics) for k in ("success", "collision", "timeout")}
    if sum(counts.values()) != n:
        raise ValueError("Episode outcomes must be mutually exclusive and exhaustive")
    successful = [m["steps"] for m in metrics if m["outcome"] == "success"]
    return {"episodes": n, "counts": counts, **{f"{k}_rate": v / n for k, v in counts.items()},
            "spl": float(np.mean([m["spl"] for m in metrics])),
            "mean_return": float(np.mean([m["return"] for m in metrics])),
            "mean_successful_route_length": float(np.mean(successful)) if successful else None,
            "successful_route_count": len(successful), "interactions": sum(m["steps"] for m in metrics)}


def validate_replay(task, episode, *, allow_final=False):
    task.check(episode["signature"])
    if episode["mode"] not in ("validation", "training", "custom demonstration") and not (episode['mode']=='final evaluation' and allow_final):
        raise ValueError("Sealed final-test replays are unavailable")
    scenario = episode["scenario"]
    validate_scenario(task.grid, scenario)
    env = NavigationEnv(task, [scenario])
    observation, _ = env.reset(seed=episode["seed"], options={"scenario": scenario})
    if observation.tolist() != episode["initial_observation"]:
        raise ValueError("Replay initial state mismatch")
    for i, transition in enumerate(episode["transitions"]):
        if observation.tolist() != transition["observation"]:
            raise ValueError("Replay before-state mismatch")
        observation, reward, terminated, truncated, info = env.step(transition["action"])
        if (observation.tolist() != transition["next_observation"] or
                reward != transition["reward"] or terminated != transition["terminated"] or
                truncated != transition["truncated"] or info != transition["info"]):
            raise ValueError(f"Replay transition mismatch at action {i + 1}")
    if episode_metrics(episode) != episode["metrics"]:
        raise ValueError("Replay metrics mismatch")
    return episode


def frame(episode, index):
    index = max(0, min(int(index), len(episode["transitions"])))
    prefix = episode["transitions"][:index]
    last = prefix[-1] if prefix else None
    trail = [episode["initial_observation"][:2]] + [t["next_observation"][:2] for t in prefix]
    return {"index": index, "position": trail[-1], "trail": trail,
            "action": ACTION_NAMES[last["action"]] if last else "—",
            "step_reward": last["reward"] if last else 0.0,
            "return": sum(t["reward"] for t in prefix),
            "status": last["info"]["outcome"] if last else "ready",
            "attempted_position": last["info"]["attempted_position"] if last else None,
            "revisited_cells": len(trail) - len(set(map(tuple, trail)))}


class Playback:
    """UI-independent playback state: no policy calls and no environment changes."""
    def __init__(self, episode):
        self.episode, self.index, self.playing = episode, 0, False

    def seek(self, index):
        self.index = max(0, min(int(index), len(self.episode["transitions"])))
        if self.index == len(self.episode["transitions"]):
            self.playing = False

    def play(self):
        if self.index < len(self.episode["transitions"]):
            self.playing = True

    def pause(self):
        self.playing = False

    def next(self):
        self.seek(self.index + 1)

    def reset(self):
        self.index, self.playing = 0, False

    def tick(self):
        if self.playing:
            self.next()

    @property
    def current(self):
        return frame(self.episode, self.index)


class ComparisonPlayback:
    """One action-index clock; finished agents remain at their terminal frames."""
    def __init__(self, episodes):
        if len(episodes) not in (2, 4):
            raise ValueError('Two or four episodes are required')
        first = episodes[0]
        if any(first[k] != episode[k] for episode in episodes[1:]
               for k in ('signature', 'scenario', 'mode', 'seed')):
            raise ValueError('Comparison requires matching task, scenario and evaluation mode')
        self.episodes = tuple(episodes)
        self.index, self.playing = 0, False
        self.length = max(len(e['transitions']) for e in episodes)

    def seek(self, index):
        self.index = max(0, min(int(index), self.length))
        if self.index == self.length:
            self.playing = False

    def play(self):
        self.playing = self.index < self.length

    def pause(self):
        self.playing = False

    def next(self):
        self.seek(self.index + 1)

    def reset(self):
        self.index, self.playing = 0, False

    def tick(self):
        if self.playing:
            self.next()

    @property
    def current(self):
        return [frame(e, self.index) for e in self.episodes]
