"""HospitalNavEnv - the Gymnasium environment every agent is trained and tested in.

Problem statement (short version for the report)
------------------------------------------------
A delivery robot carries medicine from the pharmacy to a ward bed. It only sees
the world through a 2D LiDAR (24 distance rays) and knows where the goal is
relative to itself (like a GPS/odometry estimate). It must reach the goal
quickly, never collide with walls, clutter or people, and keep a comfortable
social distance from people.

Robot model
-----------
Differential drive (like a TurtleBot): state (x, y, heading), commands are
linear velocity v (m/s) and angular velocity w (rad/s), integrated every dt.

Action spaces (config `action_mode`)
-----------------------------------
discrete   5 actions: forward, forward-left, forward-right, rotate-left, rotate-right
           (Q-learning, SARSA, DQN family)
continuous Box[-1,1]^2 -> (v, w)  (SAC, TD3, PPO/A2C)

Observation (all in [-1, 1], float32)
-------------------------------------
[0 : n_rays]      LiDAR distances / max_range   (dropped ray = -1 sentinel, see wrappers.py)
[n_rays]          distance to goal / dist_norm (clipped at 1)
[n_rays+1 : +3]   sin, cos of goal bearing relative to heading
[n_rays+3 : +5]   last v / v_max, last w / w_max
(+2 if guidance)  sin, cos of bearing to the A* look-ahead waypoint (hybrid agent)

Reward (each term is logged in info["reward_terms"] for analysis)
-----------------------------------------------------------------
progress   k_progress * (previous distance-to-go - new distance-to-go)
           distance-to-go is geodesic (walks around walls) or euclidean (ablation)
goal       +r_goal once the robot is within goal_tolerance
collision  r_collision (episode ends)
time       r_time every step (encourages short paths)
social     -w_social * (social_radius - d) / social_radius when a person is closer
           than social_radius (the "hospital etiquette" constraint)
clearance  -w_clearance when the nearest LiDAR return is closer than clearance
"""

from __future__ import annotations

import math
from copy import deepcopy

import gymnasium as gym
import numpy as np

from .humans import Human
from .maps import generate_layout, map_seed
from .planning import descend

DEFAULT_CONFIG = {
    # world
    "map_size": 10.0,
    "resolution": 0.25,
    "difficulty": 1,
    "n_obstacles": None,          # None = use difficulty preset
    "n_humans": None,
    "n_walls": None,
    "train_map_pool": 500,        # number of distinct training maps
    "split": "train",
    # robot
    "robot_radius": 0.2,
    "dt": 0.2,
    "v_max": 0.6,
    "w_max": 1.5,
    "action_mode": "discrete",
    "max_steps": 400,
    "goal_tolerance": 0.3,
    # sensor
    "n_rays": 24,
    "max_range": 3.0,
    "lidar_noise_std": 0.0,       # metres, gaussian
    "dropout_prob": 0.0,          # probability each ray returns nothing
    "dist_norm": 15.0,
    # people
    "human_speed": 0.5,
    "humans_start_step": 0,       # >0 = people frozen until this step (non-stationary test)
    "humans_yield": True,         # people wait instead of walking into the robot (False = env v1)
    # reward
    "shaping": "geodesic",        # geodesic | euclidean
    "k_progress": 1.0,
    "r_goal": 10.0,
    "r_collision": -10.0,
    "r_time": -0.01,
    "social_radius": 0.7,
    "w_social": 0.1,
    "clearance": 0.3,
    "w_clearance": 0.05,
    # hybrid A* + RL
    "guidance": False,
    "guidance_lookahead": 6,      # cells along the geodesic descent (~1.5 m)
}

# (v fraction, w fraction) for the discrete action set
DISCRETE_ACTIONS = np.array(
    [
        [1.0, 0.0],    # 0 forward
        [0.6, 0.6],    # 1 forward-left
        [0.6, -0.6],   # 2 forward-right
        [0.0, 1.0],    # 3 rotate left
        [0.0, -1.0],   # 4 rotate right
    ]
)
ACTION_NAMES = ["forward", "fwd-left", "fwd-right", "rot-left", "rot-right"]


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class HospitalNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: dict | None = None, render_mode=None):
        super().__init__()
        self.cfg = deepcopy(DEFAULT_CONFIG)
        if config:
            unknown = set(config) - set(DEFAULT_CONFIG)
            if unknown:
                raise KeyError(f"Unknown env config keys: {sorted(unknown)}")
            self.cfg.update(config)
        self.render_mode = render_mode

        n = self.cfg["n_rays"]
        self.obs_dim = n + 5 + (2 if self.cfg["guidance"] else 0)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        if self.cfg["action_mode"] == "discrete":
            self.action_space = gym.spaces.Discrete(len(DISCRETE_ACTIONS))
        elif self.cfg["action_mode"] == "continuous":
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            raise ValueError("action_mode must be 'discrete' or 'continuous'")

        self.ray_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # sample points along each ray for grid ray-marching (half-cell spacing)
        step = self.cfg["resolution"] / 2
        self._ray_samples = np.arange(step, self.cfg["max_range"] + 1e-9, step)
        self.layout = None

    # ------------------------------------------------------------------ setup
    def set_difficulty(self, level: int):
        """Called by the curriculum callback during training."""
        self.cfg["difficulty"] = int(level)

    def get_difficulty(self) -> int:
        return int(self.cfg["difficulty"])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "map_seed" in options:
            mseed = int(options["map_seed"])
        else:
            pool = self.cfg["train_map_pool"] if self.cfg["split"] == "train" else 10_000
            mseed = map_seed(self.cfg["split"], int(self.np_random.integers(pool)))

        c = self.cfg
        self.layout = generate_layout(
            mseed,
            difficulty=c["difficulty"],
            size_m=c["map_size"],
            resolution=c["resolution"],
            robot_radius=c["robot_radius"],
            n_obstacles=c["n_obstacles"],
            n_humans=c["n_humans"],
            n_walls=c["n_walls"],
        )
        hrng = np.random.default_rng(mseed + 7)  # people behave the same for the same map
        self.humans = [
            Human(xy, self.layout, hrng, speed=c["human_speed"], start_step=c["humans_start_step"],
                  yield_to_robot=c["humans_yield"])
            for xy in self.layout.human_spawns
        ]
        self.pos = self.layout.start.astype(float).copy()
        self.heading = float(self.layout.start_heading)
        self.v = self.w = 0.0
        self.t = 0
        self.path = [self.pos.copy()]
        self.path_length = 0.0
        self.optimal_length = self.layout.geodesic_m(self.layout.start)
        self.prev_dist = self._distance_to_go()
        self._last_valid = None
        self.outcome = "running"
        obs = self._observe()
        return obs, self._info(reward_terms={})

    # --------------------------------------------------------------- dynamics
    def _command(self, action):
        c = self.cfg
        if c["action_mode"] == "discrete":
            fv, fw = DISCRETE_ACTIONS[int(action)]
        else:
            a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1, 1)
            fv, fw = (a[0] + 1) / 2, a[1]  # forward-only robot: v in [0, v_max]
        return fv * c["v_max"], fw * c["w_max"]

    def _collides(self, xy):
        """Robot circle vs static map (8 perimeter points + centre) and vs people."""
        r = self.cfg["robot_radius"]
        g = self.layout.grid
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        pts = np.vstack([xy, xy + r * np.c_[np.cos(angles), np.sin(angles)]])
        cols = np.floor(pts[:, 0] / self.layout.resolution).astype(int)
        rows = np.floor(pts[:, 1] / self.layout.resolution).astype(int)
        h, w = g.shape
        if (rows < 0).any() or (cols < 0).any() or (rows >= h).any() or (cols >= w).any():
            return "wall"
        if g[rows, cols].any():
            return "wall"
        for hu in self.humans:
            if np.linalg.norm(hu.pos - xy) < r + hu.radius:
                return "human"
        return None

    def _distance_to_go(self):
        if self.cfg["shaping"] == "geodesic":
            return self.layout.geodesic_m(self.pos)
        return float(np.linalg.norm(self.layout.goal - self.pos))

    def step(self, action):
        if self.outcome != "running":
            raise RuntimeError("Episode finished: call reset()")
        c = self.cfg
        self.v, self.w = self._command(action)
        # integrate kinematics in two sub-steps (prevents tunnelling through thin walls)
        collision = None
        for _ in range(2):
            self.heading = wrap_angle(self.heading + self.w * c["dt"] / 2)
            new_pos = self.pos + self.v * c["dt"] / 2 * np.array([math.cos(self.heading), math.sin(self.heading)])
            collision = self._collides(new_pos)
            if collision:
                break
            self.path_length += float(np.linalg.norm(new_pos - self.pos))
            self.pos = new_pos
        self.t += 1
        for hu in self.humans:
            hu.step(c["dt"], self.t, self.pos, c["robot_radius"])
        if collision is None:
            for hu in self.humans:  # a person may walk into the robot
                if np.linalg.norm(hu.pos - self.pos) < c["robot_radius"] + hu.radius:
                    collision = "human"
        self.path.append(self.pos.copy())

        # ---- reward terms
        dist = self._distance_to_go()
        terms = {"progress": c["k_progress"] * (self.prev_dist - dist), "time": c["r_time"]}
        self.prev_dist = dist
        d_h = self.min_human_distance()
        terms["social"] = -c["w_social"] * (c["social_radius"] - d_h) / c["social_radius"] if d_h < c["social_radius"] else 0.0

        obs = self._observe()
        terms["clearance"] = -c["w_clearance"] if self._min_static_range < c["clearance"] else 0.0

        terminated = truncated = False
        if collision:
            terms["collision"] = c["r_collision"]
            self.outcome = f"collision_{collision}"
            terminated = True
        elif np.linalg.norm(self.layout.goal - self.pos) < c["goal_tolerance"]:
            terms["goal"] = c["r_goal"]
            self.outcome = "success"
            terminated = True
        elif self.t >= c["max_steps"]:
            self.outcome = "timeout"
            truncated = True
        reward = float(sum(terms.values()))
        return obs, reward, terminated, truncated, self._info(terms)

    # ---------------------------------------------------------------- sensing
    def min_human_distance(self) -> float:
        """Surface-to-surface distance to the closest person (inf if none)."""
        if not self.humans:
            return float("inf")
        r = self.cfg["robot_radius"]
        return float(min(np.linalg.norm(h.pos - self.pos) - r - h.radius for h in self.humans))

    def lidar(self) -> np.ndarray:
        """True (noise-free) ranges of all rays in metres."""
        c = self.cfg
        angles = self.heading + self.ray_angles
        dirs = np.c_[np.cos(angles), np.sin(angles)]                    # [R, 2]
        pts = self.pos + dirs[:, None, :] * self._ray_samples[None, :, None]  # [R, S, 2]
        res = self.layout.resolution
        g = self.layout.grid
        cols = np.floor(pts[..., 0] / res).astype(int)
        rows = np.floor(pts[..., 1] / res).astype(int)
        h, w = g.shape
        outside = (rows < 0) | (cols < 0) | (rows >= h) | (cols >= w)
        hit = outside.copy()
        hit[~outside] = g[rows[~outside], cols[~outside]]
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        ranges = np.where(first >= 0, self._ray_samples[np.maximum(first, 0)], c["max_range"])
        self._min_static_range = float(ranges.min())
        # analytic ray-circle intersection for people
        for hu in self.humans:
            rel = hu.pos - self.pos
            b = dirs @ rel
            cc = rel @ rel - hu.radius ** 2
            disc = b * b - cc
            ok = disc >= 0
            t_hit = np.where(ok, b - np.sqrt(np.maximum(disc, 0)), np.inf)
            t_hit = np.where(t_hit > 0, t_hit, np.inf)
            ranges = np.minimum(ranges, t_hit)
        return np.minimum(ranges, c["max_range"])

    def _observe(self) -> np.ndarray:
        c = self.cfg
        ranges = self.lidar()
        if c["lidar_noise_std"] > 0:
            ranges = np.clip(ranges + self.np_random.normal(0, c["lidar_noise_std"], ranges.shape), 0, c["max_range"])
        scan = ranges / c["max_range"]
        if c["dropout_prob"] > 0:
            dropped = self.np_random.random(scan.shape) < c["dropout_prob"]
            scan = np.where(dropped, -1.0, scan)  # -1 = "no return" sentinel
        to_goal = self.layout.goal - self.pos
        bearing = wrap_angle(math.atan2(to_goal[1], to_goal[0]) - self.heading)
        parts = [
            scan,
            [min(np.linalg.norm(to_goal) / c["dist_norm"], 1.0), math.sin(bearing), math.cos(bearing)],
            [self.v / c["v_max"], self.w / c["w_max"]],
        ]
        if c["guidance"]:
            wp = self.waypoint()
            d = wp - self.pos
            wb = wrap_angle(math.atan2(d[1], d[0]) - self.heading)
            parts.append([math.sin(wb), math.cos(wb)])
        return np.clip(np.concatenate(parts), -1, 1).astype(np.float32)

    def waypoint(self) -> np.ndarray:
        """A* look-ahead point: follow the geodesic field downhill from the robot."""
        cell = descend(self.layout.geodesic, self.layout.cell(self.pos), self.cfg["guidance_lookahead"])
        if self.layout.geodesic[cell] == 0:
            return self.layout.goal.copy()
        return self.layout.centre(cell)

    # ---------------------------------------------------------------- helpers
    def _info(self, reward_terms):
        return {
            "outcome": self.outcome,
            "is_success": self.outcome == "success",
            "reward_terms": reward_terms,
            "min_human_dist": self.min_human_distance(),
            "pos": self.pos.copy(),
            "heading": self.heading,
            "t": self.t,
            "path_length": self.path_length,
            "optimal_length": self.optimal_length,
            "map_seed": self.layout.seed,
        }

    def state_snapshot(self) -> dict:
        """Privileged full state - used by the renderer and the A* baseline, never by RL agents."""
        return {
            "grid": self.layout.grid,
            "resolution": self.layout.resolution,
            "pos": self.pos.copy(),
            "heading": self.heading,
            "goal": self.layout.goal.copy(),
            "humans": [h.pos.copy() for h in self.humans],
            "human_radius": Human.radius,
            "robot_radius": self.cfg["robot_radius"],
            "path": np.array(self.path),
            "lidar": self.lidar(),
            "ray_angles": self.ray_angles + self.heading,
            "waypoint": self.waypoint() if self.cfg["guidance"] else None,
            "t": self.t,
            "outcome": self.outcome,
        }

    def render(self):
        from ..render import draw_frame_rgb

        return draw_frame_rgb(self.state_snapshot())


def make_env(config: dict | None = None, wrappers: dict | None = None):
    """Factory used everywhere: env config + observation wrappers (see wrappers.py)."""
    from .wrappers import apply_wrappers

    return apply_wrappers(HospitalNavEnv(config), wrappers or {})
