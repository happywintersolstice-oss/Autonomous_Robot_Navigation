# Milestone 4 — Part B refinement results

Milestones 1–3 were completed and committed before this work. Milestone 4 now implements the bounded search, paired confirmations, reward-only experiment and frozen voting. The next planned milestone is 5: integration and core robustness on the fixed map. The final test remains sealed.

## Frozen protocol and actual coverage

The search sampled eight distinct settings from the declared finite grid, using RNG seed 20261106. Both Q-learning and SARSA received the same eight settings and development seeds 7/19. Settings were selected by mean complete-validation success, then SPL, seed SD and candidate ID. Checkpoints were selected by the unchanged 30-pair monitor, then evaluated on all 928 validation scenarios.

Confirmations and matched original-setting controls used fresh training seeds 101, 211, 307, 401, 503. Every run received 200,000 interactions. The selected reward experiment changed only ordinary step reward from -.01 to 0, using a separate reward version/hash; it did not rewrite maps or scenario manifests. No settings were retuned for that variant.

Completed **62 training runs / 12,400,000 interactions**: 32 search (6.4m), 10 matched controls (2m), 10 tuned confirmations (2m), and 10 reward ablations (2m). The run phase took **858.57 seconds (14.31 minutes)**. New training-run artifacts used **266.19 MiB**, below the declared 45-minute / 750-MiB limits. Voting reuses already trained members and adds zero training interactions.

The full run budget is counted even when monitoring selects an earlier checkpoint. New records retain selected-prefix visitation separately from full-budget visitation. Fresh training seeds check stability on familiar validation tasks; they do not make these independently held-out final-test scores.

## Selected settings

| Family | Candidate | Learning rate | Discount | Initial epsilon | Final epsilon | Decay fraction |
|---|---|---:|---:|---:|---:|---:|
| Q-learning | c07 | 0.9 | 0.99 | 0.7 | 0.03 | 0.3 |
| SARSA | c07 | 0.9 | 0.99 | 0.7 | 0.03 | 0.3 |

Original settings: learning rate .3, discount .99, epsilon 1 → .1 over 80% of budget. All candidates, both development seeds and poor trials remain available in `search_summary.json`; no best-seed selection or extra search followed.

## Paired baseline versus tuned confirmations

| Family | Baseline success | Tuned success ± seed SD | Paired success change ± SD | Baseline SPL | Tuned SPL |
|---|---:|---:|---:|---:|---:|
| Q-learning | 64.74% | 89.96% ± 0.79 pp | +25.22 ± 2.30 pp | 0.611012 | 0.858850 |
| SARSA | 24.55% | 33.00% ± 1.64 pp | +8.45 ± 0.81 pp | 0.231198 | 0.309894 |

SD is sample SD over five independent training seeds, or over five paired seed differences as labeled. It is not an episode-level confidence interval or a significance claim. The matched controls use the same fresh seeds, budget, monitor, checkpoint rule and complete-validation cases as tuned runs; historical Part A means are kept separate.

| Family | Seed | Baseline goals / 928 | Tuned goals / 928 | Baseline SPL | Tuned SPL |
|---|---:|---:|---:|---:|---:|
| Q-learning | 101 | 603 | 842 | 0.604973 | 0.870814 |
| Q-learning | 211 | 623 | 840 | 0.633871 | 0.864899 |
| Q-learning | 307 | 618 | 828 | 0.632751 | 0.846258 |
| Q-learning | 401 | 598 | 838 | 0.612009 | 0.859996 |
| Q-learning | 503 | 562 | 826 | 0.571456 | 0.852285 |
| SARSA | 101 | 194 | 282 | 0.200802 | 0.288278 |
| SARSA | 211 | 227 | 309 | 0.232517 | 0.311595 |
| SARSA | 307 | 244 | 323 | 0.244877 | 0.326799 |
| SARSA | 401 | 236 | 304 | 0.235944 | 0.306639 |
| SARSA | 503 | 238 | 313 | 0.241848 | 0.316159 |

### Predeclared targets

Targets were success >=80%, collision <=2%, and SPL >=.75 for each tuned family's five-seed mean. They were saved before search and were not changed afterward.

- Q-learning: success 89.96%, collision 0.19%, SPL 0.858850. All three mean targets met.
- SARSA: success 33.00%, collision 2.03%, SPL 0.309894. Missed targets: success, collision, spl.

Missed targets and poor individual seeds are retained. No model is described as ready for unseen maps or physical robots.

## Reward-only ablation

| Family | Ordinary step reward | Success | Collision | Timeout | SPL | Mean return (version-specific) |
|---|---:|---:|---:|---:|---:|---:|
| Q-learning | -.01 | 89.96% | 0.19% | 9.85% | 0.858850 | 8.713860 |
| Q-learning | 0 | 32.95% | 1.88% | 65.17% | 0.301593 | 3.201509 |
| SARSA | -.01 | 33.00% | 2.03% | 64.98% | 0.309894 | 1.880864 |
| SARSA | 0 | 19.70% | 5.86% | 74.44% | 0.183971 | 1.676724 |

Goal remains +10 and collision -5; ordinary timeout has no extra penalty. Returns across these reward scales are not a navigation-quality ranking. `reward_ablation.json` reports paired success/collision/timeout/SPL differences and deliberately excludes return differences. The variant is diagnostic and was not automatically promoted or combined with another search.

## Frozen voting experiment

The original-reward tuned confirmation means selected **Q-learning** before reward ablation/voting. Members are the predeclared seeds [101, 211, 307] in that fixed priority order. A majority wins; a three-way tie uses the first member. No action masking, fallback, Q-value averaging or voting training occurs. The selected single is seed 101, not a cherry-picked best seed.

| System | Success / 928 | Collision | Timeout | SPL |
|---|---:|---:|---:|---:|
| member-seed101 | 842 | 2 | 84 | 0.870814 |
| member-seed211 | 840 | 1 | 87 | 0.864899 |
| member-seed307 | 828 | 3 | 97 | 0.846258 |
| Voting | 837 | 1 | 90 | 0.883931 |

Voting changed success by **-0.54 percentage points** and SPL by **+0.013117** versus the predeclared single. Comparisons against every member are in `voting_comparison.json`; this is one fixed ensemble, not an ensemble-seed uncertainty study. Component metrics are reused on the same 928 scenarios and are not counted as extra independent samples.

There were 5,119 three-way ties in 24,810 voting actions, affecting 140 episodes. Replays retain all member proposals and tie decisions, with no invented Q scores.

Q-array memory is 960,000 bytes for voting versus 320,000 for one policy (excluding Python overhead). Compressed member checkpoint files total 376,311 bytes versus 125,522 for the selected single, excluding metadata/descriptors. Members consumed 600,000 interactions and 35.54 training-loop seconds; that cost is already included in this study and is not added again. The wrapper adds zero training.

| Policy | Mean decision microseconds | Median decision microseconds |
|---|---:|---:|
| member-seed101 | 1.659 | 1.600 |
| member-seed211 | 1.681 | 1.700 |
| member-seed307 | 1.700 | 1.700 |
| voting | 7.512 | 7.400 |

Timing uses 1,000 round-robin validation initial-state queries per policy, with `perf_counter_ns`, excluding resets/rendering/learning. These are local measurements under concurrent machine activity, not a hardware-isolated latency guarantee.

## Verification and preserved scope

Audited **57,536 training-run validation episodes / 6,543,058 actions**, **1,240 periodic models**, and **928 voting episodes / 24,810 actions**. Checks reconcile source/task/model signatures, parent scenario relationships, training counts, checkpoint rules, selected-prefix visitation and greedy scores/actions. The voting audit recomputes every member proposal/tie and verifies no parameter changes.

All benchmark inputs, original Part A baselines and earlier artifacts were preserved. Final-test policy executions: zero. Exact training resume remains unsupported; snapshots are inference-only, and failed/interrupted status is explicit. Targets, candidate count, seeds and scope were not adjusted after outcomes.

The minimal viewer loads matching step-cost/terminal-only tasks and voting descriptors, labels reward versions/member order, and never trains. Tests and browser-specific observations are recorded separately in `milestone4_verification.json` and `milestone4_ui_check.json`. The retained practical playback target is 2 actions/second; no new quantitative browser timing benchmark is claimed.

Evidence: [summary](../artifacts/studies/part-b-v1/summary.json), [all search candidates](../artifacts/studies/part-b-v1/search_summary.json), [paired changes](../artifacts/studies/part-b-v1/paired_improvements.json), [reward ablation](../artifacts/studies/part-b-v1/reward_ablation.json), [voting and cost](../artifacts/studies/part-b-v1/voting_comparison.json), [audits](../artifacts/studies/part-b-v1/verification.json), [figures](../artifacts/studies/part-b-v1/figures/), and [protocol](milestone4_protocol.md).

## Remaining work

Results remain selected development validation on a familiar map. Inspect residual timeouts, collisions, poor seeds and reward sensitivity before final freezing. Milestone 5 adds the integrated dashboard and predefined core robustness groups; milestone 6 freezes/evaluates the final system. This milestone does not execute final test, automatically publish/deploy, or add PPO, unseen maps, moving obstacles, continuous control, hardware or the full dashboard.
