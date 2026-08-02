# Full-LNS Translation Protocol

Status: locked before runner implementation.

## Question

Does the independently confirmed multi-source one-step advantage improve final solution quality and anytime behavior in iterative LNS under an equal neural-rollout budget?

## Fixed design

- Frozen checkpoint: `models/cvrp_100/checkpoint-2000.pt`.
- Six CVRP100 target distributions from the earlier protocols.
- Fresh target-instance seed: `20260806`; 24 instances per distribution.
- Candidate code panel: the same 32 codes fixed by code seed `20260803`.
- 200 destroy-repair iterations, 32 neural rollouts per instance and iteration, augmentation factor 1.
- SA schedule: temperature 0.1 to 0.001 by geometric cooling; `recreate_n=5`, `beta=0`, and insertion restricted to new tours.
- Anytime checkpoints: 0, 1, 2, 5, 10, 20, 50, 100, and 200 iterations.
- All strategies must start from numerically identical instance costs; the runner aborts otherwise.

## Locked strategies

1. `random_fixed`: assign each target instance one deterministic uniformly random member of the 32-code panel using seed `20260807`, then repeat it for all 32 rollouts and all iterations.
2. `multisource_fixed`: repeat the leave-one-distribution-out code selected in independent confirmation for all 32 rollouts and all iterations. Pool indices are fixed as: uniform 959; x_uniform_center 175; x_cluster_center 175; x_mixed_center 175; x_cluster_corner_quad 959; x_mixed_random_fewlarge 175.
3. `random_panel`: evaluate each of the 32 fixed candidates once per iteration; this is a stronger equal-forward-pass diversity baseline.
4. `uniform_fixed`: repeat uniform-calibrated pool index 910. This is a secondary baseline and is not evaluated as an external-memory baseline on the uniform target itself.

Stochastic policy sampling and repair use strategy/instance-independent common seeds where tensor shapes permit. Strategy order is fixed as above and all raw checkpoint costs are retained.

## Metrics

- Primary final metric: paired normalized cost advantage of `multisource_fixed` over `random_fixed`, `(cost_random - cost_multi) / initial_cost`, bootstrapped over target instances.
- Primary anytime metric: paired area under the best-so-far normalized-improvement curve over log-scaled iteration checkpoints.
- Secondary comparisons: `multisource_fixed` versus `random_panel` and `uniform_fixed`.
- Bootstrap: 10,000 instance resamples per distribution, seed `20260808`.

## Locked success rule

H3 passes if:

1. `multisource_fixed` has a strictly positive 95% bootstrap lower bound for final normalized advantage over `random_fixed` on at least 4 of 6 distributions;
2. it has a strictly positive 95% bootstrap lower bound for anytime-AUC advantage on at least 4 of 6 distributions; and
3. neither primary metric is significantly negative (95% bootstrap upper bound below zero) on any distribution.

Performance against `random_panel` is reported but is not part of the pass rule because that strategy searches all 32 latent policies at every iteration rather than reusing a single code.
