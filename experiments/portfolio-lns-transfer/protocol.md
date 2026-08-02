# Multi-Source Top-8 Portfolio Protocol

Status: locked before implementation.

## Question

Can cross-distribution memory improve iterative LNS when it selects a diverse high-value portfolio instead of collapsing all rollouts onto one code?

## Fixed design

- Same frozen CVRP100 checkpoint, six distributions, SA schedule, 200 iterations, 24 instances per distribution, and 32 total neural rollouts as the H3 protocol.
- Fresh target-instance seed: `20260809`.
- Random-fixed assignment seed: `20260810`.
- Bootstrap seed: `20260811`, 10,000 instance resamples.
- Anytime checkpoints: 0, 1, 2, 5, 10, 20, 50, 100, and 200.
- Starting costs must be numerically identical across strategies for every instance.

## Frozen LODO top-8 portfolios

The rankings were computed from the independent H1b confirmation source memory, excluding each target distribution and averaging source-distribution calibration means. Pool indices are frozen as:

- uniform: 959, 175, 955, 303, 267, 910, 762, 489
- x_uniform_center: 175, 959, 955, 303, 267, 910, 762, 489
- x_cluster_center: 175, 959, 303, 955, 910, 762, 267, 489
- x_mixed_center: 175, 959, 910, 955, 303, 489, 267, 762
- x_cluster_corner_quad: 959, 910, 175, 303, 955, 267, 762, 313
- x_mixed_random_fewlarge: 175, 959, 955, 303, 910, 267, 762, 489

Each portfolio code receives exactly four stochastic replicas, matching the calibration replica count and totaling 32 rollouts.

## Locked strategies

1. `random_fixed`: one deterministic random candidate code per instance, repeated 32 times.
2. `multisource_top8`: eight frozen LODO codes with four replicas each.
3. `random_panel`: all 32 fixed candidate codes with one rollout each.

## Locked success rule

H4 passes if `multisource_top8` has:

1. a strictly positive 95% bootstrap lower bound for final normalized advantage over `random_panel` on at least 4 of 6 distributions;
2. a strictly positive 95% bootstrap lower bound for anytime-AUC advantage over `random_panel` on at least 4 of 6 distributions; and
3. no distribution with a significantly negative primary final or AUC advantage.

Comparison against `random_fixed` is secondary. If H4 fails, static cross-instance memory is not sufficient and the next direction must adapt to incumbent/search state rather than add geometry to a fixed selector.
