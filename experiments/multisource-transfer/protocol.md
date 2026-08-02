# Multi-Source Leave-One-Distribution-Out Confirmation

Status: locked before analyzer implementation.

## Question

On fresh CVRP100 instances, can distribution-diverse calibration select a latent code that improves one-step destroy-repair reward on a completely held-out distribution?

## Fixed design

- Frozen checkpoint: `models/cvrp_100/checkpoint-2000.pt`.
- Candidate panel: the same 32 precommitted codes selected with code seed `20260803`.
- Fresh instance seed: `20260804`, not used in the pilot or exploratory steering analysis.
- Six fixed CVRP100 distributions from the first protocol.
- 64 instances per distribution and 4 stochastic replicas per code.
- For each held-out target distribution:
  - exclude all target-distribution instances from code selection;
  - use the first 16 instances from each of the other five distributions as calibration memory;
  - select the code with maximum mean normalized reward across the five source distributions;
  - evaluate that code on all 64 fresh target instances;
  - compare against the mean reward of a uniformly random candidate code on those same target instances.
- Secondary baselines: uniform-only global code and five-source instance-feature 5-NN retrieval.
- Bootstrap unit: target instance, 10,000 resamples with seed `20260805`.

## Locked success rule

H1b passes if the multi-source mean code achieves:

1. at least 10% relative gain over random with a strictly positive 95% bootstrap lower bound on at least 4 of 6 held-out distributions; and
2. no distribution with a strictly negative 95% bootstrap upper bound.

Only if H1b passes may a full-LNS anytime experiment be registered. Fisher geometry remains gated until a safe simple-feature baseline has been tested against the multi-source global code.

## Exploratory provenance

The rule was motivated by post-hoc analysis of run `crossdist-code-transfer-20260803-072321`, where multi-source mean-code point gain was positive on 6/6 targets. Those values are not confirmation evidence.
