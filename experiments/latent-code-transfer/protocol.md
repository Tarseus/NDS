# Protocol: H0/H1 Latent-Code Transfer Pilot

Status: locked before implementation or execution.

## Question

Does the pretrained CVRP100 NDS checkpoint contain latent codes whose destroy-repair performance is sufficiently reliable across stochastic rollouts and sufficiently transferable across instances to justify learned cross-instance adaptation?

## Fixed experimental object

- Backbone: `models/cvrp_100/checkpoint-2000.pt`.
- Problem and scale: CVRP100 only.
- Backbone parameters: frozen.
- Latent space: existing 10-bit binary codes.
- Candidate panel: 32 distinct codes sampled once from the 1024-code pool with a recorded seed.
- Replicas: 4 stochastic rollouts per code, arranged so all candidates see the same incumbent.
- Instances: 64 per distribution, split into 16 calibration and 48 evaluation instances.
- Destroy size and repair settings: inherited from the checkpoint.

## Distribution families

1. `uniform`: original Kool-style uniform generator.
2. `x_uniform_center`: X generator, uniform customers, centered depot, demand type 2, route-size regime 3.
3. `x_cluster_center`: clustered customers with the remaining settings unchanged.
4. `x_mixed_center`: random-clustered customers with the remaining settings unchanged.
5. `x_cluster_corner_quad`: clustered customers, corner depot, quadrant-dependent demand type 6.
6. `x_mixed_random_fewlarge`: random-clustered customers, random depot, few-large demand type 7, short-route regime 2.

All distributions use exactly 100 customers.

## Measurements

For instance \(i\), code \(z\), and replica \(r\):

\[
u_{izr}=\max(C_i-C_{izr}',0)/(C_i+\epsilon).
\]

Primary code utility is the mean over replicas. The experiment records the full instance × code × replica tensor.

### H0 reliability test

- Split replicas into two halves.
- Compute within-instance and within-distribution Spearman rank correlation between code utilities from the two halves.
- Estimate bootstrap confidence intervals over instances.

### H1 transfer tests

- Random code.
- Globally best code estimated from calibration instances of the source distributions.
- Distribution-best code transferred from each source distribution to every target distribution.
- Feature-nearest source-instance code, using only coordinate, depot, demand, capacity and route-size summary statistics.
- Target-distribution calibration code and per-instance oracle code are reported only as upper bounds.

## Confirmatory success criteria

Proceed to full-search warm-start experiments only if:

1. Median split-half code-rank Spearman correlation is at least 0.20 with a positive 95% bootstrap lower bound on at least three distributions; and
2. Feature-nearest or source-distribution transfer improves mean normalized reward by at least 10% relative to random code on at least three OOD distributions, with a positive paired-bootstrap 95% lower bound.

Proceed to Fisher/connection experiments only if H1 succeeds and ordinary features leave meaningful unexplained transfer regret.

## Failure interpretation

- Low reliability refutes the existence of a stable transport target in this checkpoint.
- Reliable codes but no cross-instance transfer supports per-instance search, not cross-instance transport.
- Strong simple-feature retrieval weakens the case for Fisher geometry but supports memory-based automatic adaptation.

## Non-claims

This one-step pilot does not establish better full-LNS final objective, anytime AUC, or wall-clock time-to-target. Those require a second, sequential search experiment.

