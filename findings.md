# Research Findings

## Research Question

Can a frozen seed-conditioned Neural Deconstruction Search solver reuse latent-code experience across same-size CVRP instances and unseen distributions to reduce the adaptation/search budget?

## Current Understanding

The broad cross-distribution objective is established in prior neural routing work. The narrower potential contribution is safe reuse of search adaptation across instances. Before constructing Fisher metrics or learned connections, we must establish that the pretrained code space contains stable, transferable performance structure.

## Key Results

- The end-to-end CPU smoke test completed across all six planned distributions and produced the expected raw utility tensor, manifest and analysis report.
- In the g51 pilot, split-half latent-code rank reliability passed on all six distributions. Median per-instance Spearman correlations ranged from 0.294 to 0.387 and all distribution-level bootstrap confidence intervals were strictly positive. H0 is rejected: code performance is not dominated by rollout noise.
- Uniform-only feature kNN did not generalize safely. It passed the locked OOD criterion on only `x_uniform_center` (1/5 OOD targets), was significantly negative on `x_cluster_center` (-24.8% relative; 95% CI [-0.007266, -0.000124]), and therefore failed H1.
- This failure is specific to the single-source retrieval rule. Exploratory leave-one-distribution-out selection by mean reward across the other five distributions produced positive point gains on 6/6 targets (+0.003740 to +0.009304). An independent confirmation is required before treating this as evidence.

## Patterns and Insights

- The clean upstream checkpoint contains a 10-bit latent-conditioned decoder.
- The user's active worktree removes that conditioning and is preserved as a separate no-seed line.
- The initial pilot can evaluate all codes from the same incumbent in one batched destroy-repair call, avoiding target-model training.
- Distribution-level transfer is structured: code 3, selected by both `x_uniform_center` and `x_mixed_center`, improved all six observed targets, whereas the uniform-selected code 8 harmed the corner/quad distribution.
- Simple instance-feature nearest-neighbor selection can be worse than a single robust global code. The next iteration should test distribution-diverse calibration before adding Fisher geometry.

## Lessons and Constraints

- The local Python installation is CPU-only; GPU experiments should run in a clean remote clone.
- The original working tree is dirty in core model/trainer files and must not be used as the source boundary for this experiment.
- One-step reward is a screening proxy, not evidence of improved full LNS anytime performance.
- The first inferential GPU run used one instance seed. H1b must be evaluated on a fresh seed with a locked leave-one-distribution-out rule.
- Local cppimport compilation requires running outside the restricted Windows sandbox because the compiler creates nested temporary build directories.

## Open Questions

- Does multi-source mean-code selection retain positive transfer on fresh instances for at least four of six held-out distributions?
- Can a safe router recognize when instance-level retrieval should defer to the multi-source global code?
- Does the one-step gain translate to improved full-LNS anytime performance after H1b is confirmed?

## Optimization Trajectory

1. CPU smoke: pipeline validation only.
2. g51 pilot: rejected rollout-noise H0, rejected uniform-only kNN H1, and did not proceed to full search.
3. Current direction: independently confirm distribution-diverse leave-one-distribution-out memory (H1b), then test a safe router; Fisher geometry remains gated.
