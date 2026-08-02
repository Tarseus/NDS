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
- H1b was independently confirmed on fresh instance seed `20260804`: multi-source mean-code transfer passed on 6/6 held-out distributions, with +21.4% to +45.3% relative gain and strictly positive 95% bootstrap lower bounds. No target was significantly negative.
- On `x_cluster_corner_quad`, uniform-only transfer remained harmful (-0.003927 raw gain) while multi-source transfer was strongly positive (+0.012654, 95% CI [+0.008145, +0.017639]). Distribution diversity is therefore causally relevant to the selection rule, not merely extra calibration volume.

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
- H1b concerns a single destroy-repair step. It does not establish final solution quality, anytime behavior, or convergence under iterative LNS.
- Local cppimport compilation requires running outside the restricted Windows sandbox because the compiler creates nested temporary build directories.

## Open Questions

- Does the confirmed one-step advantage improve 200-iteration full-LNS final cost and anytime area under an equal 32-rollout budget?
- Can a safe router recognize when instance-level retrieval should defer to the multi-source global code?
- Does the one-step gain translate to improved full-LNS anytime performance after H1b is confirmed?

## Optimization Trajectory

1. CPU smoke: pipeline validation only.
2. g51 pilot: rejected rollout-noise H0, rejected uniform-only kNN H1, and did not proceed to full search.
3. Fresh-seed confirmation: H1b passed on 6/6 held-out distributions with no significant negative target.
4. Current direction: full-LNS translation test (H3), followed by a safe router if fixed-code gains are not uniform; Fisher geometry remains gated.
