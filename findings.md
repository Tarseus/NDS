# Research Findings

## Research Question

Can a frozen seed-conditioned Neural Deconstruction Search solver reuse latent-code experience across same-size CVRP instances and unseen distributions to reduce the adaptation/search budget?

## Current Understanding

The pretrained code space contains stable, cross-distribution performance structure, but that structure is useful only as a prior. Static single-code and static top-k reuse lose to a diverse equal-budget panel in iterative LNS. The viable research target is now dynamic, incumbent-conditioned rollout allocation with explicit diversity and safe fallback.

## Key Results

- The end-to-end CPU smoke test completed across all six planned distributions and produced the expected raw utility tensor, manifest and analysis report.
- In the g51 pilot, split-half latent-code rank reliability passed on all six distributions. Median per-instance Spearman correlations ranged from 0.294 to 0.387 and all distribution-level bootstrap confidence intervals were strictly positive. H0 is rejected: code performance is not dominated by rollout noise.
- Uniform-only feature kNN did not generalize safely. It passed the locked OOD criterion on only `x_uniform_center` (1/5 OOD targets), was significantly negative on `x_cluster_center` (-24.8% relative; 95% CI [-0.007266, -0.000124]), and therefore failed H1.
- This failure is specific to the single-source retrieval rule. Exploratory leave-one-distribution-out selection by mean reward across the other five distributions produced positive point gains on 6/6 targets (+0.003740 to +0.009304). An independent confirmation is required before treating this as evidence.
- H1b was independently confirmed on fresh instance seed `20260804`: multi-source mean-code transfer passed on 6/6 held-out distributions, with +21.4% to +45.3% relative gain and strictly positive 95% bootstrap lower bounds. No target was significantly negative.
- On `x_cluster_corner_quad`, uniform-only transfer remained harmful (-0.003927 raw gain) while multi-source transfer was strongly positive (+0.012654, 95% CI [+0.008145, +0.017639]). Distribution diversity is therefore causally relevant to the selection rule, not merely extra calibration volume.
- H3 failed decisively in full LNS. Repeating the selected code for all 32 rollouts was worse than a random fixed code by 0.28%–1.67% final normalized cost on every distribution, with significant harm on four; it was also 0.64%–2.05% worse than the 32-code panel.
- The failure begins immediately rather than only after convergence: at iteration one the multi-source code led random fixed on only 3/6 targets. Mean four-replica calibration utility is therefore mismatched to the best-of-32 proposal mechanism used by iterative LNS.
- H4 also failed. A multi-source top-8 portfolio with four replicas per code was 0.15%-0.96% worse in final normalized cost than the equal-budget 32-code panel on all six fresh targets. Final and anytime criteria each passed on 0/6, and every target had significant harm on at least one primary metric.
- The top-8 portfolio was slightly better than random fixed on 5/6 targets (-0.03% to +0.54%). Cross-domain memory therefore contains weak useful prior information, but the benefit is smaller than the loss from removing latent-policy diversity.

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
- A single static latent code collapses proposal diversity. Future automatic adaptation must retain a portfolio or change codes as the incumbent state evolves.
- Static distribution-level top-k selection is also insufficient. The target object for adaptation should be the evolving incumbent/search state, not merely the original instance distribution.
- Local cppimport compilation requires running outside the restricted Windows sandbox because the compiler creates nested temporary build directories.

## Open Questions

- Can an incumbent-conditioned allocator reweight all 32 codes online while retaining a safe full-panel fallback?
- Can adaptive allocation match the 32-code panel with fewer distinct code evaluations, yielding a real search-budget reduction?

## Optimization Trajectory

1. CPU smoke: pipeline validation only.
2. g51 pilot: rejected rollout-noise H0, rejected uniform-only kNN H1, and did not proceed to full search.
3. Fresh-seed confirmation: H1b passed on 6/6 held-out distributions with no significant negative target.
4. Full-LNS translation rejected static single-code reuse (H3).
5. Calibrated top-8 diversity also failed against the full panel (H4).
6. The next justified experiment is dynamic incumbent-conditioned allocation (H5); Fisher geometry remains deferred.
