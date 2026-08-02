# Research Findings

## Research Question

Can a frozen seed-conditioned Neural Deconstruction Search solver reuse latent-code experience across same-size CVRP instances and unseen distributions to reduce the adaptation/search budget?

## Current Understanding

The broad cross-distribution objective is established in prior neural routing work. The narrower potential contribution is safe reuse of search adaptation across instances. Before constructing Fisher metrics or learned connections, we must establish that the pretrained code space contains stable, transferable performance structure.

## Key Results

No experimental result yet.

## Patterns and Insights

- The clean upstream checkpoint contains a 10-bit latent-conditioned decoder.
- The user's active worktree removes that conditioning and is preserved as a separate no-seed line.
- The initial pilot can evaluate all codes from the same incumbent in one batched destroy-repair call, avoiding target-model training.

## Lessons and Constraints

- The local Python installation is CPU-only; GPU experiments should run in a clean remote clone.
- The original working tree is dirty in core model/trainer files and must not be used as the source boundary for this experiment.
- One-step reward is a screening proxy, not evidence of improved full LNS anytime performance.

## Open Questions

- Are code rankings reproducible across stochastic rollout replicas?
- Does source-instance code selection beat a random or global code on held-out distributions?
- If transfer exists, do simple instance features already explain it?

## Optimization Trajectory

No runs completed.

