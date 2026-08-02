# Research Log

Chronological, append-only record of decisions and evidence.

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-08-02 | bootstrap | Scoped the first phase to CVRP100 because the repository already contains a matching seed-conditioned checkpoint and the local machine has no GPU. The first test is deliberately pre-geometry: measure whether fixed latent-code performance is reliable and transferable across generated same-size distributions. H2 (Fisher/transport) is gated on positive H0/H1 evidence. |
| 2 | 2026-08-02 | inner-loop | Local CPU smoke run completed on all six distributions with 2 instances, 2 codes, and 2 replicas. It compiled the CVRP extension, loaded checkpoint epoch 2000, executed same-incumbent code tournaments, wrote raw metrics and completed analysis in about 21 seconds including compilation. This run validates the pipeline only; its sample size is intentionally non-inferential. |
