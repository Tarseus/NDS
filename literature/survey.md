# Focused Literature Survey

## Neural Deconstruction Search for Vehicle Routing Problems

- Hottung, Wong-Chung, Tierney, TMLR 2025.
- Introduces the NDS destroy-policy plus greedy repair and simulated annealing framework.
- Relevant limitation: the published generalization experiment uses a comparatively mild CVRP500 shift and does not reuse adaptation state across different instances.
- URL: https://openreview.net/forum?id=bCmEP1Ltwq

## COMPASS

- Chalumeau et al., NeurIPS 2023.
- Learns a continuous latent policy space and searches it separately for each test instance.
- It is the principal baseline for any claim about reducing per-instance latent search.
- URL: https://proceedings.neurips.cc/paper_files/paper/2023/hash/18d3a2f3068d6c669dcae19ceca1bc24-Abstract-Conference.html

## Omni-VRP

- Zhou et al., ICML 2023.
- Meta-learns a model initialization for fast adaptation across sizes and node distributions.
- It is the principal alternative explanation: a good shared initialization may remove the need for pairwise transport.
- URL: https://proceedings.mlr.press/v202/zhou23o.html

## MEMENTO

- Chalumeau et al., 2024.
- Uses inference-time memory to update a neural solver's action distribution from earlier outcomes.
- It motivates memory-based adaptation but primarily reuses attempts within the current solving process rather than explicitly transporting search state across different instances.
- URL: https://openreview.net/forum?id=VHGZjZmzsO

## Decision from the survey

The pilot must include random-code, global-code and ordinary feature-retrieval baselines. Geometry is not introduced until these simpler mechanisms are measured.

