# LNS 跨分布自动泛化：阶段实验结论

日期：2026-08-03

## 一句话结论

这个想法**部分成立，但不能按“为每个分布或实例静态选一个低维 latent code”来做**。NDS 的 latent code 确实有稳定、可跨分布复用的性能结构；然而在完整 200 步 LNS 中，静态单 code 和静态 top-8 都输给保留全部 32 个 code 的等预算 panel。真正有希望的创新点应改成：**根据不断变化的 incumbent/search state 在线分配 latent rollout 预算，并保留多样性与安全回退**。

## 已完成的证据链

1. **latent 排序不是噪声。** 64 个实例、32 个 code、4 replicas 的 split-half 排序可靠性在 6/6 分布通过；median Spearman 为 0.294-0.387，所有 bootstrap 下界均大于 0。
2. **单一 uniform 记忆不安全。** uniform-only feature kNN 只在 1/5 个 OOD 分布达到预注册标准，并在 cluster-center 上显著负迁移。
3. **多源分布记忆的一步迁移成立。** 用其余五个分布的校准均值选 code，在全新实例种子上对 6/6 留出分布均显著优于随机 code；相对增益为 +21.4% 到 +45.3%。
4. **静态单 code 无法转化为完整 LNS。** 200 次迭代、相同 32-rollout 预算下，相对随机固定 code 的 final 和 anytime 标准均为 0/6，通过显著伤害分布为 4/6。
5. **静态 top-8 仍不够。** 8 个跨域高分 code 各 4 replicas，相对完整 32-code panel 的 final 和 anytime 标准仍为 0/6；六个目标至少一个主指标显著更差。

## 研究判断

- 可以保留的核心主张：跨实例、跨分布的 latent 经验是可复用的，适合作为 prior、warm-start 或预算分配信号。
- 被实验否定的主张：一个静态 instance/distribution-specific latent，或静态 top-k latent，本身足以形成自动泛化 LNS。
- 下一步最有信息量的机制：每隔若干 LNS 步，用 incumbent 特征、近期 code reward 和不确定性对 32-code panel 在线重加权；任何低置信决策回退到完整 panel。
- Fisher 几何目前不应优先。若不先解决“搜索状态变化 + 多样性损失”，更复杂的静态相似度仍可能重复失败。

## 可复现锚点

- 源分支：`Tarseus/NDS:agent/crossdist-latent-transfer`
- 一步确认结果：`results/multisource-confirm-20260803-073235/g51`，提交 `df393a1`
- 静态单 code full-LNS：`results/full-lns-transfer-20260803-074321/g51`，提交 `1c0c177`
- top-8 full-LNS：`results/portfolio-lns-transfer-20260803-075441/g51`，提交 `7b30a2f`
- 原始指标与张量已拉回各实验目录的 ignored `results/` 子目录，并与 g51 SHA-256 逐文件一致。
