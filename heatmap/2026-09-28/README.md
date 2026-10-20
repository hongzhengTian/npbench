# CoVA L 2026-09-28批次（09-29完成及整理）

CoVA `f929db7f`，集合`cova-L-20260928-201630`；原始331单元291严格通过、36失败、4缺失源码。
预定helper映射后的54×6视图291通过；与09-25/26功能状态一致。
对照组2268个生命周期格完全不变，未重跑；CoVA只使用本轮完整数据，不混入更快的专项值。

**收尾预检发现外部GPU约70GB、99%利用率。** 缺少逐单元竞争时间线，不能直接认证GPU回退或同期排名。
三张图均保留实测值；CPU/GPU?标记及JSON中的mixed/placement资格沿用现有规则。
热调用按进程等权中位数，初始化/fresh分开；L、system、严格3×3、golden零重算，ADI仅corrected。

相对09-25：50项热时间下降≥20%（26项≥2×加速），41项上升≥20%（12项≥2×变慢），无≥10×回退；不是因果或显著性检验。
报告与精炼证据在前身报告 `raw://cova/2026-10-20-predecessor-archive/notes.tar.gz#notebook/01-research/cova/reports/2026-09-26-compute-stencil-access-storage-performance.md` 和 `raw://cova/2026-10-20-predecessor-archive/step-evidence.tar.gz#2026-09-29-npbench-cova-L`。

NPBench根目录重绘（不运行benchmark）：

```bash
python heatmap/scripts/plot_cova_collection.py \
  --baseline heatmap/2026-09-17/baseline_heatmaps_data.json \
  --evidence <unpacked step-evidence directory>/2026-09-29-npbench-cova-L \
  --output heatmap/2026-09-28
```
