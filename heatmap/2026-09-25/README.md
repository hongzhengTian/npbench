# 2026-09-25 修复增量视图

以 2026-09-24 完整 L 矩阵为底稿，只更新 `selection.json` 列出的九个目标单元、共 27 个生命周期格子。
包含 Azimint naive/hist helper 的 OpenMP GPU，以及七项性能回归复测；Symm GPU 如实更新为仍未改善的观测，不按速度筛选结果。
非目标抽查和 Symm CPU 归因样本不覆盖底稿。
其余 3213 个格子逐字段保持不变，其中包括全部 2268 个对照组格子。
这是混合版本的增量视图，不是 3ec3dae8 的全矩阵重新认证，也不是同期性能排名。

更新样本来自 Research 长期证据 `notebook/01-research/cova/reports/evidence/2026-09-25-regression-repairs/records/targets/candidate`。
测量时为 316658e2 加未提交补丁，现已提交为 CoVA 3ec3dae8；归档中每个修改文件的 SHA256 已与提交后的源码核对一致。
总 JSON 保留原始测量基线、补丁 SHA256、提交身份、各进程输入/golden 身份及证据文件哈希。
每单元严格三进程 × 三调用，L、system、CPU affinity 0–95、同一 GPU，无 profiler。
初始化 1 个样本，新进程首调 2 个样本，热调用 6 个样本；热值为各进程热中位的中位。
三类图均沿用 09-24 的色标；GPU? 仍表示设备 metadata，没有独立的逐单元主计算 trace。
更新后合并视图有 291/324 个 CoVA 科学程序×路线单元通过；这是底稿加两项恢复的汇总，不能写成当前版本完整认证。
ADI 公式、golden 和缺失状态保持不变。

- [热调用图](baseline_with_cova_same_process_heatmap.png)
- [新进程首调用图](baseline_with_cova_fresh_process_heatmap.png)
- [初始化图](baseline_with_cova_initialization_heatmap.png)
- [总 JSON](baseline_with_cova_heatmaps_data.json)
- [验证记录](baseline_with_cova_verification.json)

从 NPBench 根目录，使用已有 matplotlib 环境重建；将 EVIDENCE 替换为上述长期证据目录：

```bash
python heatmap/scripts/overlay_cova_repairs.py \
  --previous heatmap/2026-09-24/baseline_with_cova_heatmaps_data.json \
  --baseline heatmap/2026-09-17/baseline_heatmaps_data.json \
  --evidence EVIDENCE \
  --selection heatmap/2026-09-25/selection.json \
  --source-version 3ec3dae8 \
  --output heatmap/2026-09-25
```

本次只读取已有证据并生成结果文件，没有重跑 benchmark。
