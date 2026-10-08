# 2026-10-07 hsc-12 完整结果快照

CoVA 为 `7eb28298` 的 331 单元完整 L 采集，GPU worker 在测量时绑定 GPU 所在节点的 CPU 和内存（cpu-memory）。
14 列对照来自同机 hsc-12 的 10-02 主采集（584 单元）和修复采集（15 单元），均未绑定 NUMA。
没有使用 hsc-11 的 09-17 时间，没有执行新 benchmark。
两组资源放置不同、采集时间不同，颜色表示绝对时间，不代表同期或统一放置下的性能排名。

- [热调用图](baseline_with_cova_same_process_heatmap.png)
- [新进程首调用图](baseline_with_cova_fresh_process_heatmap.png)
- [初始化图](baseline_with_cova_initialization_heatmap.png)
- [总 JSON](baseline_with_cova_heatmaps_data.json)、[CSV](baseline_with_cova_results.csv)、[合并核对](baseline_with_cova_verification.json)

## 生成规则与历史差异

新脚本 `heatmap/scripts/extract_baseline_collection.py` 接收完整标量提取目录，输出与 09-17 相同的 14 列 JSON 结构。
可通过 `--repair-map` 和 `--replace-map` 自定义显式来源映射，不选择最快批次。
本次使用已归档的 hsc12-baselines/hsc12-repairs；其原始记录 hash 逐个与 NPBench 的只读采集核对。
ADI 整行使用 adi_corrected（不同公式/golden），不复用原 ADI；另外替换 azimint_hist 的 Numba parallel-range、correlation 的 CuPy、mlp 的三条 Numba 路线。
修复集合与 09-17 相同，但本次全部采用 10-02 修复主采集，没有使用 09-17 的冷启动或其他增量数据。
修复来源可能有独立 golden 身份（如 correlation_return、azimint_hist_private），均核对自身严格 golden；不强迫它与已修复的原始失败契约相同。
CoVA 的有值格另与对应 NumPy golden 身份核对。

初始化是主采集进程 0 的单次空产物首调用；09-17 则使用独立冷启动观测，包括修复单元的三次冷启动中位数。
新进程首调用使用后两个进程；hot 先对每进程两次调用取中位数，再对三个进程中位数取中位数，不合并样本挑最快。
14 个 Numba object-mode 单元只有原始进程可用：初始化/hot 保留，fresh 显示 unsupported，标注受限。
其他失败整单元排除计时；缺源为 missing，未注册为 N/A，不用历史成功补值。
对照协议为 6（09-17 的历史主矩阵为 5），新增复用核验在计时区之外；框架版本及来源记录在 JSON。

CoVA 继续使用既有显式 helper 映射与严格 3×3 统计。
54 行、20 列、3 个阶段共 3240 格；对照 2268 格、CoVA 972 格。
对照初始化/hot 各 511 格有值、fresh 497 格有值；CoVA 每阶段 291 格有值。
GPU 列的 CPU/GPU? 仍是原 placement 资格标记，mixed 不等于 CPU 回退，也不是独立主 GPU trace。

## 关键数字核对

热图 LLVM GPU 列（秒）：

| 程序 | hot |
| --- | ---: |
| channel_flow | 2.949713261 |
| compute | 0.757896601 |
| correlation | 1.545394528 |
| covariance | 1.660969657 |
| resnet | 10.894358742 |

全部 324 个 CoVA hot 格的时间和有值/失败状态与绑定全量 `derived/cova-bound.json.gz` 一致。
报告的通用 failed 状态在图中按 error、validation、missing 细分，失败都没有计时。
输入、命令、测试及 hash 在 Research 的 `notebook/01-research/cova/reports/evidence/2026-10-07-npbench-heatmap/`。
heatmap 相关单元测试 13 项通过，原始对照 hash 与合并审计通过；未替换指标。
