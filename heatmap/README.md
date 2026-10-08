# NPBench 全量结果快照

全量运行及后续明确标注的增量更新，其三张heatmap和总JSON按日期保存。
日期是批次标识，不表示所有历史对照组都在当天运行。

| 目录 | 内容 |
| --- | --- |
| [2026-09-17](2026-09-17/baseline_heatmaps_data.json) | 原14条对照路线，保留原始JSON和三张图，字节不变。 |
| [2026-09-22](2026-09-22/baseline_with_cova_heatmaps_data.json) | 原先根目录的合并视图：历史全量CoVA、覆盖闭合及最后azimint专项观测，混合版本。 |
| [2026-09-24](2026-09-24/baseline_with_cova_heatmaps_data.json) | 09-24启动、09-25完成的CoVA完整L矩阵；只替换CoVA六列，原对照组不变。 |
| [2026-09-25](2026-09-25/baseline_with_cova_heatmaps_data.json) | 09-24底稿加九个修复目标单元的严格3×3复测；混合版本增量视图，其他单元不变。 |
| [2026-09-26](2026-09-26/baseline_with_cova_heatmaps_data.json) | 275b9806完整L矩阵，09-27整理；对照不变，291/324通过，完整呈现新增性能回退。 |
| [2026-09-28](2026-09-28/baseline_with_cova_heatmaps_data.json) | f929db7f完整L矩阵，09-29完成；291/324通过，收尾存在GPU竞争，无10×热回退，保留12项≥2×观测回退。 |
| [2026-09-29-reference](2026-09-29-reference/README.md) | 当前f929db7f基线参考图：09-28底稿加九项完整产物复核，更新18个fresh/hot格；混合采样，R标注，对照与初始化不变。 |
| [2026-10-07](2026-10-07/README.md) | hsc-12 CoVA 7eb28298 完整 L、GPU CPU/内存绑定；对照改用同机 10-02 未绑定主采集及修复。 |

最新完整图：[same_process](2026-10-07/baseline_with_cova_same_process_heatmap.png)、[fresh_process](2026-10-07/baseline_with_cova_fresh_process_heatmap.png)、[初始化](2026-10-07/baseline_with_cova_initialization_heatmap.png)。
本次两组均在 hsc-12，CoVA GPU 为 cpu-memory 绑定，对照为 10-02 未绑定；口径与历史差异见[说明](2026-10-07/README.md)。
历史基线参考图仍保留在[09-29-reference](2026-09-29-reference/README.md)，原始全量保留在[09-28](2026-09-28/README.md)。

失败不保留历史成功时间，helper显式映射，ADI只采用adi_corrected公式和golden。
颜色表示绝对时间，不是同期排名；GPU路线名称不保证主计算在GPU执行，查看CPU/GPU?标注与JSON中的reported_devices。

## 保存下一次完整CoVA运行

从NPBench根目录，在已有Python/matplotlib环境中执行。
以下命令只读取已完成运行，不执行benchmark；证据目录必须尚不存在。
将 `RUN`、`EVIDENCE`、`SNAPSHOT` 替换为具体路径。

```bash
python heatmap/scripts/extract_cova_collection.py --run-dir RUN --output EVIDENCE
python heatmap/scripts/plot_cova_collection.py \
  --baseline BASELINE_JSON \
  --evidence EVIDENCE --output heatmap/SNAPSHOT
python -m unittest discover -s tests -p 'test_cova_heatmap*.py' -v
```

提取器面向当前完整L、3进程×3调用协议，多次attempt不会自动选择最快一批，而是要求显式整理。
先保留每个注册单元的精炼标量、身份、失败、artifact摘要和资源，再用该独立证据渲染。
绘图必须使用同机、冻结的14列baseline JSON，不能把已合并JSON作为baseline再追加CoVA。
重新生成最新图使用 Research 证据 `notebook/01-research/cova/reports/evidence/2026-10-07-npbench-heatmap` 的 baseline_heatmaps_data.json 和 cova/。
新快照保留完整JSON和三张PNG，不归档数组、native二进制或编译缓存。

旧绘图入口已移动至 `heatmap/scripts/plot_baseline_heatmaps.py` 和 `heatmap/scripts/plot_baseline_with_cova.py`。
历史选择器用于重建旧混合视图，新完整collection使用 `plot_cova_collection.py`。
历史文件迁移的SHA256对应关系见[migration.json](migration.json)。

部分单元复测可使用 `overlay_cova_repairs.py`，显式选择已完成严格3×3的目标单元；命令和限制见[09-25说明](2026-09-25/README.md)。

## 生成新的同机对照组输入

`extract_baseline_collection.py` 读取已提取的主矩阵和修复矩阵，不运行 benchmark。
两者必须来自同一机器、L/system、未绑定 NUMA；缺源单元继承冻结 collection 的机器身份。
初始化保留该采集的单次首调用，不自动引入历史独立冷启动值。
默认沿用已有修复集合，可用 JSON 参数 `--repair-map`、`--replace-map` 改成另一个明确的来源选择。

```bash
python heatmap/scripts/extract_baseline_collection.py \
  --main MAIN_SCALAR_EVIDENCE --repairs REPAIR_SCALAR_EVIDENCE \
  --dataset-id DATASET_ID --output BASELINE_JSON
```

绘图器在双方均有 host 身份时拒绝跨机器合并，并保留 GPU NUMA 策略到 JSON 和图注。
新图的完整命令保存在 Research 新证据目录的 tools/generate.sh。
