# NPBench 全量结果快照

全量运行及后续明确标注的增量更新，其三张heatmap和总JSON按日期保存。
日期是批次标识，不表示所有历史对照组都在当天运行。

| 目录 | 内容 |
| --- | --- |
| [2026-09-17](2026-09-17/baseline_heatmaps_data.json) | 原14条对照路线，保留原始JSON和三张图，字节不变。 |
| [2026-09-22](2026-09-22/baseline_with_cova_heatmaps_data.json) | 原先根目录的合并视图：历史全量CoVA、覆盖闭合及最后azimint专项观测，混合版本。 |
| [2026-09-24](2026-09-24/baseline_with_cova_heatmaps_data.json) | 09-24启动、09-25完成的CoVA完整L矩阵；只替换CoVA六列，原对照组不变。 |
| [2026-09-25](2026-09-25/baseline_with_cova_heatmaps_data.json) | 09-24底稿加九个修复目标单元的严格3×3复测；混合版本增量视图，其他单元不变。 |

最新热调用图：[same_process](2026-09-25/baseline_with_cova_same_process_heatmap.png)。
另有[新进程首调用](2026-09-25/baseline_with_cova_fresh_process_heatmap.png)和[初始化](2026-09-25/baseline_with_cova_initialization_heatmap.png)。
失败不保留历史成功时间，helper显式映射，ADI只采用adi_corrected公式和golden。
颜色表示绝对时间，不是同期排名；GPU路线名称不保证主计算在GPU执行，查看CPU/GPU?标注与JSON中的reported_devices。

## 保存下一次完整CoVA运行

从NPBench根目录，在已有Python/matplotlib环境中执行。
以下命令只读取已完成运行，不执行benchmark；证据目录必须尚不存在。
将 `RUN`、`EVIDENCE`、`SNAPSHOT` 替换为具体路径。

```bash
python heatmap/scripts/extract_cova_collection.py --run-dir RUN --output EVIDENCE
python heatmap/scripts/plot_cova_collection.py \
  --baseline heatmap/2026-09-17/baseline_heatmaps_data.json \
  --evidence EVIDENCE --output heatmap/SNAPSHOT
python -m unittest discover -s tests -p 'test_cova_heatmap*.py' -v
```

提取器面向当前完整L、3进程×3调用协议，多次attempt不会自动选择最快一批，而是要求显式整理。
先保留每个注册单元的精炼标量、身份、失败、artifact摘要和资源，再用该独立证据渲染。
绘图必须使用原14列baseline JSON，不能把已合并JSON作为baseline再追加CoVA。
重新生成本次图可使用Research长期证据 `notebook/01-research/cova/reports/evidence/2026-09-25-npbench-cova-L`。
新快照保留完整JSON和三张PNG，不归档数组、native二进制或编译缓存。

旧绘图入口已移动至 `heatmap/scripts/plot_baseline_heatmaps.py` 和 `heatmap/scripts/plot_baseline_with_cova.py`。
历史选择器用于重建旧混合视图，新完整collection使用 `plot_cova_collection.py`。
历史文件迁移的SHA256对应关系见[migration.json](migration.json)。

部分单元复测可使用 `overlay_cova_repairs.py`，显式选择已完成严格3×3的目标单元；命令和限制见[09-25说明](2026-09-25/README.md)。
