# CoVA L 2026-09-26 批次

CoVA `275b9806d38b1fd8119ee27bbc86fe06cc1e39a5`，集合 `cova-L-20260926-082805`，09-27整理。
完整54×6矩阵按预定helper映射后291通过、23错误、6数值失败、4缺失源码；原始331注册单元291通过。
L、system、CPU0–95、一张A100X、非top_graph、无profiler、严格3进程×3调用；golden producer为0。

三张PNG分别为初始化、新进程首调和同进程热调用，总JSON和CSV保留状态、标量及来源。
原14条对照的2268个生命周期单元完全不变、未重测；横向结果不能认证同期排名。
CPU表示请求GPU路线报告仅主机执行；GPU?表示元数据没有独立逐单元主GPU trace，mixed保留于JSON。
失败没有历史成功兜底，ADI只匹配adi_corrected。

相对09-25无新增功能失败，但有四项≥10×及十项2–10×热调用变慢，不能只看目标收益推断全矩阵无回退。
分析见Research `notebook/01-research/cova/reports/2026-09-26-compute-stencil-access-storage-performance.md` 的全量复核补充。
独立证据为同一reports下 `evidence/2026-09-27-npbench-cova-L/`。

在NPBench根目录使用已有Python/matplotlib环境重绘，只读证据，不运行benchmark：

```bash
python heatmap/scripts/plot_cova_collection.py \
  --baseline heatmap/2026-09-17/baseline_heatmaps_data.json \
  --evidence /HSC/users/tianhong/myHome/myResearchLife/notebook/01-research/cova/reports/evidence/2026-09-27-npbench-cova-L \
  --output heatmap/2026-09-26
```
