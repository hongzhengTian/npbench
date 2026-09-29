# 2026-09-29 当前基线参考图

代码基线为 CoVA `f929db7f0404f03c17fe480f504da8ae3cfdbee5`。
这是09-28全量结果叠加09-29完整产物复核的**混合采样参考快照**，不是一次新的全量认证或同期排名。
原始全量图和数据完整保留在[09-28快照](../2026-09-28/README.md)。

- [热调用图](baseline_with_cova_same_process_heatmap.png)
- [新进程首调用图](baseline_with_cova_fresh_process_heatmap.png)
- [初始化图](baseline_with_cova_initialization_heatmap.png)：数值与09-28逐格相同。
- [总JSON](baseline_with_cova_heatmaps_data.json)、[CSV](baseline_with_cova_results.csv)、[合并审计](baseline_with_cova_verification.json)、[复核标量证据](replay_evidence.json)。

## 选择与统计规则

仅替换九个已完成严格原产物复核单元的fresh和hot，共18格，图中用`R`标明。
各单元均采用新侧全部完整ABBA批次的进程；旧侧只用于归因，不用于当前基线的时间。
每进程先对该阶段调用取中位数，再对进程中位数取中位数；同一输入和当前产物下，每进程等权，不挑选最快批次或样本。
八项各使用两个新进程，每进程一次fresh、两次hot；Mandelbrot helper LLVM CPU合并两轮完整复核，共四进程、四次fresh、14次hot。
较长确认轮不会因为热样本更多而获得更高进程权重。
mandelbrot1 CPU因资源门槛中止的追加批次整轮排除，原完整批次仍保留；其波动及Mandelbrot helper CPU残余差距没有标记为已解决。
因此这里的时间与探查报告中的“合并所有热样本取中位数”略有差别，不能混用。

对照组2268格、全部1080个初始化格、所有功能失败和缺失源码格不变；总共3222格保持原样。
原日志来自L、system、96 CPU、同一golden及host-to-host边界的严格复核；未重新编译，也没有本次新增benchmark运行。
GPU路线保留原CPU/GPU?及mixed资格；前置空闲采样不能证明运行全程独占。
helper保持显式映射，ADI只使用adi_corrected及匹配golden。

## 本次热调用替换（秒）

| benchmark / route | 原全量值 | 当前参考值 | 热样本 / 进程 |
| --- | ---: | ---: | ---: |
| azimint_naive / llvm_cpu | 2.438441 | 2.489222 | 4 / 2 |
| doitgen / llvm_cpu_serial | 0.768945 | 0.633856 | 4 / 2 |
| hdiff / openmp_gpu | 3.130830 | 2.047839 | 4 / 2 |
| lenet / llvm_gpu | 54.795501 | 14.349175 | 4 / 2 |
| mandelbrot1 / llvm_cpu | 0.254498 | 0.187658 | 4 / 2 |
| mandelbrot2 / llvm_cpu | 1.531860 | 1.359445 | 14 / 4 |
| mandelbrot2 / llvm_gpu | 7.548198 | 3.833950 | 4 / 2 |
| vadv / serial_cpu | 13.565297 | 9.563334 | 4 / 2 |
| vadv / llvm_gpu | 20.999509 | 6.836605 | 4 / 2 |

总JSON逐格保存日期、证据索引、输入及当前产物摘要、全部标量样本和进程中位数；`overlay_previous_cells`完整保存被替换格的原记录。
未复核单元仍可能受全量运行时的环境影响，不能将当前参考图解读为全矩阵性能无回退。
完整归因说明见Research长期报告`notebook/01-research/cova/reports/2026-09-26-compute-stencil-access-storage-performance.md`及同级`evidence/2026-09-29-regression-attribution/`。
本目录只含图、文本和标量记录，不含数组、编译缓存或native二进制。

## 重绘（NPBench根目录，不执行benchmark）

```bash
python heatmap/scripts/overlay_cova_replays.py \
  --previous heatmap/2026-09-28/baseline_with_cova_heatmaps_data.json \
  --baseline heatmap/2026-09-17/baseline_heatmaps_data.json \
  --evidence heatmap/2026-09-29-reference/replay_evidence.json \
  --output heatmap/2026-09-29-reference
```
