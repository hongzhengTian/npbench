# CoVA L 2026-09-24 批次

CoVA `316658e292d02e63abcfdf443be60bbb58c10d82`，运行09-24启动、09-25完成。
54个科学程序、6条CoVA路线，显式helper映射后289/324通过。
原始注册为331单元：289通过、38失败、4缺失源。
每个通过单元严格3×3调用；system资源策略，CPU0–95，一张A100X，无profiler。
JSON保留三种生命周期、标量样本、输入golden身份、失败及设备声明。

14条对照路线没有重测或更改；与历史对照的横向差异不能认证排名。
历史成功没有填补当前失败；原版ADI没有填补adi_corrected。
GPU?表示没有独立的逐单元主GPU执行trace，不表示运行报错；CPU表示该请求GPU路线报告仅主机执行。
性能与回归归因见Research长期报告 `notebook/01-research/cova/reports/2026-09-24-iteration-private-storage-performance.md` 的全量L复核章节。
精炼证据为同一reports下 `evidence/2026-09-25-npbench-cova-L`。
