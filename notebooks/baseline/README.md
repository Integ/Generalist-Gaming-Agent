# 公开基线：Qwen3.8-Flash-Next + 后台 coach + 样例游戏求解器

Kaggle：`adeleascloud/gga-baseline-qwen-coach`（私有），2026-11-26 从
[nihilisticneuralnet/gga-qwen3-8-flash-next-agent](https://www.kaggle.com/code/nihilisticneuralnet/gga-qwen3-8-flash-next-agent)
原样 fork，`USE_CONTROLLERS = False`（作者说明这等同于公开榜 0.339 那一版的行为）。

- 隐藏游戏：start step（代码点掉标题页）→ Qwen3.8-Flash-Next（125B MoE，6B 激活，w4a16，SGLang Pennyroyal fork + MTP 草稿模型）planner/actor，后台 coach 线程。
- 4 个样例游戏：Tong Hui Kang 的脚本像素求解器（只抬高公开榜，私有榜没有样例游戏）。
- GPU：RTX Pro 6000；输入：`dfranzen/pennyroyal-v253` 数据集，`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`、`dfranzen/albucino-qwen3-8-flash-next-drafter` 两个模型。
- 其他人重跑同一方案的公开榜分数在 0.32–0.34 之间。

拉取 notebook：`kaggle kernels pull adeleascloud/gga-baseline-qwen-coach -p notebooks/baseline -m`
（notebook 不进仓库：它很大，而且是第三方代码，以 Kaggle 上的版本为准。）

## 提交记录

| 日期 | 版本 | 公开榜 | 备注 |
|---|---|---|---|
| 2026-11-26 | v1 | 待出分 | 原样 fork，用来跑通流程 |
