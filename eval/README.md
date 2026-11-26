# 本地评估

## 现在能做的

```bash
scripts/setup_data.sh                                   # 下载比赛数据到 data/gga（不进 git）
python eval/run_local.py --game off_line --minutes 1    # 跑一个样例游戏
python eval/run_local.py --games public --minutes 2     # 跑 4 个样例游戏
```

每次运行写到 `runs/<时间>_<名字>/`，内容来自官方 `gga.runner`：

- `<game>/steps.jsonl`：每个决策点一行（动作、参数、延迟、`Action.meta`、是否被拒）
- `<game>/frames/NNNN.jpg`：智能体当时看到的画面（`--frames`，默认开启）
- `<game>/video/*.webm`：整局录像；`python -m gga.replay runs/.../<game>`（在 data/gga 下运行）可以把动作叠加到视频上
- `index.html` 和 `<game>/trace.html`：`eval/trace_viewer.py` 生成的可视化轨迹，用浏览器打开

## 下一步：本地计分

数据集**不带计分器**（`gga/readers/__init__.py` 里 `READERS` 是空的），所以本地跑出来 `score=None`。
官方框架支持读取器：`gga.state.Reader` 按 sha256 固定游戏文件，在加载时往游戏 JS 里插入
`globalThis.__ggaGroundTruthRead()`，返回 `{state, value}`。计划：

1. 给 4 个样例游戏写 Reader（读游戏自己的分数变量），在本地复现官方分数。
2. 收集 20–40 个开源（MIT/CC0）HTML5 小游戏，覆盖 10 个类型，作为「隐藏游戏」验证集，各自写 Reader，并定好 floor 和人类参考值。
   起点：讨论区 alexchilton 的 `gga-open-jam-games-test-set` 数据集。
3. `eval/` 加一个汇总脚本：多个智能体在验证集上的平均归一化分数 + 置信区间（`gga.scoring.compare` 可以复用）。

验证集上的分数是私有榜的代理指标；公开榜只作参考（它包含 4 个样例游戏）。
