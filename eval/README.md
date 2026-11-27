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

## 本地计分

数据集**不带计分器**（`gga/readers/__init__.py` 里 `READERS` 是空的）。官方框架支持读取器：
`gga.state.Reader` 按 sha256 固定游戏文件，在加载时往游戏 JS 里插入
`globalThis.__ggaGroundTruthRead()`，返回 `{state, value}`。

我们的读取器在 `eval/readers/`。`eval/harness.py` 在 `build/harness/` 组装一个本地 harness：
数据集文件用软链接，`gga/readers` 换成我们的包，数据集本身不改。`run_local.py` 默认使用它
（`--no-readers` 关闭）。本地分数是估计值：官方读取器和比赛里的游戏修正可能不同。

| 游戏 | 状态 |
|---|---|
| off_line | ✅ 读屏幕上的计数 `floor(score/10)` |
| blarglike | ✅ GameMaker 全局变量 `floor × nutrients` |
| asteroid_capitalist | ✅ 打包模块里的 `x.score`（美元） |
| cavebots | 待做（Ebitengine WASM，读不到 JS 变量，可能要用像素读数） |

验证（45 秒，用公开的样例游戏脚本求解器跑）：asteroid_capitalist 0.148，blarglike 0.071，
off_line 用我们的 CPU 基线 0.059。分数随时间上涨，和 10 分钟接近 1.0 的说法一致。

后续：

1. 补上 cavebots 的读取器。
2. 收集 20–40 个开源（MIT/CC0）HTML5 小游戏，覆盖 10 个类型，作为「隐藏游戏」验证集，各自写 Reader，并定好 floor 和人类参考值。
   起点：讨论区 alexchilton 的 `gga-open-jam-games-test-set` 数据集。
3. `eval/` 加一个汇总脚本：多个智能体在验证集上的平均归一化分数 + 置信区间（`gga.scoring.compare` 可以复用）。

验证集上的分数是私有榜的代理指标；公开榜只作参考（它包含 4 个样例游戏）。

## 开源小游戏测试集（私有榜的替代指标）

私有榜的 16 个游戏看不到，所以用 agent 从没调过的游戏衡量泛化能力：alexchilton 在 Kaggle 上整理的
54 个 js13kGames 小游戏（`alexchilton/gga-open-jam-games-test-set`，大多 MIT，每个游戏目录带自己的
`LICENSE.repo`）。游戏文件不进 git，用脚本下载：

```bash
scripts/setup_data.sh && scripts/setup_jam.sh     # 数据放到 data/gga、data/jam
python eval/jam.py --list                         # 默认的 24 个游戏
python eval/jam.py --jobs 3                       # agent/agent.py，每个游戏 2 分钟
python eval/jam.py --games roads,pulse --minutes 1 --env GGA_MODEL=mock --tag quick
```

- 默认只算有非平凡计分的游戏，去掉也是比赛游戏的 off_line（共 24 个）。
- 归一化和官方一样：clip((最好值 - idle) / (reference - idle), 0, 1)。reference 是随机或固定按键
  策略 2 分钟里的最好成绩，**不是人类成绩**，所以 1.0 只代表"和随机按键一样好"，不能和榜上分数直接比。
- 数据集里的 Python 脚本不运行：`eval/jam.py` 把 `games_extra.json` 当数据读，并用 `ast` 解析
  `tools/extra_games.py` 里的 `_reader(...)` 调用拿到计分器（不执行），然后在 `build/jam/` 组装 harness。
- 结果在 `runs/jam/<tag>/`：每个游戏一个 json，`summary.json` 是汇总。
- 容器里渲染只有实时的 0.3–0.4 倍，24 个游戏 `--jobs 3` 大约要 1 小时。

CPU 基线（2026-11-27，`agent/agent.py` 不带规划器，2 分钟/游戏，`--jobs 3` 用时 25 分钟）：**0.510**。
得 0 分的 7 个：lovely_drive、falling_down、bridge_the_gap、hyperspace、captain_katamov、
stay_off_the_line、sopwith_back，是下一步改进的重点。
