# Generalist Gaming Agent

Kaggle [Generalist Gaming Agent](https://www.kaggle.com/competitions/generalist-gaming-agent) 参赛代码：
一个离线智能体，只看截图和简短说明，实时玩 32 个没见过的浏览器游戏。

规则要点和调研：[docs/competition-brief.md](docs/competition-brief.md)

## 目录

| 路径 | 内容 |
|---|---|
| `agent/agent.py` | 提交入口：`Agent.step(obs, tools) -> Action`，一个实例玩全部游戏 |
| `agent/ggagent/` | 智能体模块：`brief.py` 解析游戏说明，`vision.py` 像素特征，`heuristic.py` 无模型的 CPU 基线策略 |
| `eval/` | 本地评估：`run_local.py` 用官方 harness 跑游戏，`trace_viewer.py` 生成 HTML 轨迹 |
| `notebooks/` | `build_notebook.py` 把 `agent/` 打包成 Kaggle notebook 并推送；`baseline/` 记录 fork 的公开基线 |
| `scripts/setup_data.sh` | 下载比赛数据（放在 `data/`，按规则不进 git） |

## 快速开始

```bash
export KAGGLE_API_TOKEN=...        # 或 KAGGLE_USERNAME / KAGGLE_KEY
scripts/setup_data.sh
python eval/run_local.py --game off_line --minutes 1
# 打开 runs/<最新>/index.html 查看轨迹
```

## 提交

```bash
python notebooks/build_notebook.py --id <kaggle用户名>/gga-ours --push
kaggle kernels status <kaggle用户名>/gga-ours            # 等到 COMPLETE
kaggle competitions submit generalist-gaming-agent -k <kaggle用户名>/gga-ours -v <版本> -f submission.parquet -m "说明"
```

提交后约 6 小时出分。每天最多 5 次提交。
