# Generalist Gaming Agent 比赛调研（2026-11-26，通过 Kaggle API 获取）

原始材料：比赛数据集（`scripts/setup_data.sh` 下载到 `data/gga/`）里的 `gga/protocol.py`、`games.json`、`submission/agent.py`

## 1. 核实结果（对照之前的二手信息）

| 之前的说法 | 官方结论 |
|---|---|
| 32 个没见过的浏览器游戏 | ✅ 32 个游戏、10 个类型，每个玩 10 分钟游戏时间。公开 4 个样例游戏，其余 28 个不分发 |
| `Agent.step(obs, tools)`，11 个工具 | ✅ click/move/drag/scroll/press_key/hold_keys/key_down/key_up/wait/sequence/skip |
| 观测是截图 + 游戏简介 | ✅ 另外还有 frames（上次决策后的帧）、history（最多 8 帧）、held、program、last_action |
| 离线 CPU notebook，约 9 小时 | ⚠️ 部分正确：CPU 或 **GPU（RTX Pro 6000 96GB）** 都可以，各 ≤9 小时，禁网 |
| 相对人类基线的归一化分数 | ✅ 每个游戏 clip((value−floor)/(human−floor), 0, 1)，32 个游戏全跑 |
| 奖金 5 万美元 | ✅ $16k / $12k / $9k / $7k / $6k |

## 2. 规则要点

- **时间线**：2026-11-23 开赛，**2027-03-01 23:59 UTC** 截止。报名/组队截止日期页面上还没填。
- **排行榜**：公开榜 = 16 个游戏的平均（**含 4 个样例游戏**）；私有榜 = 另外 16 个（**不含任何样例游戏**），决定最终名次。可选 2 个最终提交。
- **提交**：每天 5 次；队伍最多 5 人；提交后约 6 小时出分（32×10 分钟实时游戏）。提交不消耗每周 GPU 配额。
- **时限**：启动（安装、起模型服务、`Agent.__init__`）≤ **15 分钟**；单次 `step` ≤ **90 秒**，超时则该游戏直接结束；异常或无效回复按 `skip` 处理。
- **外部数据/模型**：允许免费公开的预训练模型和数据。获奖需以 MIT 协议开源。
- **数据保密**：比赛数据不得分发给非参赛者（所以 repo 里不要直接提交 assets）。

## 3. 接口与时序（最影响设计的部分）

- 游戏以 **20 fps** 运行，每 4 帧（**200 ms**）一个决策点。回复在到达后的第一个决策点生效；慢回复不会丢，只是更晚生效。
- 除 `skip` 外，每个回复都会**替换**正在运行的输入程序。所以模型思考时应回 `skip`，或者用 `sequence`（最多 200 步、≤10 秒的定时输入程序）提前排好动作。
- 程序的最后状态会保持 10 秒（watchdog），然后全部释放。最多同时按 8 个键。
- 一个 Agent 实例连续玩 32 个游戏：`obs.elapsed_s` 回到接近 0 就是新游戏，要重置状态。
- 只有 4 个样例游戏会提供 `game_id/title/genre`；隐藏游戏只有像素 + `description` + `controls`。
- 坐标就是 viewport 像素（默认 1600×1000，各游戏可能不同，看 `obs.info.width/height`）。
- 游戏本身确定性：同样时刻同样动作得分相同；波动来自智能体（采样、延迟），同一 notebook 重跑约 ±0.01。
- 本地测试：`GGA_HEADLESS=1 python -m gga.game --game off_line --agent path/to/agent.py --minutes 1`（Python ≥3.11，playwright==1.60.0）。**本地不出分**，数据集没有计分器。

### 4 个样例游戏

| game_id | 类型 | 计分 | 人类基线 |
|---|---|---|---|
| asteroid_capitalist | Action 射击 | 击杀赚的美元 | 5760 |
| blarglike | Adventure 回合制地牢 | 层数 × 养分 | 902（floor 1）|
| cavebots | Strategy 塔防/挖掘 | 核心存活秒数 | 600 |
| off_line | Racing 沿线躲避 | 在线上的时间 | 781 |

## 4. 排行榜与公开 notebook 现状（2026-11-26）

- 参赛 395 队。公开榜第 1 名 **0.408**（Keith Tyser），之后是 0.358、0.349，0.336–0.345 一大堆（基本都是同一套公开方案的复制）。
- 官方 starter：Gemma 4 31B（transformers）约 **0.09**；Drex Max 决策模型（Nace.AI，255 个离散动作单次前向）。
- **主流高分方案（0.32–0.34）**= 三件套：
  1. Tong Hui Kang 的 **4 个样例游戏脚本求解器**（纯像素 numpy，接近 1.0 分）；
  2. alexchilton 的 **start step**（用代码点掉标题页/菜单）；
  3. **Qwen3.8-Flash-Next**（125B MoE，6B 激活，w4a16）+ SGLang（Pennyroyal fork）+ MTP 草稿模型，planner/actor 结构，**后台 coach** 线程（一次只发一个请求，超过 12–15 秒的回复丢弃）。
- 最新版本在试 **模型写控制器**（coach 输出一个 Python `step(frame,t,mem)`，在独立进程每 200ms 运行，参考 AutoPILOT 论文）。
- 只用 CPU 的「专用控制器 + 自适应探索」方案 0.215；Gemma 后台规划 + CPU 反射层 0.304。

### 关键结论：公开榜严重被样例游戏抬高

样例游戏占公开榜 4/16。脚本求解器拿满就是 +0.25，而私有榜一个样例游戏都没有。
按 Tong Hui Kang 的估算：Qwen 在隐藏游戏上平均约 **0.12**，12/16×0.12 + 4/16×1.0 ≈ 0.34。
由此推算：第一名 0.408 在 12 个隐藏游戏上大约 0.21；0.215 的 CPU 方案在隐藏游戏上接近 0。
**真正决定名次的是通用智能体在没见过的游戏上的分数，目前大家都在 0.1–0.2 之间。**

## 5. 讨论区要点

- hengck23：先做 **tracer**（记录并可视化轨迹）、调试时打印 COT，让 AI 读 trace 日志找问题；提出「假设 → 验证」式 prompt。
- alexchilton 做了一个 **开源 jam 游戏测试集**（dataset `alexchilton/gga-open-jam-games-test-set` + runner notebook），用来本地评估泛化，避免浪费提交次数。建议去 itch.io 找 MIT/开源 HTML5 游戏扩充。
- Tong Hui Kang 把 4 个公开游戏放到了网页上：https://tonghuikang.github.io/generalist-gaming-agents/
- 论文 **AutoPILOT: A Visual Agent Harness for Real-Time Gameplay**（https://autopilot-harness.github.io/），思路和本赛题完全一致。
- 有官方 Discord（见讨论帖 760722）。

## 6. 起步方案建议

**原则：用私有榜思维做决策，公开榜只作参考。**

1. **第 0 周：先复现基线（1–2 天）**
   - Fork 当前最好的公开方案（ozertuu 0.339 / huikang public solver），提交一次，确认整套流程（模型挂载、15 分钟启动、出分）能跑通。
   - 保留样例游戏求解器（公开榜上有用，私有榜无害）。
2. **搭本地评估框架（最重要的基础设施）**
   - 本地没有计分器：给开源测试游戏写分数读取器（读 DOM/JS 变量），或者用帧差/OCR 估分。
   - 收集 20–40 个开源 HTML5 小游戏，覆盖 10 个类型，作为私有的「隐藏游戏」验证集。
   - tracer：每局保存帧、动作、延迟、模型输出，生成可回放的 HTML。
3. **把精力投在通用智能体上（私有榜的全部来源）**，按预期收益排序：
   - **开局与菜单处理**：大量隐藏游戏可能卡在标题页、Game Over 后不会重开。规则化处理「点 Play / 按 Space / Enter / 读 controls 里的指令」。成本低、收益大。
   - **快慢双系统**：慢的 VLM 在后台理解规则、出策略；快的 CPU 层每 200ms 行动（可以由 VLM 写出的控制器，或者参数化反射策略：躲避、追踪、持续开火）。
   - **从 brief 中抽取控制方式**：解析 `controls`，限制动作空间，减少无效探索。
   - **卡住检测与恢复**：画面不变、得分不变时换策略。
   - **跨局记忆**：同一局 10 分钟里多次死亡，要积累 `LEARNED` 笔记。
4. **模型选择**：Qwen3.8-Flash-Next w4a16 + SGLang 是当前最快的公开可用栈（RTX Pro 6000 96GB 上能跑）。之后可以评估更小更快的 VLM 专门做 actor。

### 需要你决定
- 是否同意按上面的顺序开始：先 fork 公开最佳 notebook 提交一次，同时在代码库里搭本地评估框架和 tracer？
- 代码库结构建议：`agent/`（智能体代码）、`eval/`（本地 runner、测试游戏、计分器）、`notebooks/`（提交用 notebook 生成脚本）。比赛数据不进仓库。
