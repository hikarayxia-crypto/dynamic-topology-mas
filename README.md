# 动态拓扑多智能体协同仿真

本项目研究动态拓扑下多智能体协同策略的弹性泛化。目前已完成动态拓扑、通信
扰动、连续二维协同搜索环境、规则基线、节点缺失协作补位，以及可训练的图编码
与参数共享 MAPPO 基础实现。短时训练仅用于验证训练链路，正式结论必须通过
后续对照实验获得。

## 当前功能

- 动态增删节点和边、节点失效/恢复、链路断开/重连；
- 邻接表、加权邻接矩阵、邻居和多跳可达性查询；
- 版本化拓扑变化记录和未来事件调度；
- 连续、离散和空动作统一表示；
- 数量可变的邻域观测及拓扑无关均值聚合基线；
- 智能体状态、抽象智能体和多智能体环境生命周期；
- 单播和邻居广播、通信时延、抖动、丢包、数值噪声与消息过期；
- 动态断链对在途消息的真实影响。
- 连续二维运动、边界裁剪、能量消耗、碰撞与连通性统计；
- 距离驱动通信拓扑、定时链路故障和节点故障/恢复；
- 未知目标探测、共享奖励、成功与时间截断条件；
- 不读取未知目标坐标的分带往返搜索规则基线；
- 固定成员搜索带、心跳超时/二次确认/稳定恢复的节点状态机；
- 基于距离、负载、连通度和能量的去中心化竞价与 gossip 一致选择；
- 节点缺失后的搜索带接管、稳定驻留、恢复交还和补位者切换统计；
- 对邻居顺序与数量不敏感的加权图消息聚合；
- 参数共享连续 Actor 和在线智能体全局池化的集中式 Critic；
- GAE、PPO 裁剪目标、熵正则、梯度裁剪及动态在线智能体轨迹；
- JSONL 训练指标和 PyTorch 检查点保存；
- 补位响应时间、覆盖恢复时间、未覆盖搜索带比例、切换次数和补位成功率。

## 环境要求

- Python 3.10 或更高版本；
- NumPy 1.24 或更高版本；
- PyTorch 2.2 或更高版本。
- Matplotlib 3.8 或更高版本。

安装为可编辑包：

```powershell
python -m pip install -e .
```

若 Windows 因安装路径过长而安装 PyTorch 失败，建议在较短目录创建虚拟环境：

```powershell
python -m venv D:\venvs\dynamic-topology-mas
D:\venvs\dynamic-topology-mas\Scripts\python.exe -m pip install -e .
```

## 运行测试

```powershell
python -m unittest discover -s tests -v
```

## 运行动态拓扑演示

```powershell
python scripts/demo_dynamic_topology.py
```

## 运行连续二维协同搜索

```powershell
python scripts/run_simulation.py
```

该快速示例使用 4 个规则智能体、6 个随机目标，并在回合中注入短时链路和节点
故障。输出来自实际仿真，不包含预生成实验结果。

## 运行协作补位快速演示

```powershell
python scripts/run_replacement_demo.py --max-steps 40 --seed 7
```

该演示使用 4 个补位智能体、一个临时节点故障和 10% 配置丢包率，打印实际执行
步数、确认缺失任务数、产生响应/恢复覆盖的任务数和实测丢包率。输出完全来自
本次仿真；脚本不会把预期成功写成结果。

## 运行 Graph-MAPPO 训练

先用小规模参数验证完整训练链路：

```powershell
python scripts/train_mappo.py --updates 1 --rollout-steps 8 --update-epochs 1 --agents 3 --hidden-dim 16 --output-dir outputs/quick_validation
```

执行较长训练：

```powershell
python scripts/train_mappo.py --updates 100 --rollout-steps 256 --update-epochs 4 --agents 4
```

训练指标保存到 `training_metrics.jsonl`，模型与优化器状态保存到
`mappo_checkpoint.pt`，训练曲线保存到 `training_curves.png`。检查点同时保存
网络维度，评估时会优先使用该配置。损失下降本身不等同于任务成功，必须结合
独立验证集的成功率、故障恢复和规模泛化指标评价。

## 运行弹性与规模泛化评估

评估规则基线和协作补位策略：

```powershell
python scripts/evaluate_policies.py --episodes 10 --agent-counts 3,4,6
```

同时加入已保存的学习策略：

```powershell
python scripts/evaluate_policies.py --episodes 10 --agent-counts 3,4,6 --checkpoint outputs/mappo_checkpoint.pt
```

每种规模都会运行无故障场景和链路/节点故障场景，并保存：

- `episode_metrics.jsonl`：逐回合原始指标；
- `scenario_summary.csv/json`：成功率、收敛步数、累计奖励、一致性误差、
  连通率、碰撞、故障恢复率、恢复时间、相对无故障性能下降和补位专项指标；
- `evaluation_summary.png`：不同策略、规模和故障条件的常规对比图；
- `replacement_summary.png`：补位响应、覆盖恢复、未覆盖比例和切换次数专项图。

为保证公平，无故障和故障场景使用相同种子序列；每个回合重新创建环境与智能
体，避免控制器内部状态泄漏。未成功回合的收敛步数、未恢复故障的恢复时间保持
为空，不会以最大步数或零值代替。

协作补位是可解释、确定性的分布式规则基线，不是全局最优分配器，也不等同于
经过训练的鲁棒控制策略。当前 gossip 只能沿当时可用的通信边传播；网络分区时
各连通分量会独立形成局部认知，重连后再通过新消息收敛。因此正式实验应分别
报告丢包、分区时长和拓扑规模，不应只给出单一成功率。

## 主要接口

- `core.topology.DynamicTopology`：动态图和邻接关系；
- `core.action.Action`：统一动作；
- `core.observation.Observation`：自身、邻域和任务观测；
- `core.message.Message`：通信消息；
- `core.agent.BaseAgent`：智能体抽象；
- `core.environment.BaseEnvironment`：环境抽象；
- `interaction.communication.CommunicationBus`：动态拓扑通信总线。
- `environments.continuous_2d.Continuous2DSearchEnv`：连续协同搜索环境；
- `agents.rule_based_agent.RuleBasedSearchAgent`：规则搜索基线；
- `coordination.replacement.ReplacementCoordinator`：缺失确认、竞价、gossip、
  稳定恢复与任务交还；
- `agents.replacement_agent.ReplacementSearchAgent`：执行搜索带补位的规则智能体；
- `algorithms.graph_encoder.GraphObservationEncoder`：可变邻域图编码器；
- `algorithms.shared_actor_critic.SharedGraphActorCritic`：共享 Actor/集中 Critic；
- `training.mappo.MAPPOTrainer`：MAPPO 采样、GAE 与 PPO 更新器；
- `agents.learning_agent.SharedPolicyAgent`：共享策略分散执行封装；
- `evaluation.evaluator`：独立场景、规则/补位/学习策略与规模泛化评估；
- `evaluation.metrics`：协同、连通、补位与故障恢复指标；
- `evaluation.plotting`：训练及评估 PNG 图表。

总体设计与当前实现状态见 `docs/architecture.md`。`legacy/` 仅保存不可运行的
早期原型，不属于正式代码。
