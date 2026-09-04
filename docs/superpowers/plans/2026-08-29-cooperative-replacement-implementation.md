# Cooperative Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有动态拓扑协同搜索系统中实现基于心跳、消息传播和确定性竞价的节点缺失协作补位算法，并提供鲁棒控制、评价指标和真实演示。

**Architecture:** 每个 `ReplacementSearchAgent` 持有独立的 `ReplacementCoordinator`，只通过现有通信总线交换心跳和竞价摘要；环境不决定赢家，只验证动作元数据和真实位置。固定搜索带、缺失状态机、竞价与运动控制分层实现，确保纯协调算法可以脱离环境单测。

**Tech Stack:** Python 3.10+、NumPy 1.24+、现有 `unittest`、现有动态拓扑和 `CommunicationBus`；不新增第三方依赖。

**Spec:** `docs/superpowers/specs/2026-08-29-cooperative-replacement-design.md`

## Global Constraints

- 不读取未知目标坐标，不在观测或协调消息中暴露目标位置。
- 环境只记录补位行为，不直接指定或修改竞价赢家。
- 不改变 `RuleBasedSearchAgent`、`Observation`、`Action` 和 `DynamicTopology` 的现有公开行为。
- 新增重要类、函数、状态机、竞价公式和指标必须有准确中文注释。
- 采用严格红—绿—重构；每项生产行为必须先看到对应测试因缺失功能而失败。
- 快速仿真只验证流程，不得描述为正式算法效果。
- 当前目录不是 Git 仓库，不执行或伪造提交；每个任务以指定测试命令通过作为检查点。

---

## File Structure

- Create `coordination/__init__.py`：导出补位协调公开类型。
- Create `coordination/replacement.py`：配置、搜索带、消息、存活状态、竞价和分配状态机。
- Create `agents/replacement_agent.py`：消息驱动的分布式补位规则智能体。
- Modify `agents/rule_based_agent.py`：仅提取可复用扫描和避碰计算，保持旧策略输出一致。
- Modify `agents/__init__.py`：导出 `ReplacementSearchAgent`。
- Modify `environments/continuous_2d.py`：从动作元数据计算实际补位指标。
- Modify `evaluation/metrics.py`：汇总响应时间、覆盖恢复和切换指标。
- Modify `evaluation/evaluator.py`：增加 `replacement` 策略评估。
- Modify `evaluation/plotting.py`：增加补位专项对比图。
- Modify `evaluation/reporting.py`：沿用通用序列化，覆盖新增字段。
- Create `scripts/run_replacement_demo.py`：节点故障、丢包和恢复的最小真实演示。
- Modify `scripts/evaluate_policies.py`：规则、补位、学习策略统一对照。
- Create `tests/test_replacement_coordination.py`：纯状态机和竞价单元测试。
- Create `tests/test_replacement_agent.py`：消息、运动和交还测试。
- Modify `tests/test_continuous_2d.py`：环境补位指标集成测试。
- Modify `tests/test_evaluation.py`：补位指标和策略汇总测试。
- Modify `pyproject.toml`：把 `coordination` 加入包清单。
- Modify `README.md`、`docs/architecture.md`：记录功能、命令和限制。

---

### Task 1: 补位配置、固定搜索带和数据模型

**Files:**
- Create: `coordination/__init__.py`
- Create: `coordination/replacement.py`
- Create: `tests/test_replacement_coordination.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `ReplacementConfig`, `CoverageLane`, `ReplacementBid`, `ReplacementAssignment`, `NodeLiveness`, `build_coverage_lanes(roster, world_height)`。
- Consumes: Python `dataclasses`、`Enum` 和有限数值校验。

- [ ] **Step 1: 写搜索带和配置失败测试**

```python
class ReplacementCoordinationTests(unittest.TestCase):
    def test_fixed_lanes_follow_stable_roster_order(self) -> None:
        lanes = build_coverage_lanes(("C", "A", "B"), 12.0)
        self.assertEqual([lane.owner_id for lane in lanes], ["A", "B", "C"])
        self.assertEqual([lane.center_y for lane in lanes], [2.0, 6.0, 10.0])

    def test_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            ReplacementConfig(
                distance_weight=0.5,
                load_weight=0.2,
                connectivity_weight=0.2,
                energy_weight=0.2,
            )
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run:

```powershell
python -m unittest tests.test_replacement_coordination -v
```

Expected: import error，指出 `coordination` 或上述类型尚不存在。

- [ ] **Step 3: 实现最小数据模型**

```python
@dataclass(frozen=True)
class ReplacementConfig:
    failure_timeout: float = 1.0
    failure_confirmation: float = 0.6
    bid_window: float = 0.4
    recovery_stability: float = 1.0
    broadcast_interval: float = 0.2
    bid_ttl: float = 2.0
    switch_margin: float = 0.05
    lane_tolerance: float = 1.0
    dwell_steps: int = 20
    distance_weight: float = 0.55
    load_weight: float = 0.20
    connectivity_weight: float = 0.15
    energy_weight: float = 0.10

@dataclass(frozen=True)
class CoverageLane:
    lane_id: str
    owner_id: str
    center_y: float

class NodeLiveness(str, Enum):
    HEALTHY = "healthy"
    SUSPECTED = "suspected"
    MISSING = "missing"
    RECOVERING = "recovering"
```

`build_coverage_lanes` 必须对名单按 `str(agent_id)` 稳定排序、拒绝空名单和非正
区域高度，并按规格公式生成中心纵坐标。配置必须校验时间、容差、权重有限非负，
`dwell_steps` 为正整数，四个权重和在 `1e-9` 容差内等于 1。

- [ ] **Step 4: 运行目标测试**

Expected: 两项测试通过。

- [ ] **Step 5: 将 `coordination` 加入包清单并运行完整测试**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: 原41项测试和新增测试全部通过。

---

### Task 2: 心跳、缺失确认和恢复状态机

**Files:**
- Modify: `coordination/replacement.py`
- Modify: `tests/test_replacement_coordination.py`

**Interfaces:**
- Produces: `ReplacementCoordinator.__init__`, `advance_time`, `build_gossip`, `ingest_gossip`, `missing_nodes`, `liveness_of`。
- Consumes: Task 1 的配置、搜索带和存活枚举。

- [ ] **Step 1: 写心跳超时测试**

```python
def test_timeout_requires_confirmation_before_missing(self) -> None:
    coordinator = ReplacementCoordinator("A", ("A", "B"), 10.0, ReplacementConfig())
    coordinator.ingest_gossip({
        "kind": "replacement_gossip",
        "sender": "B",
        "sent_at": 0.0,
        "heartbeats": {"B": 0.0},
        "bids": {},
    }, received_at=0.0)
    coordinator.advance_time(1.1)
    self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.SUSPECTED)
    coordinator.advance_time(1.7)
    self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.MISSING)
```

- [ ] **Step 2: 运行并确认失败原因是 `ReplacementCoordinator` 尚未实现**

- [ ] **Step 3: 实现本地心跳表和状态转换**

协调器保存：

```python
self._last_heartbeats: dict[Hashable, float]
self._suspected_since: dict[Hashable, float]
self._recovering_since: dict[Hashable, float]
self._liveness: dict[Hashable, NodeLiveness]
self._last_time: float
```

`advance_time` 只能前进。超过 `failure_timeout` 进入 `SUSPECTED`，持续达到
`failure_confirmation` 进入 `MISSING`；`MISSING` 后收到更新心跳进入
`RECOVERING`，持续收到未超时心跳达到 `recovery_stability` 才回到 `HEALTHY`。

- [ ] **Step 4: 写并运行短暂丢包不触发缺失测试**

```python
def test_short_heartbeat_gap_does_not_create_missing_task(self) -> None:
    coordinator = self._coordinator()
    coordinator.advance_time(1.1)
    coordinator.ingest_gossip(self._heartbeat("B", sent_at=1.2), received_at=1.2)
    coordinator.advance_time(1.3)
    self.assertEqual(coordinator.missing_nodes, ())
```

先确认测试在恢复处理缺失时失败，再实现 `SUSPECTED -> HEALTHY`。

- [ ] **Step 5: 写并运行稳定恢复测试**

```python
def test_recovered_node_must_remain_stable_before_handover(self) -> None:
    coordinator = self._missing_b()
    coordinator.ingest_gossip(self._heartbeat("B", 2.0), received_at=2.0)
    self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.RECOVERING)
    coordinator.ingest_gossip(self._heartbeat("B", 2.8), received_at=2.8)
    coordinator.advance_time(3.0)
    self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.HEALTHY)
```

- [ ] **Step 6: 写并运行消息校验测试**

手工验证未知发送者、未来时间戳、非有限心跳和错误 `kind` 均返回 `False`，且
`missing_nodes` 与心跳表不变；合法消息返回 `True`。

- [ ] **Step 7: 运行全部测试作为检查点**

---

### Task 3: 代价竞价、传播合并和稳定分配

**Files:**
- Modify: `coordination/replacement.py`
- Modify: `tests/test_replacement_coordination.py`

**Interfaces:**
- Produces: `update_local_status(position_y, energy, neighbor_count, timestamp)`, `local_bid_for`, `assignments`, `assignment_for`, `known_bids`。
- Consumes: Task 2 的缺失节点集合和 gossip 协议。

- [ ] **Step 1: 写手工代价测试**

```python
def test_bid_score_uses_distance_load_connectivity_and_energy(self) -> None:
    config = ReplacementConfig()
    coordinator = ReplacementCoordinator("A", ("A", "B"), 10.0, config)
    coordinator.update_local_status(
        position_y=4.5, energy=0.8, neighbor_count=1, timestamp=2.0
    )
    bid = coordinator.local_bid_for("B", current_load=1, timestamp=2.0)
    expected = 0.55 * 0.3 + 0.20 * 1.0 + 0.15 * 0.5 + 0.10 * 0.2
    self.assertAlmostEqual(bid.score, expected)
```

该期望值独立手算：两节点时B搜索带中心为7.5，纵向距离5.5；测试位置改为4.5
使归一化距离严格为0.3，避免期望值与实现共享公式。

- [ ] **Step 2: 运行并确认因竞价API缺失而失败**

- [ ] **Step 3: 实现代价计算和有限值校验**

实现规格中的四项加权和。能量裁剪到 `[0, 1]`；邻居数不得为负；负载按当前
有效分配数量传入。`ReplacementBid` 保存 `created_at` 和
`expires_at=created_at+bid_ttl`。

- [ ] **Step 4: 写同分稳定裁决测试**

```python
def test_equal_scores_choose_lexicographically_smaller_bidder(self) -> None:
    coordinator = self._missing_c(self_id="A", roster=("A", "B", "C"))
    coordinator.ingest_gossip(self._gossip_with_bid("C", "B", 0.4), 2.0)
    coordinator.ingest_gossip(self._gossip_with_bid("C", "A", 0.4), 2.0)
    coordinator.advance_time(2.5)
    self.assertEqual(coordinator.assignment_for("C").assignee_id, "A")
```

- [ ] **Step 5: 实现竞价合并、过期清理和 bid window**

同一缺失节点保留排序键 `(score, str(bidder_id))` 最小的未过期竞价。未经过
`bid_window` 不生成 `ReplacementAssignment`。赢家失联或竞价过期后重新竞价。

- [ ] **Step 6: 写多任务负载均衡测试**

构造A、B在线，C、D缺失；第一次C由A获胜后，D竞价中A负载为1、B为0，手工
设置位置使B总分更低，断言两个任务由不同节点接管。

- [ ] **Step 7: 写多跳传播和坏消息测试**

构造A接收B转发的C心跳和D竞价，断言合法的更新进入下一次 `build_gossip`；
断言过期、`NaN`、未知竞价者和未知缺失节点不会进入 `known_bids`。

- [ ] **Step 8: 写 switch margin 测试并实现稳定切换**

现有赢家分数0.50，新竞价0.48且 `switch_margin=0.05` 时保持原赢家；新竞价
0.44时切换并增加协调器内部切换计数。

- [ ] **Step 9: 运行协调模块与完整测试**

---

### Task 4: 可复用扫描控制与补位智能体

**Files:**
- Modify: `agents/rule_based_agent.py`
- Create: `agents/replacement_agent.py`
- Modify: `agents/__init__.py`
- Create: `tests/test_replacement_agent.py`

**Interfaces:**
- Produces: `ReplacementSearchAgent`；`RuleBasedSearchAgent._sweep_command(observation, dt, target_y=None)`。
- Consumes: `ReplacementCoordinator`、`BaseAgent`消息收件箱和现有动作接口。

- [ ] **Step 1: 写原规则策略行为刻画测试**

固定观测、内部方向和通过边界条件，记录重构前动作的手工期望方向：目标在右上
时两个动作分量均为正且范数为1；目标在右下时横向为正、纵向为负。该测试保护
公开行为，不断言私有代码结构。

- [ ] **Step 2: 提取 `_sweep_command` 并确认原测试全部通过**

方法签名：

```python
def _sweep_command(
    self,
    observation: Observation,
    dt: float,
    *,
    target_y: float | None = None,
) -> np.ndarray:
    """计算搜索带扫描与邻居排斥后的单位速度指令。"""
```

`target_y=None` 完全沿用旧搜索带公式；传值时只替换纵向目标，不改变边界折返和
邻居排斥。

- [ ] **Step 3: 写补位动作失败测试**

```python
def test_winning_agent_moves_toward_missing_lane(self) -> None:
    agent = self._replacement_agent("A", roster=("A", "B"))
    observation = self._observation(agent_id="A", y=1.0, timestamp=3.0)
    self._make_assignment(agent.coordinator, missing="B", winner="A")
    action = agent.step(observation, 0.2)
    self.assertGreater(action.value[1], 0.0)
    self.assertEqual(action.metadata["replacement_for"], "B")
    self.assertEqual(action.metadata["replacement_lane_y"], 7.5)
```

- [ ] **Step 4: 实现消息消费、广播和补位目标选择**

`perceive` 消费 `pop_messages()` 中 `replacement_gossip`，再用当前位置、能量和
邻居数更新协调器。`decide` 按广播间隔调用 `send_message(None, payload)`；发送
失败不终止动作。赢得任务时调用 `_sweep_command(..., target_y=lane.center_y)`。

- [ ] **Step 5: 写多任务驻留轮转测试**

固定 `dwell_steps=2` 和两个本节点获胜任务，连续四次决策的
`replacement_for` 应为 `C,C,D,D`，而不是每步切换或永久停在C。

- [ ] **Step 6: 写恢复平滑交还测试**

节点B在当前两步驻留周期第1步恢复时仍输出 `replacement_for="B"`；周期结束后
下一步动作不再包含该字段，且恢复自己的固定搜索带扫描。

- [ ] **Step 7: 写分区降级测试**

无邻居、广播被拒绝时，智能体仍产生有效二维动作；已有本地缺失任务时允许
本地自竞价和接管，不抛出异常。

- [ ] **Step 8: 运行智能体测试和完整测试**

---

### Task 5: 环境真实补位跟踪

**Files:**
- Modify: `environments/continuous_2d.py`
- Modify: `tests/test_continuous_2d.py`

**Interfaces:**
- Produces: `StepResult.info` 中的 `replacement_active_count`, `replacement_targets`, `replacement_coverage_restored`, `uncovered_lane_ratio`, `replacement_switches`；`replacement_snapshot()`。
- Consumes: 动作 `metadata` 的 `known_missing`, `replacement_for`, `replacement_lane_y`, `replacement_bid_score`。

- [ ] **Step 1: 写环境只信任真实动作和位置的失败测试**

构造A动作声明替代B、搜索带中心为8.0，但A真实纵坐标为1.0。执行一步后断言：

```python
self.assertEqual(result.info["replacement_targets"], ("B",))
self.assertEqual(result.info["replacement_active_count"], 1)
self.assertFalse(result.info["replacement_coverage_restored"]["B"])
```

- [ ] **Step 2: 运行并确认缺少info字段**

- [ ] **Step 3: 实现动作元数据解析和真实位置校验**

环境保存每个缺失任务当前补位者、首次响应时间、首次到达时间和切换次数。到达
条件为 `abs(agent_y - lane_y) <= max(configured_tolerance, sensor_range)`；未知
智能体、非有限搜索带、补位者与动作所有者不一致的元数据被忽略。

- [ ] **Step 4: 写未覆盖比例和切换计数测试**

三个初始节点中B、C被报告缺失，只有B有有效补位动作时，未覆盖比例严格为
`1/3`。随后B补位者从A换成C，断言累计切换次数增加1。

- [ ] **Step 5: 写 reset 清理测试**

先产生补位记录，再 `reset(seed=...)`，断言 `replacement_snapshot()` 中不存在
上一回合的响应、恢复和切换状态。

- [ ] **Step 6: 运行环境和完整测试**

---

### Task 6: 补位评价指标与统一策略对照

**Files:**
- Modify: `evaluation/metrics.py`
- Modify: `evaluation/evaluator.py`
- Modify: `evaluation/plotting.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `replacement_response_time`, `coverage_recovery_time`, `mean_uncovered_lane_ratio`, `replacement_switches`, `replacement_success_rate`；评估策略值 `"replacement"`。
- Consumes: Task 5 的环境info字段和 Task 4 的智能体。

- [ ] **Step 1: 写指标语义失败测试**

扩展 `StepTrace` 后构造：故障在1.0确认，1.4首次补位，2.0首次到达。手工断言
响应时间0.4、覆盖恢复时间1.0、一次任务切换；从未补位的轨迹两个时间均为
`None`。

- [ ] **Step 2: 实现时间序列汇总**

只使用时间戳和布尔状态计算首次事件；不把回合结束时间当作恢复时间。平均未
覆盖比例取所有时间步真实比例均值。成功率分母为实际确认缺失任务数，无任务
场景保持 `None`。

- [ ] **Step 3: 写补位策略构造失败测试**

```python
scenario = ExperimentScenario(
    name="replacement_fault",
    agent_count=3,
    episodes=1,
    max_steps=20,
    node_faults=(NodeFaultSpec("agent-2", 0.4, 1.2),),
)
episodes = evaluate_scenario(scenario, policy="replacement")
self.assertEqual(episodes[0].policy, "replacement")
```

- [ ] **Step 4: 扩展 `_build_environment`**

对 `policy="replacement"` 创建共享完整名单但各自独立协调器的
`ReplacementSearchAgent`。规则和学习策略路径不变。评估回合继续每次新建环境，
避免协调状态泄漏。

- [ ] **Step 5: 写专项图生成测试并实现绘图**

新增 `plot_replacement_summary`，四个面板展示响应时间、覆盖恢复时间、未覆盖
比例和切换次数；`None` 标记为 `N/A`。测试只断言真实生成PNG且尺寸大于0。

- [ ] **Step 6: 运行评估和完整测试**

---

### Task 7: 真实演示、评估入口和项目文档

**Files:**
- Create: `scripts/run_replacement_demo.py`
- Modify: `scripts/evaluate_policies.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`

**Interfaces:**
- Produces: 可直接运行的补位演示和 `rule/replacement/learned` 对照命令。
- Consumes: Tasks 1-6 的公开接口。

- [ ] **Step 1: 写演示脚本的集成测试**

在 `tests/test_replacement_agent.py` 中直接调用脚本导出的
`run_demo(max_steps=30, seed=7)`，断言返回字典包含实际步数、确认缺失任务数、
补位响应次数、覆盖恢复次数和消息丢包率；不对成功率写死为理想值。

- [ ] **Step 2: 运行并确认 `run_replacement_demo` 不存在**

- [ ] **Step 3: 实现最小演示**

使用4个补位智能体、固定随机种子、一个临时节点故障、`packet_loss_rate=0.1`，
每步通过智能体正常决策触发消息。返回和打印实际结果，不内置预生成数字。

- [ ] **Step 4: 扩展正式评估入口**

默认策略顺序为规则基线、补位规则基线；提供检查点时再加入学习策略。保存原有
汇总图和新增 `replacement_summary.png`。命令：

```powershell
python scripts\evaluate_policies.py --episodes 10 --agent-counts 3,4,6 --max-steps 300
```

- [ ] **Step 5: 更新README和架构文档**

文档必须说明算法状态机、竞价公式、运行命令、指标定义、通信分区局限，以及
“分布式确定性基线不是全局最优或学习式鲁棒控制”的边界。

- [ ] **Step 6: 运行快速真实演示**

Run:

```powershell
python scripts\run_replacement_demo.py --max-steps 40 --seed 7
```

记录真实输出；若故障窗口太短而未确认，调整演示配置而不是伪造响应。

- [ ] **Step 7: 运行快速规则/补位对照**

Run:

```powershell
python scripts\evaluate_policies.py --episodes 1 --agent-counts 3 --max-steps 40 --output-dir outputs\replacement_quick
```

验证JSONL、CSV、JSON和两张PNG路径存在且文件非空；明确结果只用于流程验证。

- [ ] **Step 8: 最终验证**

Run:

```powershell
python -m unittest discover -s tests -v
python -m pip check
python -c "import ast,pathlib; files=list(pathlib.Path('.').glob('**/*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(len(files))"
```

Expected: 所有测试零失败、依赖无损坏、所有Python文件可解析。

- [ ] **Step 9: 检查改动范围**

使用 `rg --files` 和文件时间核对，只包含本计划列出的源码、测试、文档和真实
输出；不得删除原始数据、旧测试、规则基线或已有实验结果。
