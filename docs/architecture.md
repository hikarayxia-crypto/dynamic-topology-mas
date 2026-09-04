# 智能体架构设计文档

> 文档状态：2026-09-01 更新。本文件同时记录目标架构与当前实现；目录树中标注
> “已实现”的文件已经存在，其余条目仍是后续规划，不能作为现成功能使用。

## 0. 当前实现状态

当前项目以 Python 为主，已完成动态拓扑、仿真基础层、协作补位规则基线和首个
Graph-MAPPO 训练链路，实际可运行结构如下：

```text
大创/
├── README.md
├── pyproject.toml
├── core/
│   ├── __init__.py
│   ├── action.py
│   ├── agent.py
│   ├── environment.py
│   ├── message.py
│   ├── observation.py
│   └── topology.py
├── interaction/
│   ├── __init__.py
│   └── communication.py
├── agents/
│   ├── __init__.py
│   ├── learning_agent.py
│   ├── replacement_agent.py
│   └── rule_based_agent.py
├── coordination/
│   ├── __init__.py
│   └── replacement.py
├── algorithms/
│   ├── __init__.py
│   ├── graph_encoder.py
│   └── shared_actor_critic.py
├── environments/
│   ├── __init__.py
│   └── continuous_2d.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── metrics.py
│   ├── plotting.py
│   └── reporting.py
├── tests/
│   ├── __init__.py
│   ├── test_communication.py
│   ├── test_continuous_2d.py
│   ├── test_core_models.py
│   ├── test_evaluation.py
│   ├── test_learning.py
│   ├── test_replacement_agent.py
│   ├── test_replacement_coordination.py
│   └── test_topology.py
├── training/
│   ├── __init__.py
│   └── mappo.py
├── scripts/
│   ├── demo_dynamic_topology.py     # 时间变化与邻接矩阵演示
│   ├── run_simulation.py            # 规则策略快速仿真
│   ├── run_replacement_demo.py       # 协作补位与丢包快速演示
│   ├── train_mappo.py               # Graph-MAPPO 训练入口
│   └── evaluate_policies.py          # 弹性与规模泛化评估
├── docs/
│   └── architecture.md              # 本架构文档
└── legacy/
    ├── README.md
    └── topology_prototype_invalid.txt
```

正式拓扑接口统一由 `core.topology.DynamicTopology` 提供，主要能力包括：

- 节点新增、永久移除、暂时失效与恢复；
- 无向图和有向图的建边、断开、重连与边权更新；
- 邻居、直连关系、多跳可达性和连接状态查询；
- 邻接表维护，以及带节点顺序映射的 NumPy 邻接矩阵生成；
- 基于时间戳和版本号的拓扑变化历史；
- 未来拓扑事件调度与可序列化快照。

基础交互层还提供统一动作、可变长度邻域观测、消息、智能体/环境抽象，以及
`interaction.communication.CommunicationBus`。通信总线直接查询当前拓扑，
能够模拟发送时不可达、传输途中断链、节点离线、时延、丢包、噪声和消息过期。

`Continuous2DSearchEnv` 已实现连续二维未知目标搜索、距离拓扑、共享奖励、碰撞
与连通性指标、链路/节点故障和通信投递。`RuleBasedSearchAgent` 使用分带往返
扫描和邻居排斥，可作为学习策略的非学习基线。

`GraphObservationEncoder` 对当前邻居消息执行共享映射和加权均值聚合，不依赖
邻居顺序和固定节点数；`SharedGraphActorCritic` 为所有智能体共享 Actor，并由
集中式 Critic 池化当时全部在线智能体表示。`MAPPOTrainer` 已实现动态规模轨迹
采样、GAE、PPO 裁剪更新和检查点保存。当前仍需执行足够长的正式训练、增加
重复实验与统计置信区间，不能把最小训练验证视为研究结果。

`evaluation/` 已提供独立评估层：同一随机种子下比较无故障与故障场景，记录任务
成功率、成功步数、累计团队奖励、速度一致性误差、平均连通率、碰撞、故障恢复
率/时间、相对无故障收益下降，以及补位响应、覆盖恢复、未覆盖搜索带比例、
补位者切换和补位成功率，并保存逐回合数据、CSV/JSON 汇总和 PNG 图。当前快速
评估只验证流程；正式研究仍需增加训练时长、回合数量和统计置信区间。

### 0.1 协作补位算法

每个智能体按照固定成员表获得稳定搜索带。协调器以仿真时间推进节点状态：心跳
超过 `failure_timeout` 后先进入怀疑态，继续超过 `failure_confirmation` 才确认
缺失；恢复心跳必须持续 `recovery_stability`，避免短暂重连造成任务来回切换。
这种二阶段状态机牺牲少量响应速度，用于抑制丢包和瞬态断链引起的误判。

确认缺失后，每个候选者只用本地可得信息计算竞价：

```text
score = 距离成本 + 当前补位负载成本 + 低连通度成本 + 低能量成本
```

各项权重由 `ReplacementConfig` 配置且总和为 1。节点通过
`replacement_gossip` 转发心跳和当前最优竞价，并按分数、候选者标识稳定破同分；
赢家接管缺失节点的固定搜索带。环境不相信策略自报的“已恢复”，而是用动作执行
后的真实纵坐标和容差判断覆盖，并记录首次响应、首次覆盖、未覆盖比例和切换次数。
恢复节点通过稳定窗口后，补位者完成当前驻留周期再交还，减少控制抖动。

该方法是去中心化、确定性的规则基线，不保证全局最优，也不是学习得到的鲁棒
控制器。网络完全分区时各分量只能维护局部视图，可能暂时产生重复或冲突补位；
重连后依靠 gossip 重新收敛。正式结论必须通过多种分区、丢包、规模和随机种子
实验获得，不能由快速演示替代。

`legacy/` 只保存早期概念原型，不属于运行时依赖。后续环境、智能体、通信和
训练模块应调用 `DynamicTopology`，不要复制或重新维护第二套拓扑状态。

## 1. 目录结构

```
dt_ms/
├── CMakeLists.txt                   # 根 CMake：负责构建 C++ 扩展 (可选)
├── pyproject.toml                   # Python 包元数据 (PEP 621)
├── README.md
├── .gitignore
├── config/                          # 配置文件 (YAML/JSON)
│   ├── default.yaml                 # 默认参数：环境尺寸、智能体数量等
│   └── experiments/                 # 不同实验场景配置
│       └── dynamic_topo_bench.yaml
├── core/                            # Python 纯抽象层与基类
│   ├── __init__.py
│   ├── agent.py                     # BaseAgent 等（已实现）
│   ├── environment.py               # BaseEnvironment（已实现）
│   ├── message.py                   # Message（已实现）
│   ├── observation.py               # Observation（已实现）
│   ├── action.py                    # Action（已实现）
│   └── topology.py                  # 动态拓扑管理接口（已实现）
├── agents/                          # 具体智能体算法实现 (纯 Python)
│   ├── __init__.py
│   ├── random_agent.py              # 随机游走测试基准
│   ├── rule_based_agent.py          # 分带扫描规则基线（已实现）
│   └── learning_agent.py            # 共享图策略推理封装（已实现）
├── algorithms/
│   ├── graph_encoder.py             # 拓扑无关图聚合（已实现）
│   └── shared_actor_critic.py       # 共享 Actor/集中 Critic（已实现）
├── training/
│   └── mappo.py                     # GAE/PPO 训练循环（已实现）
├── evaluation/
│   ├── evaluator.py                 # 独立场景评估（已实现）
│   ├── metrics.py                   # 弹性和协同指标（已实现）
│   ├── reporting.py                 # JSONL/CSV/JSON 保存（已实现）
│   └── plotting.py                  # PNG 曲线与对比图（已实现）
├── environments/                    # 仿真环境实现 (纯 Python)
│   ├── __init__.py
│   ├── simple_grid.py               # 2D 网格环境
│   └── continuous_2d.py             # 连续协同搜索环境（已实现）
├── src/                             # C++ 加速后端源码 (可选)
│   ├── CMakeLists.txt               # 子目录 CMake：组织 C++ 子库
│   ├── core/                        # 底层核心算法 C++ 实现
│   │   ├── graph_engine.h
│   │   ├── graph_engine.cpp         # 动态图邻居搜索、最短路径
│   │   ├── spatial_hash.h
│   │   └── spatial_hash.cpp         # 空间哈希加速感知查询
│   ├── bindings/                    # pybind11 绑定代码
│   │   ├── bindings.cpp
│   │   └── ...
│   └── utils/
├── interaction/                     # 交互协议的具体实现 (Python)
│   ├── __init__.py
│   ├── communication.py             # 动态拓扑消息总线（已实现）
│   ├── perception.py                # 感知剔除 (可调用 C++ 空间查询)
│   └── action_executor.py           # 动作解析与冲突处理
├── utils/                           # 通用工具 (Python)
│   ├── __init__.py
│   ├── logging_config.py
│   ├── metrics.py
│   ├── visualization.py
│   └── profiling.py                 # 计时对比 Python vs C++ 加速
├── tests/                           # 单元测试
│   ├── __init__.py
│   ├── test_communication.py        # 通信扰动测试（已实现）
│   ├── test_continuous_2d.py        # 连续环境与规则基线测试（已实现）
│   ├── test_core_models.py          # 基础数据与生命周期测试（已实现）
│   ├── test_learning.py             # 图编码与最小训练测试（已实现）
│   ├── test_evaluation.py           # 指标与图表测试（已实现）
│   ├── test_agent.py
│   ├── test_topology.py             # 动态拓扑测试（已实现）
│   ├── test_cpp_extension.py        # 测试 C++ 模块是否正常加载
│   └── conftest.py
├── scripts/                         # 运行脚本与实验入口
│   ├── demo_dynamic_topology.py     # 动态拓扑演示（已实现）
│   ├── run_simulation.py            # 连续搜索快速验证（已实现）
│   ├── train_mappo.py               # Graph-MAPPO 训练入口（已实现）
│   ├── evaluate_policies.py         # 弹性/规模评估入口（已实现）
│   └── benchmark_topo_change.py
└── docs/                            # 文档
```

## 2. 智能体的统一表示结构

本节保留最初的接口设计思路。当前可执行接口以 `core/` 源码和项目 README 为准；
实现已增加通信范围、输入校验、独立 NumPy 状态数组、可变邻域观测以及同步动作
批次检查，以支持后续动态规模实验。

### 2.1 静态属性 (AgentAttributes)

智能体的静态属性是指在智能体生命周期内通常保持不变的特性，定义在 `core/agent.py` 文件中：

```python
@dataclass
class AgentAttributes:
    agent_type: str = "base"
    max_speed: float = 1.0
    sensor_range: float = 10.0
```

- `agent_type`: 智能体类型，默认为 "base"
- `max_speed`: 智能体最大速度，默认为 1.0
- `sensor_range`: 传感器感知范围，默认为 10.0

### 2.2 动态状态 (AgentState)

智能体的动态状态是指在智能体生命周期内会不断变化的状态，定义在 `core/agent.py` 文件中：

```python
@dataclass
class AgentState:
    pos: tuple = (0.0, 0.0)
    vel: tuple = (0.0, 0.0)
    neighbors: List[str] = field(default_factory=list)   # 动态拓扑邻居列表
    inbox: List[Any] = field(default_factory=list)       # 消息队列
```

- `pos`: 智能体位置，默认为 (0.0, 0.0)
- `vel`: 智能体速度，默认为 (0.0, 0.0)
- `neighbors`: 动态拓扑邻居列表，存储邻居智能体的 ID
- `inbox`: 消息队列，存储接收到的消息

### 2.3 智能体基类 (BaseAgent)

智能体基类定义了所有智能体共有的接口和行为，是一个抽象基类，定义在 `core/agent.py` 文件中：

```python
class BaseAgent(ABC):
    def __init__(self, agent_id: Optional[str] = None, attributes: AgentAttributes = None):
        self.id = agent_id or str(uuid.uuid4())
        self.attr = attributes or AgentAttributes()
        self.state = AgentState()
        self._env = None                        # 环境引用，由环境注入

    def bind_env(self, env):
        """由环境调用，建立双向关联"""
        self._env = env

    # ---------- 感知接口（子类必须实现）----------
    @abstractmethod
    def perceive(self, global_obs: Any) -> Any:
        """处理环境原始观测，返回智能体内部可用的感知数据"""
        pass

    # ---------- 通信接口 ----------
    def send_msg(self, target: str, content: Any):
        """向目标智能体发送消息，经由环境路由"""
        if self._env:
            self._env.route_msg(sender=self.id, receiver=target, content=content)

    def receive_msg(self, msg: Any):
        """环境将消息投递至此处"""
        self.state.inbox.append(msg)

    # ---------- 决策接口（子类必须实现）----------
    @abstractmethod
    def decide(self, dt: float) -> Any:
        """基于当前状态生成动作指令"""
        pass

    # ---------- 执行接口 ----------
    def act(self, action: Any, dt: float):
        """将动作提交给环境执行"""
        if self._env:
            self._env.apply_action(self.id, action)

    # ---------- 模板方法 ----------
    def step(self, dt: float):
        """单步迭代的骨架，通常由环境循环调用"""
        # 注意：实际使用中，感知可能由环境在调用 step 前统一注入
        # 此处可根据需要调用 self.perceive()
        action = self.decide(dt)
        self.act(action, dt)
```

## 3. 智能体间交互的统一接口

### 3.1 感知接口 (Perception)

- **接口方法**: `perceive(global_obs: Any) -> Any`
- **作用**: 处理环境提供的原始观测数据，转换为智能体内部可用的感知信息
- **实现要求**: 子类必须实现此方法
- **数据流**: 环境 → 智能体感知方法 → 智能体内部状态

### 3.2 通信接口 (Communication)

- **发送方法**: `send_msg(target: str, content: Any)`
  - **作用**: 向目标智能体发送消息
  - **实现**: 经由环境路由消息

- **接收方法**: `receive_msg(msg: Any)`
  - **作用**: 接收来自其他智能体的消息
  - **实现**: 将消息添加到智能体的消息队列中

### 3.3 动作接口 (Action)

- **决策方法**: `decide(dt: float) -> Any`
  - **作用**: 基于当前状态生成动作指令
  - **实现要求**: 子类必须实现此方法

- **执行方法**: `act(action: Any, dt: float)`
  - **作用**: 将动作提交给环境执行
  - **实现**: 调用环境的 `apply_action` 方法

## 4. 代码框架与命名规范

### 4.1 代码框架

1. **核心抽象层** (`core/`):
   - 定义智能体、环境、消息、观测、动作等基本数据结构和抽象接口
   - 提供统一的接口规范，确保不同实现的兼容性

2. **具体实现层**:
   - **智能体实现** (`agents/`): 包含各种具体的智能体算法实现
   - **环境实现** (`environments/`): 包含各种具体的仿真环境实现
   - **交互实现** (`interaction/`): 包含通信、感知、动作执行的具体实现

3. **加速层** (`src/`):
   - 提供 C++ 实现的核心算法，如动态图邻居搜索、空间哈希加速等
   - 通过 pybind11 绑定到 Python

4. **工具层** (`utils/`):
   - 提供通用工具，如日志配置、 metrics 计算、可视化等

5. **测试层** (`tests/`):
   - 包含单元测试，确保代码的正确性

6. **脚本层** (`scripts/`):
   - 提供运行脚本和实验入口

### 4.2 命名规范

1. **文件命名**:
   - 使用小写字母和下划线组合，如 `agent.py`、`rule_based_agent.py`
   - 模块文件使用单数形式，如 `agent.py` 而非 `agents.py`

2. **类命名**:
   - 使用驼峰命名法，如 `BaseAgent`、`AgentAttributes`
   - 抽象基类以 `Base` 开头，如 `BaseAgent`

3. **函数命名**:
   - 使用小写字母和下划线组合，如 `send_msg`、`receive_msg`
   - 方法名应清晰表达其功能

4. **变量命名**:
   - 使用小写字母和下划线组合，如 `agent_id`、`sensor_range`
   - 私有变量以 `_` 开头，如 `_env`

5. **常量命名**:
   - 使用全大写字母和下划线组合，如 `MAX_SPEED`

## 5. 扩展与定制

### 5.1 自定义智能体

要实现自定义智能体，需要继承 `BaseAgent` 类并实现以下抽象方法：

1. `perceive(global_obs: Any) -> Any`:
   - 处理环境提供的原始观测数据
   - 返回智能体内部可用的感知信息

2. `decide(dt: float) -> Any`:
   - 基于当前状态和感知信息生成动作指令
   - 返回动作对象

### 5.2 自定义环境

要实现自定义环境，需要定义环境类并实现以下功能：

1. 管理智能体的生命周期
2. 提供观测数据给智能体
3. 路由智能体间的消息
4. 执行智能体的动作
5. 更新环境状态

### 5.3 性能优化

对于大规模仿真场景，可以：

1. 使用 C++ 实现的核心算法，如动态图邻居搜索和空间哈希
2. 利用 `utils/profiling.py` 工具对比 Python 和 C++ 实现的性能
3. 根据需要调整数据结构和算法，优化计算效率

## 6. 示例代码

### 6.1 实现一个简单的智能体

```python
from core.agent import BaseAgent, AgentAttributes

class SimpleAgent(BaseAgent):
    def __init__(self, agent_id=None, attributes=None):
        super().__init__(agent_id, attributes)
        
    def perceive(self, global_obs):
        # 处理观测数据
        return global_obs
    
    def decide(self, dt):
        # 简单的随机决策
        import random
        action = {
            "type": "move",
            "direction": (random.uniform(-1, 1), random.uniform(-1, 1))
        }
        return action
```

### 6.2 运行仿真

```python
# scripts/run_simulation.py
from environments.simple_grid import SimpleGridEnv
from agents.random_agent import RandomAgent

# 创建环境
env = SimpleGridEnv(grid_size=(10, 10))

# 添加智能体
for i in range(5):
    agent = RandomAgent()
    env.add_agent(agent)

# 运行仿真
for step in range(100):
    env.step(0.1)
    # 可选：可视化或记录数据
```
