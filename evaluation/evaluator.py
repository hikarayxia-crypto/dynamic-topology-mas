"""规则策略与学习策略共用的独立场景评估器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np

from agents import ReplacementSearchAgent, RuleBasedSearchAgent, SharedPolicyAgent
from algorithms import SharedGraphActorCritic
from core import AgentAttributes
from environments import Continuous2DConfig, Continuous2DSearchEnv

from .metrics import (
    EpisodeMetrics,
    StepTrace,
    summarize_episode,
    velocity_consistency_error,
)


@dataclass(frozen=True)
class LinkFaultSpec:
    """评估场景中的可恢复链路故障。"""

    source: str
    target: str
    start_time: float
    duration: float


@dataclass(frozen=True)
class NodeFaultSpec:
    """评估场景中的可恢复节点故障。"""

    agent_id: str
    start_time: float
    duration: float


@dataclass(frozen=True)
class ExperimentScenario:
    """可复现的规模与拓扑扰动实验定义。"""

    name: str
    agent_count: int
    episodes: int = 10
    base_seed: int = 1000
    max_steps: int = 300
    n_targets: int = 6
    width: float = 50.0
    height: float = 50.0
    link_faults: tuple[LinkFaultSpec, ...] = ()
    node_faults: tuple[NodeFaultSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("场景名称不能为空")
        if self.agent_count <= 0 or self.episodes <= 0 or self.max_steps <= 0:
            raise ValueError("智能体数、回合数和最大步数必须为正整数")
        valid_ids = {f"agent-{index}" for index in range(self.agent_count)}
        if any(
            fault.source not in valid_ids or fault.target not in valid_ids
            for fault in self.link_faults
        ):
            raise ValueError("链路故障引用了场景中不存在的智能体")
        if any(fault.agent_id not in valid_ids for fault in self.node_faults):
            raise ValueError("节点故障引用了场景中不存在的智能体")
        for fault in (*self.link_faults, *self.node_faults):
            if fault.start_time < 0 or fault.duration <= 0:
                raise ValueError("故障开始时间不能为负，持续时间必须为正")
        if any(fault.source == fault.target for fault in self.link_faults):
            raise ValueError("链路故障的两个端点不能相同")

    @property
    def fault_start(self) -> float | None:
        starts = [fault.start_time for fault in (*self.link_faults, *self.node_faults)]
        return min(starts) if starts else None

    @property
    def fault_end(self) -> float | None:
        ends = [
            fault.start_time + fault.duration
            for fault in (*self.link_faults, *self.node_faults)
        ]
        return max(ends) if ends else None


@dataclass(frozen=True)
class ScenarioSummary:
    """同一策略和场景多回合统计。"""

    scenario: str
    policy: str
    agent_count: int
    episodes: int
    success_rate: float
    mean_convergence_steps: float | None
    mean_cumulative_reward: float
    mean_connectivity: float
    mean_consistency_error: float
    mean_collisions: float
    mean_final_detection_ratio: float
    recovery_rate: float | None
    mean_recovery_time: float | None
    mean_topology_performance_drop: float | None
    mean_replacement_response_time: float | None
    mean_coverage_recovery_time: float | None
    mean_uncovered_lane_ratio: float
    mean_replacement_switches: float
    mean_replacement_success_rate: float | None
    relative_reward_drop_vs_nominal: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _build_environment(
    scenario: ExperimentScenario,
    policy: Literal["rule", "replacement", "learned"],
    model: SharedGraphActorCritic | None,
) -> Continuous2DSearchEnv:
    environment = Continuous2DSearchEnv(
        Continuous2DConfig(
            width=scenario.width,
            height=scenario.height,
            max_steps=scenario.max_steps,
            n_targets=scenario.n_targets,
        )
    )
    attributes = AgentAttributes(
        agent_type="uav",
        max_speed=3.0,
        sensor_range=4.0,
        communication_range=20.0,
    )
    roster = tuple(f"agent-{index}" for index in range(scenario.agent_count))
    for index in range(scenario.agent_count):
        agent_id = f"agent-{index}"
        if policy == "rule":
            agent = RuleBasedSearchAgent(agent_id, attributes)
        elif policy == "replacement":
            # 每个节点使用相同固定名单但持有独立协调器，避免跨节点共享可变共识。
            agent = ReplacementSearchAgent(
                agent_id,
                roster,
                scenario.height,
                attributes,
            )
        else:
            if model is None:
                raise ValueError("评估学习策略时必须提供 model")
            agent = SharedPolicyAgent(agent_id, model, attributes, deterministic=True)
        environment.add_agent(agent)
    for fault in scenario.link_faults:
        environment.schedule_link_fault(
            fault.source,
            fault.target,
            start_time=fault.start_time,
            duration=fault.duration,
        )
    for fault in scenario.node_faults:
        environment.schedule_node_fault(
            fault.agent_id,
            start_time=fault.start_time,
            duration=fault.duration,
        )
    return environment


def _fault_healthy(
    environment: Continuous2DSearchEnv, scenario: ExperimentScenario
) -> bool:
    """检查受扰节点在线且受扰边已经按物理通信条件恢复。"""

    nodes_healthy = all(
        environment.get_agent(fault.agent_id).state.active
        for fault in scenario.node_faults
    )
    links_healthy = all(
        environment.topology.are_directly_connected(fault.source, fault.target)
        for fault in scenario.link_faults
    )
    return nodes_healthy and links_healthy


def run_episode(
    scenario: ExperimentScenario,
    *,
    policy: Literal["rule", "replacement", "learned"],
    episode: int,
    model: SharedGraphActorCritic | None = None,
) -> EpisodeMetrics:
    """执行一个不参与训练的独立评估回合。"""

    seed = scenario.base_seed + episode
    environment = _build_environment(scenario, policy, model)
    observations = environment.reset(seed=seed)
    traces: list[StepTrace] = []
    success = False
    while observations:
        actions = {
            agent_id: environment.get_agent(agent_id).step(
                observation, environment.config.dt
            )
            for agent_id, observation in observations.items()
        }
        result = environment.step(actions)
        acted_ids = tuple(actions)
        velocities = [
            agent.state.velocity
            for agent in environment.agents.values()
            if agent.state.active
        ]
        traces.append(
            StepTrace(
                timestamp=float(result.info["time"]),
                team_reward=float(np.mean([result.rewards[key] for key in acted_ids])),
                detection_ratio=float(result.info["detection_ratio"]),
                connectivity_ratio=float(result.info["connectivity_ratio"]),
                consistency_error=velocity_consistency_error(velocities),
                collision_count=int(result.info["collision_count"]),
                active_agents=len(velocities),
                fault_healthy=_fault_healthy(environment, scenario),
                known_missing=tuple(result.info["known_missing"]),
                replacement_targets=tuple(result.info["replacement_targets"]),
                replacement_coverage_restored=dict(
                    result.info["replacement_coverage_restored"]
                ),
                uncovered_lane_ratio=float(result.info["uncovered_lane_ratio"]),
                replacement_switches=int(result.info["replacement_switches"]),
            )
        )
        success = bool(result.info["success"])
        if success or result.info["truncated"]:
            break
        observations = result.observations

    return summarize_episode(
        traces,
        scenario=scenario.name,
        policy=policy,
        episode=episode,
        seed=seed,
        agent_count=scenario.agent_count,
        success=success,
        fault_start=scenario.fault_start,
        fault_end=scenario.fault_end,
    )


def evaluate_scenario(
    scenario: ExperimentScenario,
    *,
    policy: Literal["rule", "replacement", "learned"],
    model: SharedGraphActorCritic | None = None,
) -> list[EpisodeMetrics]:
    """以互不重复的固定种子执行场景中的全部回合。"""

    return [
        run_episode(scenario, policy=policy, episode=index, model=model)
        for index in range(scenario.episodes)
    ]


def summarize_scenario(episodes: list[EpisodeMetrics]) -> ScenarioSummary:
    """聚合同场景、同策略回合，缺失恢复值不伪造为零。"""

    if not episodes:
        raise ValueError("episodes 不能为空")
    identity = {(item.scenario, item.policy, item.agent_count) for item in episodes}
    if len(identity) != 1:
        raise ValueError("只能聚合同一场景、策略和规模的回合")
    scenario, policy, agent_count = next(iter(identity))
    converged = [item.convergence_steps for item in episodes if item.convergence_steps]
    recovery = [
        item.fault_recovery_time
        for item in episodes
        if item.fault_recovery_time is not None
    ]
    drops = [
        item.topology_performance_drop
        for item in episodes
        if item.topology_performance_drop is not None
    ]
    response_times = [
        item.replacement_response_time
        for item in episodes
        if item.replacement_response_time is not None
    ]
    coverage_times = [
        item.coverage_recovery_time
        for item in episodes
        if item.coverage_recovery_time is not None
    ]
    replacement_success = [
        item.replacement_success_rate
        for item in episodes
        if item.replacement_success_rate is not None
    ]
    has_faults = any(item.topology_performance_drop is not None for item in episodes)
    return ScenarioSummary(
        scenario=scenario,
        policy=policy,
        agent_count=agent_count,
        episodes=len(episodes),
        success_rate=float(np.mean([item.success for item in episodes])),
        mean_convergence_steps=float(np.mean(converged)) if converged else None,
        mean_cumulative_reward=float(
            np.mean([item.cumulative_team_reward for item in episodes])
        ),
        mean_connectivity=float(np.mean([item.mean_connectivity for item in episodes])),
        mean_consistency_error=float(
            np.mean([item.mean_consistency_error for item in episodes])
        ),
        mean_collisions=float(np.mean([item.collisions for item in episodes])),
        mean_final_detection_ratio=float(
            np.mean([item.final_detection_ratio for item in episodes])
        ),
        recovery_rate=(len(recovery) / len(episodes)) if has_faults else None,
        mean_recovery_time=float(np.mean(recovery)) if recovery else None,
        mean_topology_performance_drop=float(np.mean(drops)) if drops else None,
        mean_replacement_response_time=(
            float(np.mean(response_times)) if response_times else None
        ),
        mean_coverage_recovery_time=(
            float(np.mean(coverage_times)) if coverage_times else None
        ),
        mean_uncovered_lane_ratio=float(
            np.mean([item.mean_uncovered_lane_ratio for item in episodes])
        ),
        mean_replacement_switches=float(
            np.mean([item.replacement_switches for item in episodes])
        ),
        mean_replacement_success_rate=(
            float(np.mean(replacement_success)) if replacement_success else None
        ),
    )


def compare_with_nominal(
    summaries: list[ScenarioSummary], reference_scenario: str = "nominal"
) -> list[ScenarioSummary]:
    """计算同策略、同规模下相对无故障场景的累计收益下降。

    场景内指标反映故障前后变化，本指标则使用相同种子设置的独立无故障场景作
    对照，两者同时保存可避免单一口径掩盖差异。
    """

    references = {
        (item.policy, item.agent_count): item.mean_cumulative_reward
        for item in summaries
        if item.scenario == reference_scenario
    }
    compared: list[ScenarioSummary] = []
    for item in summaries:
        reference = references.get((item.policy, item.agent_count))
        if reference is None:
            drop = None
        elif item.scenario == reference_scenario:
            drop = 0.0
        else:
            drop = max(
                0.0,
                (reference - item.mean_cumulative_reward)
                / max(abs(reference), 1e-8),
            )
        compared.append(replace(item, relative_reward_drop_vs_nominal=drop))
    return compared
