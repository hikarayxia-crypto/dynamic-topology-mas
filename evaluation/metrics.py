"""协同搜索、动态拓扑和故障恢复评价指标。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StepTrace:
    """一个评估时间步的原始指标，不对结果进行补造或平滑。"""

    timestamp: float
    team_reward: float
    detection_ratio: float
    connectivity_ratio: float
    consistency_error: float
    collision_count: int
    active_agents: int
    fault_healthy: bool
    known_missing: tuple[str, ...] = ()
    replacement_targets: tuple[str, ...] = ()
    replacement_coverage_restored: Mapping[str, bool] = field(default_factory=dict)
    uncovered_lane_ratio: float = 0.0
    replacement_switches: int = 0


@dataclass(frozen=True)
class EpisodeMetrics:
    """一个完整回合的汇总结果。

    ``convergence_steps`` 采用任务成功所需步数；失败回合保持 ``None``，避免把
    时间上限错误当成收敛。故障恢复时间也只在真实恢复时给出数值。
    """

    scenario: str
    policy: str
    episode: int
    seed: int
    agent_count: int
    success: bool
    steps: int
    convergence_steps: int | None
    cumulative_team_reward: float
    mean_connectivity: float
    mean_consistency_error: float
    collisions: int
    final_detection_ratio: float
    fault_recovery_time: float | None
    topology_performance_drop: float | None
    replacement_response_time: float | None
    coverage_recovery_time: float | None
    mean_uncovered_lane_ratio: float
    replacement_switches: int
    replacement_success_rate: float | None

    def as_dict(self) -> dict[str, object]:
        """返回可直接写入 JSON/CSV 的字典。"""

        return asdict(self)


def velocity_consistency_error(velocities: Sequence[np.ndarray]) -> float:
    """计算速度相对群体均值的均方根偏差。

    单个或没有在线智能体时不存在群体分歧，返回 0；该定义适用于不同规模，且
    不会因为智能体数量增加而机械增大。
    """

    if len(velocities) <= 1:
        return 0.0
    matrix = np.stack(velocities).astype(np.float64, copy=False)
    center = matrix.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((matrix - center) ** 2, axis=1))))


def summarize_episode(
    traces: Sequence[StepTrace],
    *,
    scenario: str,
    policy: str,
    episode: int,
    seed: int,
    agent_count: int,
    success: bool,
    fault_start: float | None,
    fault_end: float | None,
) -> EpisodeMetrics:
    """从原始时间序列计算回合指标。"""

    if not traces:
        raise ValueError("traces 不能为空")
    rewards = np.asarray([trace.team_reward for trace in traces], dtype=np.float64)
    recovery_time: float | None = None
    if fault_end is not None:
        recovered = [
            trace.timestamp
            for trace in traces
            if trace.timestamp >= fault_end and trace.fault_healthy
        ]
        if recovered:
            recovery_time = max(0.0, recovered[0] - fault_end)

    performance_drop: float | None = None
    if fault_start is not None:
        before = [
            trace.team_reward for trace in traces if trace.timestamp < fault_start
        ]
        after = [
            trace.team_reward for trace in traces if trace.timestamp >= fault_start
        ]
        if before and after:
            before_mean = float(np.mean(before))
            after_mean = float(np.mean(after))
            # 以故障前每步收益为参照，只记录性能下降；性能提升时记为 0。
            performance_drop = max(
                0.0, (before_mean - after_mean) / max(abs(before_mean), 1e-8)
            )

    # 以每个任务首次进入 known_missing 的时刻作为分布式确认点；只有真实发生的
    # 首次响应/到达才产生延迟，失败任务绝不能用回合结束时刻补造结果。
    confirmed_at: dict[str, float] = {}
    responded_at: dict[str, float] = {}
    restored_at: dict[str, float] = {}
    for trace in traces:
        for missing_id in trace.known_missing:
            confirmed_at.setdefault(str(missing_id), trace.timestamp)
        for missing_id in trace.replacement_targets:
            missing_key = str(missing_id)
            if missing_key in confirmed_at:
                responded_at.setdefault(missing_key, trace.timestamp)
        for missing_id, restored in trace.replacement_coverage_restored.items():
            missing_key = str(missing_id)
            if restored and missing_key in confirmed_at:
                restored_at.setdefault(missing_key, trace.timestamp)

    response_delays = [
        responded_at[missing_id] - confirmed_at[missing_id]
        for missing_id in confirmed_at
        if missing_id in responded_at
    ]
    coverage_delays = [
        restored_at[missing_id] - confirmed_at[missing_id]
        for missing_id in confirmed_at
        if missing_id in restored_at
    ]
    replacement_success_rate = (
        len(restored_at) / len(confirmed_at) if confirmed_at else None
    )

    return EpisodeMetrics(
        scenario=scenario,
        policy=policy,
        episode=episode,
        seed=seed,
        agent_count=agent_count,
        success=success,
        steps=len(traces),
        convergence_steps=len(traces) if success else None,
        cumulative_team_reward=float(rewards.sum()),
        mean_connectivity=float(
            np.mean([trace.connectivity_ratio for trace in traces])
        ),
        mean_consistency_error=float(
            np.mean([trace.consistency_error for trace in traces])
        ),
        collisions=sum(trace.collision_count for trace in traces),
        final_detection_ratio=traces[-1].detection_ratio,
        fault_recovery_time=recovery_time,
        topology_performance_drop=performance_drop,
        replacement_response_time=(
            float(np.mean(response_delays)) if response_delays else None
        ),
        coverage_recovery_time=(
            float(np.mean(coverage_delays)) if coverage_delays else None
        ),
        mean_uncovered_lane_ratio=float(
            np.mean([trace.uncovered_lane_ratio for trace in traces])
        ),
        replacement_switches=max(trace.replacement_switches for trace in traces),
        replacement_success_rate=replacement_success_rate,
    )
