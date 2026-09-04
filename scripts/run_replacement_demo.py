"""协作补位算法的可复现快速演示。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import ReplacementSearchAgent
from coordination.replacement import ReplacementConfig
from core import AgentAttributes
from environments import Continuous2DConfig, Continuous2DSearchEnv
from interaction import CommunicationConfig


class DemoResult(TypedDict):
    """快速演示返回的真实观测结果。"""

    steps: int
    confirmed_missing_tasks: int
    replacement_responses: int
    coverage_recoveries: int
    packet_loss_rate: float


def run_demo(*, max_steps: int = 40, seed: int = 7) -> DemoResult:
    """运行四智能体节点故障与补位演示。

    参数:
        max_steps: 最大仿真步数，用于快速验证时控制运行时长。
        seed: 环境位置、目标位置和通信丢包共用的可复现随机种子。

    返回:
        包含实际执行步数、已确认缺失任务数、已响应任务数、恢复覆盖任务数和
        通信总线实测丢包率的字典。事件计数来自逐步环境信息，不预设补位成功。
    """

    if max_steps <= 0:
        raise ValueError("max_steps 必须为正整数")

    environment = Continuous2DSearchEnv(
        Continuous2DConfig(
            width=20.0,
            height=20.0,
            dt=0.2,
            max_steps=max_steps,
            # 固定远端目标可避免快速演示因随机提前完成而跳过故障窗口。
            target_positions=((19.9, 19.9),),
            collision_distance=0.0,
            energy_cost=0.0,
            replacement_lane_tolerance=1.0,
        ),
        CommunicationConfig(packet_loss_rate=0.1),
    )
    roster = tuple(f"agent-{index}" for index in range(4))
    attributes = AgentAttributes(
        agent_type="uav",
        max_speed=2.0,
        sensor_range=0.05,
        # 演示关注丢包与节点故障，使用全场通信范围避免距离断链混淆结果。
        communication_range=30.0,
    )
    replacement_config = ReplacementConfig(
        failure_timeout=0.4,
        failure_confirmation=0.2,
        bid_window=0.2,
        recovery_stability=0.4,
        broadcast_interval=0.2,
        dwell_steps=3,
    )
    for agent_id in roster:
        environment.add_agent(
            ReplacementSearchAgent(
                agent_id,
                roster,
                environment.config.height,
                attributes,
                replacement_config=replacement_config,
            )
        )

    # 故障持续时间覆盖“超时—确认—竞价”窗口，恢复后仍留出稳定交还时间。
    environment.schedule_node_fault("agent-3", start_time=0.6, duration=1.8)
    observations = environment.reset(seed=seed)
    confirmed: set[str] = set()
    responded: set[str] = set()
    recovered: set[str] = set()

    while observations:
        actions = {
            agent_id: environment.get_agent(agent_id).step(
                observation, environment.config.dt
            )
            for agent_id, observation in observations.items()
        }
        step_result = environment.step(actions)
        confirmed.update(map(str, step_result.info["known_missing"]))
        responded.update(map(str, step_result.info["replacement_targets"]))
        recovered.update(
            str(task_id)
            for task_id, restored in step_result.info[
                "replacement_coverage_restored"
            ].items()
            if restored
        )
        if step_result.info["success"] or step_result.info["truncated"]:
            break
        observations = step_result.observations

    return {
        "steps": environment.step_count,
        "confirmed_missing_tasks": len(confirmed),
        "replacement_responses": len(responded),
        "coverage_recoveries": len(recovered),
        "packet_loss_rate": float(environment.communication.stats.packet_loss_ratio),
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行协作补位快速演示")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    """执行演示并打印未经加工的实际指标。"""

    result = run_demo(**vars(parse_args()))
    print(f"实际步数: {result['steps']}")
    print(f"确认缺失任务数: {result['confirmed_missing_tasks']}")
    print(f"产生补位响应的任务数: {result['replacement_responses']}")
    print(f"恢复搜索带覆盖的任务数: {result['coverage_recoveries']}")
    print(f"实测消息丢包率: {result['packet_loss_rate']:.4f}")


if __name__ == "__main__":
    main()
