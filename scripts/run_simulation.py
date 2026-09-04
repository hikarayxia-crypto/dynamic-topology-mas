"""连续二维协同搜索的快速运行入口。"""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import RuleBasedSearchAgent
from core.agent import AgentAttributes
from environments import Continuous2DConfig, Continuous2DSearchEnv
from interaction import CommunicationConfig


def main() -> None:
    """运行小规模规则基线，打印由真实仿真产生的结果。"""

    config = Continuous2DConfig(
        width=30.0,
        height=20.0,
        dt=0.25,
        max_steps=240,
        n_targets=6,
    )
    environment = Continuous2DSearchEnv(
        config,
        CommunicationConfig(base_delay=0.1, packet_loss_rate=0.05),
    )
    attributes = AgentAttributes(
        agent_type="uav",
        max_speed=2.0,
        sensor_range=3.0,
        communication_range=18.0,
    )
    for index in range(4):
        environment.add_agent(
            RuleBasedSearchAgent(f"UAV-{index + 1}", attributes=attributes)
        )

    # 在中段制造短时链路和节点故障，验证环境能够自动断开与恢复。
    environment.schedule_link_fault(
        "UAV-1", "UAV-2", start_time=10.0, duration=4.0
    )
    environment.schedule_node_fault("UAV-4", start_time=20.0, duration=3.0)

    observations = environment.reset(seed=7)
    final_info = {
        "success": False,
        "targets_discovered": 0,
        "targets_total": config.n_targets,
    }
    while observations:
        if environment.step_count % 10 == 0:
            for agent_id in observations:
                environment.get_agent(agent_id).send_message(
                    None,
                    {"step": environment.step_count, "type": "status"},
                    ttl=1.0,
                )
        actions = {
            agent_id: environment.get_agent(agent_id).step(
                observation, config.dt
            )
            for agent_id, observation in observations.items()
        }
        result = environment.step(actions)
        final_info = dict(result.info)
        observations = dict(result.observations)
        if final_info["success"] or final_info["truncated"]:
            break

    print("仿真结束")
    print(f"执行步数: {final_info['step']}")
    print(
        "发现目标: "
        f"{final_info['targets_discovered']}/{final_info['targets_total']}"
    )
    print(f"任务成功: {final_info['success']}")
    print(f"最终连通率: {final_info['connectivity_ratio']:.3f}")
    print(f"累计成功投递消息: {environment.communication.stats.delivered}")


if __name__ == "__main__":
    main()
