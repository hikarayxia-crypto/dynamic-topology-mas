"""参数共享 Graph-MAPPO 的最小训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import SharedPolicyAgent
from algorithms import SharedGraphActorCritic
from core import AgentAttributes
from environments import Continuous2DConfig, Continuous2DSearchEnv
from evaluation.plotting import plot_training_metrics
from training import MAPPOConfig, MAPPOTrainer


def build_environment(
    model: SharedGraphActorCritic, agent_count: int, max_steps: int
) -> Continuous2DSearchEnv:
    """构造训练环境；所有智能体显式持有同一个策略对象。"""

    environment = Continuous2DSearchEnv(
        Continuous2DConfig(max_steps=max_steps, n_targets=6)
    )
    attributes = AgentAttributes(
        agent_type="uav",
        max_speed=3.0,
        sensor_range=4.0,
        communication_range=20.0,
    )
    for index in range(agent_count):
        environment.add_agent(
            SharedPolicyAgent(f"agent-{index}", model, attributes)
        )
    # 在训练中注入可恢复断链，使策略采样真实经历邻域缺失和拓扑切换。
    if agent_count >= 2:
        environment.schedule_link_fault(
            "agent-0", "agent-1", start_time=4.0, duration=2.0
        )
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练共享 Graph-MAPPO 策略")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.agents <= 0:
        raise ValueError("updates 和 agents 必须为正整数")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = SharedGraphActorCritic(6, 5, 2, hidden_dim=args.hidden_dim)
    environment = build_environment(model, args.agents, args.max_steps)
    trainer = MAPPOTrainer(
        model,
        MAPPOConfig(
            rollout_steps=args.rollout_steps,
            update_epochs=args.update_epochs,
        ),
    )
    history = trainer.train(environment, args.updates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as file:
        for metrics in history:
            file.write(json.dumps(metrics.as_dict(), ensure_ascii=False) + "\n")
    checkpoint_path = trainer.save_checkpoint(args.output_dir / "mappo_checkpoint.pt")
    curve_path = plot_training_metrics(
        metrics_path, args.output_dir / "training_curves.png"
    )
    final = history[-1]
    print("训练完成（以下为本次实际运行结果）")
    print(f"环境步数: {final.environment_steps}")
    print(f"平均团队奖励: {final.mean_team_reward:.6f}")
    print(f"Actor 损失: {final.actor_loss:.6f}")
    print(f"Value 损失: {final.value_loss:.6f}")
    print(f"指标文件: {metrics_path}")
    print(f"模型文件: {checkpoint_path}")
    print(f"训练曲线: {curve_path}")


if __name__ == "__main__":
    main()
