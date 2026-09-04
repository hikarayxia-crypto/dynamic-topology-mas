"""适用于共享团队奖励的轻量 MAPPO 训练实现。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Hashable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from algorithms.shared_actor_critic import SharedGraphActorCritic
from core.action import Action
from core.observation import Observation
from environments.continuous_2d import Continuous2DSearchEnv


@dataclass(frozen=True)
class MAPPOConfig:
    """MAPPO 优化参数；默认值偏向稳定训练而非快速演示。"""

    rollout_steps: int = 256
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.rollout_steps <= 0 or self.update_epochs <= 0:
            raise ValueError("rollout_steps 和 update_epochs 必须为正整数")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("学习率和梯度裁剪阈值必须为正数")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma 和 gae_lambda 必须位于 [0, 1]")
        if self.clip_ratio <= 0:
            raise ValueError("clip_ratio 必须为正数")
        if self.value_coef < 0 or self.entropy_coef < 0:
            raise ValueError("value_coef 和 entropy_coef 不能为负数")


@dataclass
class RolloutStep:
    """一个环境时间步的联合轨迹数据。"""

    agent_ids: tuple[Hashable, ...]
    observations: tuple[Observation, ...]
    actions: Tensor
    log_probs: Tensor
    reward: float
    done: bool
    value: float


@dataclass(frozen=True)
class TrainingMetrics:
    """一次 MAPPO 更新产生的真实优化指标。"""

    update: int
    environment_steps: int
    mean_team_reward: float
    actor_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    episodes_finished: int
    successful_episodes: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class MAPPOTrainer:
    """单环境采样、共享 Actor 更新和集中式 Critic 更新器。

    当前环境使用全队共享奖励，因此每个时间步只计算一个团队优势；该优势应用
    到当时所有在线智能体的共享 Actor。节点故障导致在线数量变化时无需填充到
    固定规模，轨迹保留各时间步真实的智能体集合。
    """

    def __init__(
        self,
        model: SharedGraphActorCritic,
        config: MAPPOConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or MAPPOConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.environment_steps = 0
        self._observations: Mapping[Hashable, Observation] | None = None
        self._next_reset_seed = 0

    def collect_rollout(
        self, environment: Continuous2DSearchEnv
    ) -> tuple[list[RolloutStep], int, int]:
        """按当前策略采集固定长度轨迹，并在回合结束后自动重置。"""

        if self._observations is None:
            self._observations = environment.reset(seed=self._next_reset_seed)
            self._next_reset_seed += 1
        rollout: list[RolloutStep] = []
        episodes_finished = 0
        successful_episodes = 0

        for _ in range(self.config.rollout_steps):
            agent_ids = tuple(self._observations)
            observations = tuple(self._observations[agent_id] for agent_id in agent_ids)
            if not observations:
                self._observations = environment.reset(seed=self._next_reset_seed)
                self._next_reset_seed += 1
                continue
            with torch.no_grad():
                actions, log_probs, value = self.model.act(observations)
            action_map = {
                agent_id: Action.continuous(
                    agent_id,
                    actions[index].cpu().numpy(),
                    timestamp=environment.current_time,
                    metadata={"policy": "shared_graph_mappo"},
                )
                for index, agent_id in enumerate(agent_ids)
            }
            result = environment.step(action_map)
            team_reward = float(np.mean([result.rewards[key] for key in agent_ids]))
            done = bool(result.info["success"] or result.info["truncated"])
            rollout.append(
                RolloutStep(
                    agent_ids,
                    observations,
                    actions.detach().cpu(),
                    log_probs.detach().cpu(),
                    team_reward,
                    done,
                    float(value.item()),
                )
            )
            self.environment_steps += 1
            self._observations = result.observations
            if done:
                episodes_finished += 1
                successful_episodes += int(result.info["success"])
                self._observations = environment.reset(seed=self._next_reset_seed)
                self._next_reset_seed += 1

        if not rollout:
            raise RuntimeError("未能采集到有效轨迹，请检查环境是否存在在线智能体")
        return rollout, episodes_finished, successful_episodes

    def _advantages(self, rollout: Sequence[RolloutStep]) -> tuple[Tensor, Tensor]:
        """使用 GAE 计算团队优势和价值回报。"""

        with torch.no_grad():
            if rollout[-1].done or not self._observations:
                next_value = 0.0
            else:
                next_value = float(self.model.value(tuple(self._observations.values())).item())
        advantages = np.zeros(len(rollout), dtype=np.float32)
        gae = 0.0
        for index in reversed(range(len(rollout))):
            step = rollout[index]
            bootstrap = next_value if index == len(rollout) - 1 else rollout[index + 1].value
            nonterminal = 0.0 if step.done else 1.0
            delta = step.reward + self.config.gamma * bootstrap * nonterminal - step.value
            gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            advantages[index] = gae
        returns = advantages + np.asarray([step.value for step in rollout], dtype=np.float32)
        advantage_tensor = torch.as_tensor(advantages, device=self.device)
        if len(rollout) > 1:
            advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
                advantage_tensor.std(unbiased=False) + 1e-8
            )
        return advantage_tensor, torch.as_tensor(returns, device=self.device)

    def update(self, rollout: Sequence[RolloutStep]) -> dict[str, float]:
        """执行多轮 PPO 裁剪更新并返回实际损失均值。"""

        advantages, returns = self._advantages(rollout)
        totals = {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        for _ in range(self.config.update_epochs):
            actor_terms: list[Tensor] = []
            entropy_terms: list[Tensor] = []
            value_predictions: list[Tensor] = []
            for index, step in enumerate(rollout):
                new_log_probs, entropy, value = self.model.evaluate_actions(
                    step.observations, step.actions
                )
                old_log_probs = step.log_probs.to(self.device)
                ratio = (new_log_probs - old_log_probs).exp()
                unclipped = ratio * advantages[index]
                clipped = ratio.clamp(
                    1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                ) * advantages[index]
                actor_terms.append(-torch.minimum(unclipped, clipped))
                entropy_terms.append(entropy)
                value_predictions.append(value.reshape(()))

            actor_loss = torch.cat(actor_terms).mean()
            entropy_mean = torch.cat(entropy_terms).mean()
            value_loss = torch.nn.functional.mse_loss(
                torch.stack(value_predictions), returns
            )
            total_loss = (
                actor_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy_mean
            )
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            totals["actor_loss"] += float(actor_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy_mean.item())
            totals["total_loss"] += float(total_loss.item())

        return {key: value / self.config.update_epochs for key, value in totals.items()}

    def train(
        self, environment: Continuous2DSearchEnv, updates: int
    ) -> list[TrainingMetrics]:
        """执行若干次采样与更新；返回值均来自本次实际运行。"""

        if updates <= 0:
            raise ValueError("updates 必须为正整数")
        history: list[TrainingMetrics] = []
        for update_index in range(1, updates + 1):
            rollout, finished, successful = self.collect_rollout(environment)
            losses = self.update(rollout)
            history.append(
                TrainingMetrics(
                    update=update_index,
                    environment_steps=self.environment_steps,
                    mean_team_reward=float(np.mean([step.reward for step in rollout])),
                    episodes_finished=finished,
                    successful_episodes=successful,
                    **losses,
                )
            )
        return history

    def save_checkpoint(self, path: str | Path) -> Path:
        """保存模型、优化器及配置，便于中断后继续训练。"""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": self.model.architecture_config(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "environment_steps": self.environment_steps,
            },
            target,
        )
        return target

    def load_checkpoint(self, path: str | Path) -> None:
        """恢复模型、优化器和累计环境步数，用于继续同配置训练。"""

        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.environment_steps = int(checkpoint.get("environment_steps", 0))
