"""图编码器、共享策略和 MAPPO 最小更新测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agents import RuleBasedSearchAgent, SharedPolicyAgent
from algorithms import GraphObservationEncoder, SharedGraphActorCritic, batch_observations
from core import AgentAttributes
from core.observation import NeighborObservation, Observation
from environments import Continuous2DConfig, Continuous2DSearchEnv
from training import MAPPOConfig, MAPPOTrainer


class GraphLearningTests(unittest.TestCase):
    @staticmethod
    def _observation(neighbors: tuple[NeighborObservation, ...]) -> Observation:
        return Observation(
            agent_id="A",
            self_features=np.arange(6, dtype=float) / 10,
            neighbors=neighbors,
            task_features=np.array([0.2, 0.8]),
            neighbor_feature_dim=5,
        )

    def test_graph_encoder_is_neighbor_order_invariant(self) -> None:
        first = NeighborObservation("B", np.array([1, 0, 0, 0, 1]), 0.8)
        second = NeighborObservation("C", np.array([0, 1, 0, 0, 1]), 0.4)
        encoder = GraphObservationEncoder(6, 5, 2, hidden_dim=16)
        batch = batch_observations(
            [self._observation((first, second)), self._observation((second, first))]
        )
        output = encoder(batch)
        self.assertTrue(torch.allclose(output[0], output[1], atol=1e-6))

    def test_graph_encoder_handles_isolated_agent(self) -> None:
        encoder = GraphObservationEncoder(6, 5, 2, hidden_dim=8)
        output = encoder(batch_observations([self._observation(())]))
        self.assertEqual(tuple(output.shape), (1, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_shared_policy_action_shape_and_bounds(self) -> None:
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=16)
        actions, log_probs, value = model.act([self._observation(())])
        self.assertEqual(tuple(actions.shape), (1, 2))
        self.assertEqual(tuple(log_probs.shape), (1,))
        self.assertEqual(value.ndim, 0)
        self.assertLessEqual(float(actions.detach().abs().max()), 1.0)

    def test_learning_agent_uses_shared_model(self) -> None:
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=8)
        agent = SharedPolicyAgent("A", model)
        action = agent.step(self._observation(()), 0.2)
        self.assertEqual(action.value.shape, (2,))

    def test_tiny_mappo_update_and_checkpoint(self) -> None:
        torch.manual_seed(3)
        model = SharedGraphActorCritic(6, 5, 2, hidden_dim=16)
        environment = Continuous2DSearchEnv(
            Continuous2DConfig(
                width=10,
                height=10,
                dt=1,
                max_steps=3,
                target_positions=((0.0, 0.0),),
                energy_cost=0,
            )
        )
        attributes = AgentAttributes(
            max_speed=1, sensor_range=0.01, communication_range=20
        )
        environment.add_agent(RuleBasedSearchAgent("A", attributes))
        environment.add_agent(RuleBasedSearchAgent("B", attributes))
        # 训练轨迹中在线智能体数从 2 变 1 再恢复，验证算法不依赖固定规模。
        environment.schedule_node_fault("B", start_time=1.0, duration=1.0)
        trainer = MAPPOTrainer(
            model,
            MAPPOConfig(rollout_steps=4, update_epochs=1),
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics = trainer.train(environment, updates=1)[0]
        changed = any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, model.parameters())
        )
        self.assertTrue(changed)
        self.assertTrue(np.isfinite(metrics.total_loss))
        self.assertEqual(metrics.environment_steps, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save_checkpoint(Path(directory) / "model.pt")
            self.assertTrue(checkpoint.exists())
            saved = torch.load(checkpoint, weights_only=False)
            self.assertIn("model", saved)
            restored = MAPPOTrainer(
                SharedGraphActorCritic(6, 5, 2, hidden_dim=16),
                MAPPOConfig(rollout_steps=4, update_epochs=1),
            )
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.environment_steps, 4)


if __name__ == "__main__":
    unittest.main()
