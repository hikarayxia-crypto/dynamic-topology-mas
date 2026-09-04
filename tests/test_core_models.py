"""动作、观测、智能体和环境基础接口测试。"""

from typing import Any, Hashable, Mapping
import unittest

import numpy as np

from core import (
    Action,
    AgentState,
    BaseAgent,
    BaseEnvironment,
    Message,
    NeighborObservation,
    Observation,
    StepResult,
)


class FixedAgent(BaseAgent):
    """测试使用的固定动作智能体。"""

    def perceive(self, observation: Observation) -> np.ndarray:
        self.features = observation.mean_pooled_vector()
        return self.features

    def decide(self, dt: float) -> Action:
        return Action.continuous(self.id, [dt, 0.0])


class MinimalEnvironment(BaseEnvironment):
    """只验证基类生命周期的最小环境，不实现具体任务。"""

    def reset(self, seed: int | None = None) -> Mapping[Hashable, Observation]:
        return {
            agent_id: Observation(
                agent_id=agent_id,
                self_features=agent.state.as_vector(),
                neighbor_feature_dim=agent.state.as_vector().size,
            )
            for agent_id, agent in self.agents.items()
            if agent.state.active
        }

    def step(self, actions: Mapping[Hashable, Action]) -> StepResult:
        self.validate_action_batch(actions)
        return StepResult({}, {}, {}, {}, {"action_count": len(actions)})

    def route_message(self, message: Message) -> Message:
        return message

    def apply_action(self, agent_id: Hashable, action: Action) -> Action:
        return action


class CoreModelTests(unittest.TestCase):
    def test_continuous_action_copies_input(self) -> None:
        values = np.array([1.0, -1.0])
        action = Action.continuous("A", values)
        values[0] = 99.0
        np.testing.assert_allclose(action.value, [1.0, -1.0])

    def test_invalid_action_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Action.continuous("A", [np.nan, 0.0])
        with self.assertRaises(ValueError):
            Action.discrete("A", True)

    def test_variable_neighbor_observation_and_pooling(self) -> None:
        observation = Observation(
            agent_id="A",
            self_features=np.array([1.0, 2.0]),
            neighbors=(
                NeighborObservation("B", np.array([2.0, 4.0]), link_weight=0.5),
                NeighborObservation("C", np.array([4.0, 8.0]), link_weight=1.0),
            ),
            task_features=np.array([3.0]),
            topology_version=4,
        )
        self.assertEqual(observation.neighbor_matrix().shape, (2, 2))
        np.testing.assert_allclose(
            observation.mean_pooled_vector(), [1.0, 2.0, 3.0, 6.0, 3.0]
        )

    def test_empty_neighbors_keep_declared_feature_shape(self) -> None:
        observation = Observation(
            agent_id="A",
            self_features=[1.0],
            neighbor_feature_dim=3,
        )
        self.assertEqual(observation.neighbor_matrix().shape, (0, 3))
        np.testing.assert_allclose(observation.mean_pooled_vector(), [1, 0, 0, 0])

    def test_agent_state_arrays_are_not_shared(self) -> None:
        first = AgentState()
        second = AgentState()
        first.position[0] = 5.0
        self.assertEqual(second.position[0], 0.0)
        first.pos = np.array([2.0, 3.0])
        np.testing.assert_allclose(first.position, [2.0, 3.0])

    def test_agent_step_checks_observation_and_action_owner(self) -> None:
        agent = FixedAgent("A")
        observation = Observation(
            agent_id="A", self_features=[0.0], neighbor_feature_dim=1
        )
        action = agent.step(observation, 0.1)
        self.assertEqual(action.agent_id, "A")
        with self.assertRaises(ValueError):
            agent.step(
                Observation(agent_id="B", self_features=[0.0]),
                0.1,
            )

    def test_environment_keeps_agent_and_topology_lifecycle_consistent(self) -> None:
        environment = MinimalEnvironment()
        agent_a = FixedAgent("A")
        agent_b = FixedAgent("B")
        environment.add_agent(agent_a)
        environment.add_agent(agent_b)
        environment.connect_agents("A", "B")
        self.assertTrue(environment.topology.are_directly_connected("A", "B"))

        environment.set_agent_active("B", False)
        self.assertFalse(agent_b.state.active)
        self.assertFalse(environment.topology.is_node_active("B"))
        with self.assertRaises(ValueError):
            environment.validate_action_batch(
                {"A": Action.noop("A"), "B": Action.noop("B")}
            )

        removed = environment.remove_agent("B")
        self.assertIs(removed, agent_b)
        self.assertIsNone(agent_b.environment)
        self.assertFalse(environment.topology.has_node("B"))

        routed = environment.route_msg("A", "A", "compatibility")
        self.assertEqual(routed.payload, "compatibility")


if __name__ == "__main__":
    unittest.main()
