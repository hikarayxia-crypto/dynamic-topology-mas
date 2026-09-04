"""连续二维协同搜索环境与规则基线测试。"""

import unittest

import numpy as np

from agents import RuleBasedSearchAgent
from core import Action, AgentAttributes, Message
from environments import Continuous2DConfig, Continuous2DSearchEnv
from interaction import CommunicationConfig, TransmissionStatus


class Continuous2DEnvironmentTests(unittest.TestCase):
    def test_replacement_lane_tolerance_must_be_finite_and_non_negative(self) -> None:
        """真实位置判定容差不能为负数或非有限值。"""

        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Continuous2DConfig(replacement_lane_tolerance=value)

    def _environment(
        self,
        *,
        dt: float = 1.0,
        max_steps: int = 10,
        target_positions: tuple[tuple[float, float], ...] = ((0.0, 0.0),),
        sensor_range: float = 0.05,
        communication_range: float = 20.0,
        communication_config: CommunicationConfig | None = None,
    ) -> Continuous2DSearchEnv:
        environment = Continuous2DSearchEnv(
            Continuous2DConfig(
                width=10.0,
                height=10.0,
                dt=dt,
                max_steps=max_steps,
                target_positions=target_positions,
                collision_distance=0.1,
                energy_cost=0.0,
            ),
            communication_config,
        )
        attributes = AgentAttributes(
            agent_type="uav",
            max_speed=1.0,
            sensor_range=sensor_range,
            communication_range=communication_range,
        )
        environment.add_agent(RuleBasedSearchAgent("A", attributes))
        environment.add_agent(RuleBasedSearchAgent("B", attributes))
        return environment

    @staticmethod
    def _noop_actions(environment: Continuous2DSearchEnv) -> dict[str, Action]:
        return {
            agent_id: Action.noop(agent_id, timestamp=environment.current_time)
            for agent_id, agent in environment.agents.items()
            if agent.state.active
        }

    @staticmethod
    def _replacement_action(
        agent_id: str,
        missing_id: str,
        lane_y: float,
        known_missing: tuple[str, ...],
        *,
        score: float | None = 0.5,
        replacement_agent: str | None = None,
    ) -> Action:
        """构造环境跟踪测试使用的真实二维补位动作。"""

        return Action.continuous(
            agent_id,
            np.zeros(2),
            metadata={
                "known_missing": known_missing,
                "replacement_for": missing_id,
                "replacement_agent": replacement_agent or agent_id,
                "replacement_lane_y": lane_y,
                "replacement_bid_score": score,
            },
        )

    def _three_agent_environment(self) -> Continuous2DSearchEnv:
        """构造用于多缺失任务和补位切换测试的三节点环境。"""

        environment = Continuous2DSearchEnv(
            Continuous2DConfig(
                width=10.0,
                height=10.0,
                dt=1.0,
                max_steps=10,
                target_positions=((9.9, 9.9),),
                collision_distance=0.0,
                energy_cost=0.0,
            )
        )
        attributes = AgentAttributes(
            agent_type="uav",
            max_speed=1.0,
            sensor_range=0.1,
            communication_range=20.0,
        )
        for agent_id in ("A", "B", "C"):
            environment.add_agent(RuleBasedSearchAgent(agent_id, attributes))
        return environment

    def test_replacement_tracking_uses_real_position_not_declared_lane(self) -> None:
        """动作声明接管并不等于恢复覆盖，必须核对执行后的真实纵坐标。"""

        environment = self._environment(target_positions=((9.9, 9.9),))
        environment.reset(seed=21)
        environment.get_agent("A").state.position = np.array([1.0, 1.0])
        actions = self._noop_actions(environment)
        actions["A"] = self._replacement_action("A", "B", 8.0, ("B",))

        result = environment.step(actions)

        self.assertEqual(result.info["replacement_targets"], ("B",))
        self.assertEqual(result.info["replacement_active_count"], 1)
        self.assertFalse(result.info["replacement_coverage_restored"]["B"])

    def test_uncovered_ratio_switch_count_and_reset_snapshot(self) -> None:
        """只覆盖两个缺失带中的一个时比例为三分之一，换人只计一次切换。"""

        environment = self._three_agent_environment()
        environment.reset(seed=22)
        environment.get_agent("A").state.position = np.array([1.0, 8.0])
        first_actions = self._noop_actions(environment)
        first_actions["A"] = self._replacement_action("A", "B", 8.0, ("B", "C"), score=0.4)
        first = environment.step(first_actions)
        self.assertTrue(first.info["replacement_coverage_restored"]["B"])
        self.assertFalse(first.info["replacement_coverage_restored"]["C"])
        self.assertAlmostEqual(first.info["uncovered_lane_ratio"], 1.0 / 3.0)
        self.assertEqual(first.info["replacement_switches"], 0)

        environment.get_agent("C").state.position = np.array([2.0, 8.0])
        second_actions = self._noop_actions(environment)
        second_actions["C"] = self._replacement_action("C", "B", 8.0, ("B", "C"), score=0.3)
        second = environment.step(second_actions)
        snapshot = environment.replacement_snapshot()
        self.assertEqual(second.info["replacement_switches"], 1)
        self.assertEqual(snapshot["tasks"]["B"]["current_replacer"], "C")
        self.assertEqual(snapshot["tasks"]["B"]["switches"], 1)

        environment.reset(seed=23)
        self.assertEqual(
            environment.replacement_snapshot(),
            {"known_missing": (), "tasks": {}, "replacement_switches": 0},
        )

    def test_replacement_metadata_validation_and_stable_candidate_choice(self) -> None:
        """坏声明被忽略；同任务的合法候选按分数和标识稳定选择。"""

        invalid_metadata = (
            {"replacement_for": "unknown", "replacement_lane_y": 5.0},
            {"replacement_for": "A", "replacement_lane_y": 5.0},
            {"replacement_for": "B", "replacement_lane_y": float("nan")},
            {"replacement_for": "B", "replacement_lane_y": 11.0},
            {"replacement_for": "B", "replacement_lane_y": 5.0, "replacement_bid_score": float("inf")},
            {"replacement_for": "B", "replacement_lane_y": 5.0, "replacement_agent": "B"},
        )
        for invalid in invalid_metadata:
            with self.subTest(invalid=invalid):
                environment = self._environment(target_positions=((9.9, 9.9),))
                environment.reset(seed=24)
                actions = self._noop_actions(environment)
                actions["A"] = Action.continuous(
                    "A",
                    np.zeros(2),
                    metadata={"known_missing": ("B",), **invalid},
                )
                result = environment.step(actions)
                self.assertEqual(result.info["replacement_targets"], ())
                self.assertEqual(result.info["replacement_active_count"], 0)

        environment = self._three_agent_environment()
        environment.reset(seed=25)
        actions = self._noop_actions(environment)
        actions["A"] = self._replacement_action("A", "C", 5.0, ("C",), score=0.4)
        actions["B"] = self._replacement_action("B", "C", 5.0, ("C",), score=0.4)
        environment.step(actions)
        self.assertEqual(
            environment.replacement_snapshot()["tasks"]["C"]["current_replacer"],
            "A",
        )

    def test_replacement_metrics_keep_reset_roster_when_agent_is_added_mid_episode(self) -> None:
        """回合中新增节点不能进入初始缺失任务集合或改变未覆盖比例分母。"""

        environment = self._three_agent_environment()
        environment.reset(seed=26)
        environment.add_agent(RuleBasedSearchAgent("D"))
        actions = self._noop_actions(environment)
        actions["A"] = Action.continuous(
            "A",
            np.zeros(2),
            metadata={"known_missing": ("B", "D")},
        )
        result = environment.step(actions)
        self.assertEqual(result.info["known_missing"], ("B",))
        self.assertAlmostEqual(result.info["uncovered_lane_ratio"], 1.0 / 3.0)

    def test_replacement_action_counts_before_fault_begins_at_step_end(self) -> None:
        """已通过批校验并执行的动作不能因步末故障而从本步补位记录中消失。"""

        environment = self._environment(dt=0.2, target_positions=((9.9, 9.9),))
        environment.schedule_node_fault("A", start_time=0.2, duration=0.2)
        environment.reset(seed=27)
        environment.get_agent("A").state.position = np.array([1.0, 5.0])
        actions = self._noop_actions(environment)
        actions["A"] = self._replacement_action("A", "B", 5.0, ("B",))
        result = environment.step(actions)
        self.assertFalse(environment.get_agent("A").state.active)
        self.assertEqual(result.info["replacement_targets"], ("B",))
        self.assertTrue(result.info["replacement_coverage_restored"]["B"])

    def test_reset_builds_variable_neighbor_observations_without_target_leak(self) -> None:
        environment = self._environment()
        observations = environment.reset(seed=5)
        environment.get_agent("A").state.position = np.array([1.0, 1.0])
        environment.get_agent("B").state.position = np.array([2.0, 1.0])
        result = environment.step(self._noop_actions(environment))
        observation = result.observations["A"]
        self.assertEqual(observation.self_features.shape, (6,))
        self.assertEqual(observation.neighbor_matrix().shape, (1, 5))
        self.assertNotIn("target_positions", observation.metadata)
        self.assertEqual(set(observations), {"A", "B"})

    def test_distance_change_updates_topology(self) -> None:
        environment = self._environment(communication_range=3.0)
        environment.reset(seed=1)
        environment.get_agent("A").state.position = np.array([1.0, 1.0])
        environment.get_agent("B").state.position = np.array([2.0, 1.0])
        environment.step(self._noop_actions(environment))
        self.assertTrue(environment.topology.are_directly_connected("A", "B"))

        environment.get_agent("B").state.position = np.array([9.0, 9.0])
        environment.step(self._noop_actions(environment))
        self.assertFalse(environment.topology.are_directly_connected("A", "B"))

    def test_link_fault_disconnects_and_recovers(self) -> None:
        environment = self._environment()
        environment.schedule_link_fault("A", "B", start_time=1.0, duration=1.0)
        environment.reset(seed=2)
        environment.get_agent("A").state.position = np.array([2.0, 2.0])
        environment.get_agent("B").state.position = np.array([3.0, 2.0])

        environment.step(self._noop_actions(environment))
        self.assertFalse(environment.topology.are_directly_connected("A", "B"))
        environment.step(self._noop_actions(environment))
        self.assertTrue(environment.topology.are_directly_connected("A", "B"))

    def test_node_fault_changes_required_action_set_and_recovers(self) -> None:
        environment = self._environment()
        environment.schedule_node_fault("B", start_time=1.0, duration=1.0)
        environment.reset(seed=3)
        first = environment.step(self._noop_actions(environment))
        self.assertNotIn("B", first.observations)
        self.assertFalse(environment.get_agent("B").state.active)

        second = environment.step(self._noop_actions(environment))
        self.assertIn("B", second.observations)
        self.assertTrue(environment.get_agent("B").state.active)

    def test_target_detection_produces_success_and_shared_reward(self) -> None:
        environment = self._environment(target_positions=((1.0, 1.0),), sensor_range=0.5)
        environment.reset(seed=4)
        environment.get_agent("A").state.position = np.array([1.0, 1.0])
        environment.get_agent("B").state.position = np.array([8.0, 8.0])
        result = environment.step(self._noop_actions(environment))
        self.assertTrue(result.info["success"])
        self.assertEqual(result.info["targets_discovered"], 1)
        self.assertTrue(all(result.terminated.values()))
        self.assertGreater(result.rewards["A"], 0.0)

    def test_time_limit_truncates_without_fabricating_success(self) -> None:
        environment = self._environment(max_steps=1)
        environment.reset(seed=6)
        environment.get_agent("A").state.position = np.array([5.0, 5.0])
        environment.get_agent("B").state.position = np.array([6.0, 5.0])
        result = environment.step(self._noop_actions(environment))
        self.assertFalse(result.info["success"])
        self.assertTrue(result.info["truncated"])
        self.assertTrue(all(result.truncated.values()))

    def test_environment_routes_messages_through_current_topology(self) -> None:
        environment = self._environment(
            communication_config=CommunicationConfig(base_delay=0.5)
        )
        environment.reset(seed=7)
        environment.get_agent("A").state.position = np.array([2.0, 2.0])
        environment.get_agent("B").state.position = np.array([3.0, 2.0])
        environment.step(self._noop_actions(environment))

        queued = environment.route_message(
            Message("A", "B", "status", created_at=environment.current_time)
        )[0]
        self.assertEqual(queued.status, TransmissionStatus.QUEUED)
        environment.step(self._noop_actions(environment))
        received = environment.get_agent("B").pop_messages()
        self.assertEqual(received[0].payload, "status")

    def test_rule_agent_generates_valid_two_dimensional_action(self) -> None:
        environment = self._environment()
        observations = environment.reset(seed=8)
        action = environment.get_agent("A").step(observations["A"], 1.0)
        self.assertEqual(action.value.shape, (2,))
        self.assertLessEqual(np.linalg.norm(action.value), 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
