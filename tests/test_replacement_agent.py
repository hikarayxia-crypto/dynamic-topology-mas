"""补位扫描智能体的可观察行为测试。"""

from __future__ import annotations

import unittest

import numpy as np

from agents import ReplacementSearchAgent, RuleBasedSearchAgent
from coordination.replacement import NodeLiveness, ReplacementConfig
from core import AgentAttributes, AgentState, Message, NeighborObservation, Observation
from environments import Continuous2DConfig, Continuous2DSearchEnv
from interaction import CommunicationConfig
from scripts.run_replacement_demo import run_demo


class _Environment:
    """记录广播请求的最小环境替身，保留 BaseAgent 的真实消息接口。"""

    def __init__(self, *, reject: bool = False) -> None:
        self.current_time = 0.0
        self.reject = reject
        self.sent: list[Message] = []

    def route_message(self, message: Message) -> bool:
        self.sent.append(message)
        if self.reject:
            raise RuntimeError("链路拒绝")
        return True

    def apply_action(self, agent_id: str, action: object) -> None:
        return None


def _observation(
    timestamp: float,
    *,
    position: tuple[float, float] = (1.0, 1.0),
    neighbors: tuple[NeighborObservation, ...] = (),
    all_targets_discovered: bool = False,
) -> Observation:
    """构造不含目标真值、但满足规则控制所需元数据的观测。"""

    return Observation(
        "A",
        np.zeros(1),
        neighbors=neighbors,
        timestamp=timestamp,
        metadata={
            "world_size": (10.0, 10.0),
            "position": position,
            "agent_index": 0,
            "agent_count": 2,
            "sensor_range": 1.0,
            "all_targets_discovered": all_targets_discovered,
        },
    )


class RuleSweepCharacterizationTests(unittest.TestCase):
    """保护重构前规则扫描的公开方向结果。"""

    def test_sweep_command_keeps_right_upper_target_direction_and_unit_norm(self) -> None:
        """若扫描目标在右上，两个分量应为正且命令恰为单位向量。"""

        agent = RuleBasedSearchAgent("A")
        command = agent._sweep_command(_observation(0.0, position=(1.0, 1.0)), 1.0)
        self.assertGreater(command[0], 0.0)
        self.assertGreater(command[1], 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(command)), 1.0)

    def test_sweep_command_keeps_right_lower_target_direction(self) -> None:
        """若扫描带在当前位置下方，横向为正且纵向为负。"""

        agent = RuleBasedSearchAgent("A")
        command = agent._sweep_command(
            _observation(0.0, position=(1.0, 9.0)), 1.0, target_y=2.5
        )
        self.assertGreater(command[0], 0.0)
        self.assertLess(command[1], 0.0)

    def test_sweep_command_preserves_boundary_turn_and_neighbor_repulsion(self) -> None:
        """折返后应朝左，过近上方邻居还必须把命令向下推开。"""

        boundary_agent = RuleBasedSearchAgent("A")
        boundary = boundary_agent._sweep_command(
            _observation(0.0, position=(10.0, 2.5)), 1.0
        )
        self.assertLess(boundary[0], 0.0)
        self.assertGreater(boundary[1], 0.0)

        repulsion_agent = RuleBasedSearchAgent("A")
        neighbor = NeighborObservation(
            "B", np.zeros(1), metadata={"relative_position": [0.0, 0.1], "distance": 0.1}
        )
        repelled = repulsion_agent._sweep_command(
            _observation(0.0, position=(1.0, 2.5), neighbors=(neighbor,)), 1.0
        )
        self.assertLess(repelled[1], 0.0)
        self.assertLessEqual(float(np.linalg.norm(repelled)), 1.0)


class ReplacementSearchAgentTests(unittest.TestCase):
    """验证补位控制只依赖本地观测、协调结果和通信接口。"""

    def _agent(self, **config_values: object) -> ReplacementSearchAgent:
        config = ReplacementConfig(
            failure_timeout=0.0,
            failure_confirmation=0.0,
            recovery_stability=0.0,
            bid_window=0.0,
            broadcast_interval=1.0,
            dwell_steps=2,
            **config_values,
        )
        return ReplacementSearchAgent("A", ("A", "B", "C", "D"), 10.0, replacement_config=config)

    @staticmethod
    def _mark_missing(agent: ReplacementSearchAgent, *node_ids: str) -> None:
        """用协调器公开状态机建立已确认缺失任务，不伪造环境赢家。"""

        agent.coordinator.advance_time(0.1)
        for node_id in agent.coordinator.roster:
            if node_id not in {str(value) for value in node_ids} and node_id != "A":
                agent.coordinator.ingest_gossip(
                    {
                        "kind": "replacement_gossip",
                        "sender": node_id,
                        "sent_at": 0.1,
                        "heartbeats": {node_id: 0.1},
                        "bids": {},
                    },
                    0.1,
                )
        agent.coordinator.advance_time(0.2)
        for node_id in agent.coordinator.roster:
            if node_id not in {str(value) for value in node_ids} and node_id != "A":
                agent.coordinator.ingest_gossip(
                    {
                        "kind": "replacement_gossip",
                        "sender": node_id,
                        "sent_at": 0.2,
                        "heartbeats": {node_id: 0.2},
                        "bids": {},
                    },
                    0.2,
                )
        for node_id in node_ids:
            assert agent.coordinator.liveness_of(node_id) is NodeLiveness.MISSING

    def test_winner_moves_toward_missing_lane_and_exposes_exact_metadata(self) -> None:
        """赢家接管 B 的搜索带时，应向其中心线移动并报告来源和竞价分数。"""

        agent = self._agent()
        self._mark_missing(agent, "B")
        observation = _observation(0.2, position=(1.0, 1.0))
        agent.perceive(observation)
        action = agent.decide(1.0)

        self.assertGreater(action.value[1], 0.0)
        self.assertEqual(action.metadata["policy"], "replacement_sweep")
        self.assertEqual(action.metadata["replacement_for"], "B")
        self.assertEqual(action.metadata["replacement_agent"], "A")
        self.assertEqual(action.metadata["known_missing"], ("B",))
        self.assertEqual(action.metadata["replacement_lane_y"], 3.75)
        self.assertIsInstance(action.metadata["replacement_bid_score"], float)

    def test_multiple_assignments_rotate_after_exact_dwell_steps(self) -> None:
        """两个补位任务每个严格驻留两次，顺序为 C、C、D、D。"""

        agent = self._agent()
        self._mark_missing(agent, "C", "D")
        agent.perceive(_observation(0.2))
        observed = [agent.decide(1.0).metadata["replacement_for"] for _ in range(4)]
        self.assertEqual(observed, ["C", "C", "D", "D"])

    def test_second_missing_bid_includes_assignment_load_from_first_bid(self) -> None:
        """同一轮 C、D 缺失时，D 竞价要计入刚赢得 C 的一项负载。"""

        agent = self._agent()
        self._mark_missing(agent, "C", "D")
        agent.perceive(_observation(0.2, position=(1.0, 1.0)))
        agent.decide(1.0)
        self.assertEqual(agent.coordinator.known_bids["D"]["A"].score, 0.77625)

    def test_recovery_hands_back_only_after_current_dwell_cycle(self) -> None:
        """当前驻留任务恢复后先用缓存完成周期，随后回归普通规则扫描。"""

        agent = self._agent()
        self._mark_missing(agent, "B")
        agent.perceive(_observation(0.2))
        self.assertEqual(agent.decide(1.0).metadata["replacement_for"], "B")
        agent.coordinator.ingest_gossip(
            {
                "kind": "replacement_gossip",
                "sender": "B",
                "sent_at": 0.2,
                "heartbeats": {"B": 0.2},
                "bids": {},
            },
            0.2,
        )
        agent.coordinator.advance_time(0.2)
        retained = agent.decide(1.0)
        returned = agent.decide(1.0)
        self.assertEqual(retained.metadata["replacement_for"], "B")
        # 平滑驻留动作仍执行，但节点已经 HEALTHY，不再计入环境的缺失任务分母。
        self.assertEqual(retained.metadata["known_missing"], ())
        self.assertNotIn("replacement_for", returned.metadata)
        self.assertEqual(returned.metadata["known_missing"], ())

    def test_perceive_consumes_only_valid_gossip_and_broadcasts_on_simulation_schedule(self) -> None:
        """合法 gossip 合并；早于间隔不广播，达到间隔才广播一次。"""

        agent = self._agent()
        environment = _Environment()
        agent.bind_environment(environment)
        agent.receive_message(Message("B", "A", {"kind": "not_protocol"}, 0.0))
        agent.receive_message(
            Message("B", "A", {"kind": "replacement_gossip", "sender": "B", "sent_at": 0.0, "heartbeats": {"B": 0.0}, "bids": {}}, 0.0)
        )
        agent.perceive(_observation(0.0))
        self.assertEqual(agent.coordinator.liveness_of("B"), NodeLiveness.HEALTHY)
        agent.decide(1.0)
        agent.perceive(_observation(0.5))
        agent.decide(1.0)
        agent.perceive(_observation(1.0))
        agent.decide(1.0)
        self.assertEqual(len(environment.sent), 2)
        self.assertTrue(all(item.payload["kind"] == "replacement_gossip" for item in environment.sent))

    def test_partition_and_rejected_broadcast_still_produce_bounded_action(self) -> None:
        """无邻居且发送被拒绝时，本地竞价和二维控制仍可继续。"""

        agent = self._agent()
        agent.bind_environment(_Environment(reject=True))
        self._mark_missing(agent, "B")
        agent.perceive(_observation(1.0))
        action = agent.decide(1.0)
        self.assertEqual(action.value.shape, (2,))
        self.assertTrue(np.all(np.isfinite(action.value)))
        self.assertLessEqual(float(np.linalg.norm(action.value)), 1.0 + 1e-12)
        self.assertEqual(action.metadata["replacement_for"], "B")

    def test_noop_skips_replacement_side_effects(self) -> None:
        """任务完成或本机离线时，必须保持原有 noop 且不发送消息。"""

        agent = self._agent()
        environment = _Environment()
        agent.bind_environment(environment)
        agent.perceive(_observation(1.0, all_targets_discovered=True))
        self.assertIsNone(agent.decide(1.0).value)
        self.assertEqual(environment.sent, [])
        agent.state.active = False
        agent.perceive(_observation(2.0))
        self.assertIsNone(agent.decide(1.0).value)
        self.assertEqual(environment.sent, [])

    def test_environment_fault_and_recovery_use_public_step_and_handover(self) -> None:
        """真实环境故障后 A 补 B，B 稳定恢复并完成驻留周期后交还。"""

        config = ReplacementConfig(
            failure_timeout=0.2,
            failure_confirmation=0.0,
            bid_window=0.0,
            recovery_stability=0.2,
            broadcast_interval=0.1,
            dwell_steps=2,
        )
        environment = Continuous2DSearchEnv(
            Continuous2DConfig(
                width=10.0, height=10.0, dt=0.1, max_steps=20,
                target_positions=((9.9, 9.9),),
                collision_distance=0.0, energy_cost=0.0,
            ),
            CommunicationConfig(),
        )
        attributes = AgentAttributes(
            agent_type="uav", max_speed=1.0, sensor_range=0.01, communication_range=20.0
        )
        environment.add_agent(ReplacementSearchAgent("A", ("A", "B"), 10.0, attributes, replacement_config=config))
        environment.add_agent(ReplacementSearchAgent("B", ("A", "B"), 10.0, attributes, replacement_config=config))
        environment.schedule_node_fault("B", start_time=0.2, duration=0.6)
        observations = environment.reset(seed=17)
        replacement_actions: list[object] = []
        handover_actions: list[object] = []
        recovery_states: list[NodeLiveness] = []
        replacement_started = False
        for _ in range(18):
            actions = {
                agent_id: environment.get_agent(agent_id).step(observation, 0.1)
                for agent_id, observation in observations.items()
            }
            action_a = actions.get("A")
            if action_a is not None and "replacement_for" in action_a.metadata:
                replacement_actions.append(action_a)
                replacement_started = True
            state_b = environment.get_agent("A").coordinator.liveness_of("B")
            if replacement_started and state_b in {
                NodeLiveness.RECOVERING,
                NodeLiveness.HEALTHY,
            }:
                recovery_states.append(state_b)
            if (
                NodeLiveness.HEALTHY in recovery_states
                and action_a is not None
                and "replacement_for" not in action_a.metadata
            ):
                # 只记录本轮真正提交的动作，不能额外调用 step 消耗驻留计数。
                handover_actions.append(action_a)
            result = environment.step(actions)
            observations = result.observations
            if handover_actions:
                break
        self.assertTrue(replacement_actions)
        self.assertIn(NodeLiveness.RECOVERING, recovery_states)
        self.assertIn(NodeLiveness.HEALTHY, recovery_states)
        self.assertTrue(handover_actions)

    def test_replacement_demo_reports_observed_results(self) -> None:
        """快速演示应返回真实事件计数，而不是预设理想补位结果。"""

        result = run_demo(max_steps=30, seed=7)
        self.assertGreater(result["steps"], 0)
        self.assertLessEqual(result["steps"], 30)
        self.assertGreaterEqual(result["confirmed_missing_tasks"], 0)
        self.assertGreaterEqual(result["replacement_responses"], 0)
        self.assertGreaterEqual(result["coverage_recoveries"], 0)
        self.assertLessEqual(
            result["coverage_recoveries"], result["confirmed_missing_tasks"]
        )
        self.assertGreaterEqual(result["packet_loss_rate"], 0.0)
        self.assertLessEqual(result["packet_loss_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
