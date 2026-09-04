import unittest
from math import nextafter

from coordination.replacement import (
    NodeLiveness,
    ReplacementConfig,
    ReplacementCoordinator,
    ReplacementBid,
    build_coverage_lanes,
)


class ReplacementCoordinationTests(unittest.TestCase):
    def test_reconnect_accepts_peer_heartbeat_with_stale_known_task_bid(self) -> None:
        """分区双方视图不一致时，已知成员的过时任务竞价不能阻塞恢复心跳。"""

        config = ReplacementConfig(
            failure_timeout=0.2,
            failure_confirmation=0.0,
            bid_window=0.0,
        )
        coordinator = ReplacementCoordinator("A", ("A", "B"), 10.0, config)
        coordinator.advance_time(0.3)
        coordinator.advance_time(0.3)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.MISSING)

        reconnect_gossip = {
            "kind": "replacement_gossip",
            "sender": "B",
            "sent_at": 0.9,
            "heartbeats": {"A": 0.0, "B": 0.9},
            # B 在分区期间把 A 视为缺失；A 本地知道 A 健康，因此合并后应丢弃该竞价。
            "bids": {
                "A": {
                    "bidder_id": "B",
                    "score": 0.2,
                    "created_at": 0.9,
                    "expires_at": 2.9,
                }
            },
        }

        self.assertTrue(coordinator.ingest_gossip(reconnect_gossip, received_at=1.0))
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.RECOVERING)
        self.assertNotIn("A", coordinator.known_bids)

    def test_local_bid_uses_configured_cost_formula(self) -> None:
        """竞价分数必须由搜索带距离、负载、连通度和能量的手工公式得出。"""
        coordinator = ReplacementCoordinator("A", ("A", "B"), 10.0, ReplacementConfig())
        coordinator.update_local_status(position_y=4.5, energy=0.8, neighbor_count=1, timestamp=0.0)
        coordinator.advance_time(1.1)
        coordinator.advance_time(1.8)

        bid = coordinator.local_bid_for("B", current_load=1, timestamp=1.8)

        self.assertAlmostEqual(bid.score, 0.55 * 0.3 + 0.20 + 0.15 * 0.5 + 0.10 * 0.2)
        self.assertEqual(bid.expires_at, 3.8)

    def test_assignment_waits_for_window_and_breaks_first_tie_by_id(self) -> None:
        """任务窗口前不分配，到期后同分候选按字符串标识稳定裁决。"""
        config = ReplacementConfig(bid_window=0.4)
        coordinator = ReplacementCoordinator("A", ("A", "B", "C"), 9.0, config)
        coordinator.ingest_gossip(self._gossip("B", 1.0, bids={}), 1.0)
        coordinator.advance_time(1.1)
        coordinator.advance_time(1.8)
        coordinator.ingest_gossip(self._gossip("A", 1.8, bids={"C": self._bid("A", 0.4, 1.8, 3.8)}), 1.8)
        coordinator.ingest_gossip(self._gossip("B", 1.8, bids={"C": self._bid("B", 0.4, 1.8, 3.8)}), 1.8)

        coordinator.advance_time(2.199999)
        self.assertIsNone(coordinator.assignment_for("C"))
        coordinator.advance_time(2.2)
        self.assertEqual(coordinator.assignment_for("C").winner_id, "A")

    def test_switch_margin_and_reassignment_count(self) -> None:
        """已有赢家只有超过切换裕量才替换，且仅实际切换计数。"""
        coordinator = ReplacementCoordinator("A", ("A", "B", "C"), 9.0, ReplacementConfig(bid_window=0.0, switch_margin=0.05))
        coordinator.ingest_gossip(self._gossip("B", 1.0, bids={}), 1.0)
        coordinator.advance_time(1.1)
        coordinator.advance_time(1.8)
        coordinator.ingest_gossip(self._gossip("A", 1.8, bids={"C": self._bid("A", .50, 1.8, 3.8)}), 1.8)
        coordinator.advance_time(1.8)
        coordinator.ingest_gossip(self._gossip("B", 1.9, bids={"C": self._bid("B", .48, 1.9, 3.9)}), 1.9)
        coordinator.advance_time(1.9)
        self.assertEqual(coordinator.assignment_for("C").winner_id, "A")
        self.assertEqual(coordinator.assignment_switch_count, 0)
        coordinator.ingest_gossip(self._gossip("B", 2.0, bids={"C": self._bid("B", .44, 2.0, 4.0)}), 2.0)
        coordinator.advance_time(2.0)
        self.assertEqual(coordinator.assignment_for("C").winner_id, "B")
        self.assertEqual(coordinator.assignment_switch_count, 1)

    @staticmethod
    def _bid(bidder_id, score, created_at, expires_at):
        return {"bidder_id": bidder_id, "score": score, "created_at": created_at, "expires_at": expires_at}

    @staticmethod
    def _gossip(sender, sent_at, bids):
        return {"kind": "replacement_gossip", "sender": sender, "sent_at": sent_at, "heartbeats": {sender: sent_at}, "bids": bids}

    def test_gossip_uses_one_best_bid_and_forwards_it_multiple_hops(self) -> None:
        """入站的固定竞价负载必须原样成为下一跳可广播的最佳候选。"""
        coordinator = ReplacementCoordinator("A", ("A", "B", "C", "D"), 8.0, ReplacementConfig())
        coordinator.advance_time(1.1)
        coordinator.advance_time(1.8)
        message = self._gossip("B", 1.8, {"D": self._bid("B", 0.3, 1.8, 3.8)})
        message["heartbeats"]["C"] = 1.8

        self.assertTrue(coordinator.ingest_gossip(message, 1.8))
        forwarded = coordinator.build_gossip()
        self.assertEqual(set(forwarded), {"kind", "sender", "sent_at", "heartbeats", "bids"})
        self.assertEqual(forwarded["bids"]["D"], self._bid("B", 0.3, 1.8, 3.8))
        self.assertEqual(forwarded["heartbeats"]["C"], 1.8)

    def test_gossip_rejects_bad_bids_and_extra_top_level_fields_atomically(self) -> None:
        """无效竞价或非协议顶层字段必须整包拒绝，不能留下部分心跳或竞价。"""
        coordinator = ReplacementCoordinator("A", ("A", "B", "C"), 6.0, ReplacementConfig())
        coordinator.advance_time(1.1)
        coordinator.advance_time(1.8)
        baseline = (dict(coordinator._last_heartbeats), coordinator.known_bids, coordinator.assignments)
        bad_payloads = (
            self._bid("B", 0.2, 1.8, 1.8),
            self._bid("B", float("nan"), 1.8, 3.8),
            self._bid("B", 0.2, 2.0, 4.0),
            self._bid("X", 0.2, 1.8, 3.8),
            self._bid("C", 0.2, 1.8, 3.8),
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                self.assertFalse(coordinator.ingest_gossip(self._gossip("B", 1.8, {"C": payload}), 1.8))
                self.assertEqual((dict(coordinator._last_heartbeats), coordinator.known_bids, coordinator.assignments), baseline)
        extra = self._gossip("B", 1.8, {})
        extra["extra"] = True
        self.assertFalse(coordinator.ingest_gossip(extra, 1.8))
        self.assertEqual((dict(coordinator._last_heartbeats), coordinator.known_bids, coordinator.assignments), baseline)

    def test_gossip_rejects_fields_newer_than_sent_time_atomically(self) -> None:
        """发送时刻后的心跳或竞价不能被消息携带，且整包必须保持原子拒绝。"""
        coordinator = ReplacementCoordinator("A", ("A", "B", "C"), 6.0, ReplacementConfig())
        coordinator.advance_time(1.1)
        coordinator.advance_time(1.8)
        baseline = (dict(coordinator._last_heartbeats), coordinator.known_bids, coordinator.assignments)
        future_heartbeat = self._gossip("B", 1.8, {})
        future_heartbeat["heartbeats"]["B"] = 1.9
        future_bid = self._gossip("B", 1.8, {"C": self._bid("B", 0.2, 1.9, 3.9)})
        future_bid["heartbeats"]["B"] = 1.8

        for message in (future_heartbeat, future_bid):
            with self.subTest(message=message):
                self.assertFalse(coordinator.ingest_gossip(message, 2.0))
                self.assertEqual((dict(coordinator._last_heartbeats), coordinator.known_bids, coordinator.assignments), baseline)

    def test_gossip_rejects_even_tiny_ttl_mismatch_atomically(self) -> None:
        """到期时刻必须严格等于本配置的创建时刻加 TTL，微小偏差也不能合并。"""
        coordinator = ReplacementCoordinator("A", ("A", "B", "C"), 6.0, ReplacementConfig())
        coordinator.advance_time(1.1)
        coordinator.advance_time(1.8)
        baseline = (dict(coordinator._last_heartbeats), coordinator.known_bids, coordinator.assignments)
        for expires_at in (3.8000000000001, 3.7999999999999):
            message = self._gossip("B", 1.8, {"C": self._bid("B", 0.2, 1.8, expires_at)})
            with self.subTest(expires_at=expires_at):
                self.assertFalse(coordinator.ingest_gossip(message, 2.0))
                self.assertEqual((dict(coordinator._last_heartbeats), coordinator.known_bids, coordinator.assignments), baseline)

    def test_expired_winner_reassignment_counts_switch_and_healthy_cleans_task(self) -> None:
        """赢家竞价失效后的接管算切换，原节点稳定健康后必须删除整个任务。"""
        config = ReplacementConfig(bid_window=0.0, bid_ttl=0.5, recovery_stability=0.2)
        coordinator = ReplacementCoordinator("A", ("A", "B", "C"), 6.0, config)
        coordinator.ingest_gossip(self._gossip("B", 1.0, {}), 1.0)
        coordinator.advance_time(1.1)
        coordinator.advance_time(1.8)
        coordinator.ingest_gossip(self._gossip("A", 1.8, {"C": self._bid("A", 0.1, 1.8, 2.3)}), 1.8)
        coordinator.ingest_gossip(self._gossip("B", 2.0, {"C": self._bid("B", 0.2, 2.0, 2.5)}), 2.0)
        self.assertEqual(coordinator.assignment_for("C").winner_id, "A")

        coordinator.advance_time(2.31)
        self.assertEqual(coordinator.assignment_for("C").winner_id, "B")
        self.assertEqual(coordinator.assignment_switch_count, 1)
        coordinator.ingest_gossip(self._gossip("C", 2.4, {}), 2.4)
        coordinator.advance_time(2.6)
        self.assertEqual(coordinator.liveness_of("C"), NodeLiveness.HEALTHY)
        self.assertIsNone(coordinator.assignment_for("C"))
        self.assertNotIn("C", coordinator.known_bids)

    def test_missing_winner_reassignment_counts_switch(self) -> None:
        """当前赢家自身确认失联时，备用候选接管同样属于一次实际切换。"""
        coordinator = ReplacementCoordinator("B", ("A", "B", "C"), 6.0, ReplacementConfig(bid_window=0.0))
        coordinator.ingest_gossip(self._gossip("A", 1.0, {}), 1.0)
        coordinator.advance_time(1.1)
        coordinator.advance_time(1.8)
        coordinator.ingest_gossip(self._gossip("A", 1.8, {"C": self._bid("A", 0.1, 1.8, 3.8)}), 1.8)
        coordinator.ingest_gossip(self._gossip("B", 1.8, {"C": self._bid("B", 0.2, 1.8, 3.8)}), 1.8)
        self.assertEqual(coordinator.assignment_for("C").winner_id, "A")

        coordinator.advance_time(2.9)
        coordinator.advance_time(3.5)
        self.assertEqual(coordinator.liveness_of("A"), NodeLiveness.MISSING)
        self.assertEqual(coordinator.assignment_for("C").winner_id, "B")
        self.assertEqual(coordinator.assignment_switch_count, 1)
    def test_timeout_requires_confirmation_before_missing(self) -> None:
        """心跳超时必须先进入怀疑，确认期结束后才判定缺失。"""
        coordinator = ReplacementCoordinator(
            "A", ("A", "B"), 10.0, ReplacementConfig(failure_confirmation=0.5)
        )
        coordinator.ingest_gossip(
            {
                "kind": "replacement_gossip",
                "sender": "B",
                "sent_at": 0.0,
                "heartbeats": {"B": 0.0},
                "bids": {},
            },
            received_at=0.0,
        )

        coordinator.advance_time(1.25)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.SUSPECTED)
        coordinator.advance_time(1.75)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.MISSING)

    def test_fresh_heartbeat_immediately_clears_suspicion(self) -> None:
        """新鲜合法心跳必须在接收时立即撤销怀疑，不依赖时钟再次推进。"""
        coordinator = ReplacementCoordinator("A", ("A", "B"), 10.0, ReplacementConfig())
        coordinator.advance_time(1.25)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.SUSPECTED)

        accepted = coordinator.ingest_gossip(
            {
                "kind": "replacement_gossip",
                "sender": "B",
                "sent_at": 1.3,
                "heartbeats": {"B": 1.3},
                "bids": {},
            },
            received_at=1.3,
        )

        self.assertTrue(accepted)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.HEALTHY)
        self.assertNotIn("B", coordinator._suspected_since)

    def test_confirmation_and_recovery_change_only_at_exact_threshold(self) -> None:
        """确认期和恢复稳定期的前一瞬不得转换，精确阈值才允许转换。"""
        config = ReplacementConfig(failure_confirmation=0.5, recovery_stability=0.5)
        coordinator = ReplacementCoordinator("A", ("A", "B"), 10.0, config)
        coordinator.advance_time(1.25)
        coordinator.advance_time(nextafter(1.75, 0.0))
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.SUSPECTED)
        coordinator.advance_time(1.75)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.MISSING)

        coordinator.ingest_gossip(
            {
                "kind": "replacement_gossip",
                "sender": "B",
                "sent_at": 2.0,
                "heartbeats": {"B": 2.0},
                "bids": {},
            },
            received_at=2.0,
        )
        coordinator.advance_time(nextafter(2.5, 0.0))
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.RECOVERING)
        coordinator.advance_time(2.5)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.HEALTHY)

    def test_sub_nanosecond_confirmation_does_not_convert_early(self) -> None:
        """小于一纳秒的确认期仍须严格比较，不能被固定容差提前跨越。"""
        config = ReplacementConfig(failure_timeout=0.5, failure_confirmation=5e-10)
        coordinator = ReplacementCoordinator("A", ("A", "B"), 10.0, config)
        coordinator.advance_time(1.0)
        boundary = 1.0 + config.failure_confirmation

        coordinator.advance_time(nextafter(boundary, 0.0))
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.SUSPECTED)
        coordinator.advance_time(boundary)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.MISSING)

    def test_short_heartbeat_gap_does_not_create_missing_task(self) -> None:
        """怀疑期内收到新心跳时，短暂丢包不应产生缺失任务。"""
        coordinator = ReplacementCoordinator("A", ("A", "B"), 10.0, ReplacementConfig())
        coordinator.advance_time(1.1)
        coordinator.ingest_gossip(
            {
                "kind": "replacement_gossip",
                "sender": "B",
                "sent_at": 1.2,
                "heartbeats": {"B": 1.2},
                "bids": {},
            },
            received_at=1.2,
        )
        coordinator.advance_time(1.3)

        self.assertEqual(coordinator.missing_nodes, ())
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.HEALTHY)

    def test_recovered_node_must_remain_stable_before_handover(self) -> None:
        """缺失节点恢复后必须通过稳定期，才可撤销补位状态。"""
        coordinator = ReplacementCoordinator(
            "A", ("A", "B"), 10.0, ReplacementConfig(failure_confirmation=0.5)
        )
        coordinator.advance_time(1.25)
        coordinator.advance_time(1.75)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.MISSING)

        coordinator.ingest_gossip(
            {
                "kind": "replacement_gossip",
                "sender": "B",
                "sent_at": 2.0,
                "heartbeats": {"B": 2.0},
                "bids": {},
            },
            received_at=2.0,
        )
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.RECOVERING)
        coordinator.ingest_gossip(
            {
                "kind": "replacement_gossip",
                "sender": "B",
                "sent_at": 2.8,
                "heartbeats": {"B": 2.8},
                "bids": {},
            },
            received_at=2.8,
        )
        coordinator.advance_time(3.0)
        self.assertEqual(coordinator.liveness_of("B"), NodeLiveness.HEALTHY)

    def test_gossip_validation_rejects_invalid_messages_without_mutation(self) -> None:
        """未知发送者、未来或非有限时间和错误类型均不得改变协调状态。"""
        coordinator = ReplacementCoordinator("A", ("A", "B"), 10.0, ReplacementConfig())
        baseline_missing = coordinator.missing_nodes
        baseline_heartbeats = dict(coordinator._last_heartbeats)
        invalid_messages = (
            {
                "kind": "replacement_gossip",
                "sender": "C",
                "sent_at": 0.0,
                "heartbeats": {"B": 0.0},
                "bids": {},
            },
            {
                "kind": "replacement_gossip",
                "sender": "B",
                "sent_at": 1.0,
                "heartbeats": {"B": 0.0},
                "bids": {},
            },
            {
                "kind": "replacement_gossip",
                "sender": "B",
                "sent_at": 0.0,
                "heartbeats": {"B": float("nan")},
                "bids": {},
            },
            {
                "kind": "wrong_kind",
                "sender": "B",
                "sent_at": 0.0,
                "heartbeats": {"B": 0.0},
                "bids": {},
            },
        )

        for message in invalid_messages:
            with self.subTest(message=message):
                self.assertFalse(coordinator.ingest_gossip(message, received_at=0.0))
                self.assertEqual(coordinator.missing_nodes, baseline_missing)
                self.assertEqual(coordinator._last_heartbeats, baseline_heartbeats)

        valid_message = {
            "kind": "replacement_gossip",
            "sender": "B",
            "sent_at": 0.5,
            "heartbeats": {"B": 0.5},
            "bids": {},
        }
        self.assertTrue(coordinator.ingest_gossip(valid_message, received_at=0.5))
        self.assertEqual(coordinator._last_heartbeats["B"], 0.5)

    def test_fixed_lanes_follow_stable_roster_order(self) -> None:
        lanes = build_coverage_lanes(("C", "A", "B"), 12.0)
        self.assertEqual([lane.owner_id for lane in lanes], ["A", "B", "C"])
        self.assertEqual([lane.center_y for lane in lanes], [2.0, 6.0, 10.0])

    def test_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            ReplacementConfig(
                distance_weight=0.5,
                load_weight=0.2,
                connectivity_weight=0.2,
                energy_weight=0.2,
            )

    def test_oversized_config_number_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ReplacementConfig(failure_timeout=10**400)

    def test_oversized_world_height_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            build_coverage_lanes(("A",), 10**400)

    def test_lanes_normalize_non_string_owner_identifiers(self) -> None:
        lanes = build_coverage_lanes((2, 10), 4.0)
        self.assertEqual([lane.owner_id for lane in lanes], ["10", "2"])
        self.assertEqual([lane.lane_id for lane in lanes], ["lane-0", "lane-1"])
        self.assertTrue(all(isinstance(lane.owner_id, str) for lane in lanes))
        self.assertTrue(all(isinstance(lane.lane_id, str) for lane in lanes))

    def test_config_rejects_non_finite_number(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ReplacementConfig(failure_timeout=value)

    def test_config_rejects_boolean_dwell_steps(self) -> None:
        with self.assertRaises(ValueError):
            ReplacementConfig(dwell_steps=True)

    def test_lanes_reject_illegal_world_height(self) -> None:
        for value in (0.0, -1.0, True, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_coverage_lanes(("A",), value)
