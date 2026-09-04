"""评估指标、场景执行和结果文件测试。"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from evaluation.evaluator import (
    ExperimentScenario,
    LinkFaultSpec,
    NodeFaultSpec,
    _build_environment,
    compare_with_nominal,
    evaluate_scenario,
    summarize_scenario,
)
from evaluation.metrics import StepTrace, summarize_episode
from evaluation.plotting import (
    plot_evaluation_summary,
    plot_replacement_summary,
    plot_training_metrics,
)
from evaluation.reporting import save_evaluation_results


class EvaluationTests(unittest.TestCase):
    def test_replacement_metrics_use_confirmed_first_events_without_fabrication(self) -> None:
        """补位时间从分布式确认起算，未发生的响应或恢复必须保持 None。"""

        common = dict(
            team_reward=0.0,
            detection_ratio=0.0,
            connectivity_ratio=1.0,
            consistency_error=0.0,
            collision_count=0,
            active_agents=2,
            fault_healthy=False,
        )
        traces = [
            StepTrace(
                timestamp=1.0,
                known_missing=("B",),
                replacement_coverage_restored={"B": False},
                uncovered_lane_ratio=0.5,
                **common,
            ),
            StepTrace(
                timestamp=1.4,
                known_missing=("B",),
                replacement_targets=("B",),
                replacement_coverage_restored={"B": False},
                uncovered_lane_ratio=0.5,
                **common,
            ),
            StepTrace(
                timestamp=2.0,
                known_missing=("B",),
                replacement_targets=("B",),
                replacement_coverage_restored={"B": True},
                uncovered_lane_ratio=0.0,
                replacement_switches=1,
                **common,
            ),
        ]
        metrics = summarize_episode(
            traces,
            scenario="replacement_fault",
            policy="replacement",
            episode=0,
            seed=1,
            agent_count=2,
            success=False,
            fault_start=None,
            fault_end=None,
        )
        self.assertAlmostEqual(metrics.replacement_response_time, 0.4)
        self.assertAlmostEqual(metrics.coverage_recovery_time, 1.0)
        self.assertAlmostEqual(metrics.mean_uncovered_lane_ratio, 1.0 / 3.0)
        self.assertEqual(metrics.replacement_switches, 1)
        self.assertEqual(metrics.replacement_success_rate, 1.0)

        never = summarize_episode(
            (traces[0],),
            scenario="never",
            policy="replacement",
            episode=0,
            seed=2,
            agent_count=2,
            success=False,
            fault_start=None,
            fault_end=None,
        )
        self.assertIsNone(never.replacement_response_time)
        self.assertIsNone(never.coverage_recovery_time)
        self.assertEqual(never.replacement_success_rate, 0.0)

        nominal_trace = StepTrace(timestamp=0.2, **common)
        nominal = summarize_episode(
            (nominal_trace,),
            scenario="nominal",
            policy="rule",
            episode=0,
            seed=3,
            agent_count=2,
            success=False,
            fault_start=None,
            fault_end=None,
        )
        self.assertIsNone(nominal.replacement_success_rate)

        second = replace(
            metrics,
            episode=1,
            replacement_response_time=None,
            coverage_recovery_time=None,
            mean_uncovered_lane_ratio=0.5,
            replacement_switches=3,
            replacement_success_rate=0.0,
        )
        summary = summarize_scenario([metrics, second])
        # 未响应回合保持缺失，不得当成零秒拉低时间均值。
        self.assertAlmostEqual(summary.mean_replacement_response_time, 0.4)
        self.assertAlmostEqual(summary.mean_coverage_recovery_time, 1.0)
        self.assertAlmostEqual(summary.mean_uncovered_lane_ratio, 5.0 / 12.0)
        self.assertEqual(summary.mean_replacement_switches, 2.0)
        self.assertEqual(summary.mean_replacement_success_rate, 0.5)

    def test_replacement_policy_runs_and_specialized_plot_is_created(self) -> None:
        """评估器使用独立协调器，并在真实补位事件后生成专项图。"""

        scenario = ExperimentScenario(
            name="replacement_fault",
            agent_count=3,
            episodes=1,
            max_steps=100,
            n_targets=1,
            # 默认确认窗口为 1.6 秒；保持故障足够久，确保测试真正走到补位覆盖。
            node_faults=(NodeFaultSpec("agent-2", 0.4, 10.0),),
        )
        first_environment = _build_environment(scenario, "replacement", None)
        second_environment = _build_environment(scenario, "replacement", None)
        first_agents = list(first_environment.agents.values())
        self.assertEqual(
            len({id(agent.coordinator) for agent in first_agents}),
            scenario.agent_count,
        )
        self.assertTrue(
            all(agent.coordinator.roster == first_agents[0].coordinator.roster for agent in first_agents)
        )
        self.assertIsNot(
            first_agents[0].coordinator,
            list(second_environment.agents.values())[0].coordinator,
        )
        episodes = evaluate_scenario(scenario, policy="replacement")
        self.assertEqual(episodes[0].policy, "replacement")
        self.assertIsNotNone(episodes[0].replacement_response_time)
        self.assertIsNotNone(episodes[0].coverage_recovery_time)
        self.assertGreater(episodes[0].replacement_success_rate or 0.0, 0.0)
        summary = summarize_scenario(episodes)
        with tempfile.TemporaryDirectory() as directory:
            figure = plot_replacement_summary(
                [summary], Path(directory) / "replacement.png"
            )
            self.assertGreater(figure.stat().st_size, 0)

    def test_episode_metrics_keep_failure_and_recovery_semantics(self) -> None:
        traces = [
            StepTrace(0.2, 1.0, 0.1, 1.0, 0.3, 0, 2, False),
            StepTrace(0.4, 1.0, 0.2, 0.5, 0.4, 0, 2, False),
            StepTrace(0.6, 0.0, 0.2, 0.5, 0.4, 1, 2, False),
            StepTrace(0.8, 0.0, 0.3, 1.0, 0.2, 0, 2, True),
        ]
        metrics = summarize_episode(
            traces,
            scenario="fault",
            policy="rule",
            episode=0,
            seed=1,
            agent_count=2,
            success=False,
            fault_start=0.3,
            fault_end=0.5,
        )
        self.assertIsNone(metrics.convergence_steps)
        self.assertAlmostEqual(metrics.fault_recovery_time, 0.3)
        self.assertGreater(metrics.topology_performance_drop, 0.0)
        self.assertEqual(metrics.collisions, 1)

    def test_small_fault_scenario_runs_and_aggregates(self) -> None:
        scenario = ExperimentScenario(
            name="tiny_fault",
            agent_count=2,
            episodes=1,
            max_steps=3,
            n_targets=1,
            link_faults=(LinkFaultSpec("agent-0", "agent-1", 0.2, 0.2),),
        )
        episodes = evaluate_scenario(scenario, policy="rule")
        summary = summarize_scenario(episodes)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(summary.agent_count, 2)
        self.assertGreaterEqual(summary.success_rate, 0.0)
        self.assertLessEqual(summary.success_rate, 1.0)

    def test_reporting_and_plotting_create_real_files(self) -> None:
        scenario = ExperimentScenario(
            name="tiny", agent_count=1, episodes=1, max_steps=2, n_targets=1
        )
        episodes = evaluate_scenario(scenario, policy="rule")
        summary = summarize_scenario(episodes)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_evaluation_results(directory, episodes, [summary])
            self.assertTrue(all(path.exists() for path in paths.values()))
            figure = plot_evaluation_summary(
                [summary], Path(directory) / "evaluation.png"
            )
            self.assertGreater(figure.stat().st_size, 0)
            training_path = Path(directory) / "training.jsonl"
            training_path.write_text(
                json.dumps(
                    {
                        "update": 1,
                        "mean_team_reward": -0.1,
                        "actor_loss": 0.01,
                        "value_loss": 0.2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            curve = plot_training_metrics(training_path, Path(directory) / "training.png")
            self.assertGreater(curve.stat().st_size, 0)

    def test_invalid_fault_reference_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentScenario(
                name="invalid",
                agent_count=1,
                link_faults=(LinkFaultSpec("agent-0", "agent-2", 1.0, 1.0),),
            )

    def test_fault_summary_compares_with_nominal_reward(self) -> None:
        nominal = ExperimentScenario(
            name="nominal", agent_count=1, episodes=1, max_steps=2, n_targets=1
        )
        # 直接使用不可变汇总副本测试比较公式，避免伪造环境实验输出。
        nominal_summary = summarize_scenario(evaluate_scenario(nominal, policy="rule"))
        worse_summary = nominal_summary.__class__(
            **{
                **nominal_summary.as_dict(),
                "scenario": "fault",
                "mean_cumulative_reward": nominal_summary.mean_cumulative_reward - 1.0,
            }
        )
        compared = compare_with_nominal([nominal_summary, worse_summary])
        self.assertEqual(compared[0].relative_reward_drop_vs_nominal, 0.0)
        self.assertGreater(compared[1].relative_reward_drop_vs_nominal, 0.0)


if __name__ == "__main__":
    unittest.main()
