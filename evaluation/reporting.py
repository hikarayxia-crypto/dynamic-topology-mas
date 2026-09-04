"""评估明细、汇总表和机器可读结果保存。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from .evaluator import ScenarioSummary
from .metrics import EpisodeMetrics


def save_evaluation_results(
    output_dir: str | Path,
    episodes: Sequence[EpisodeMetrics],
    summaries: Sequence[ScenarioSummary],
) -> dict[str, Path]:
    """同时保存逐回合 JSONL 与场景汇总 CSV/JSON。

    保留逐回合数据便于检查均值来源，避免只有汇总图却无法追溯原始结果。
    """

    if not episodes or not summaries:
        raise ValueError("episodes 和 summaries 不能为空")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    episode_path = directory / "episode_metrics.jsonl"
    summary_csv_path = directory / "scenario_summary.csv"
    summary_json_path = directory / "scenario_summary.json"

    with episode_path.open("w", encoding="utf-8") as file:
        for episode in episodes:
            file.write(json.dumps(episode.as_dict(), ensure_ascii=False) + "\n")

    rows = [summary.as_dict() for summary in summaries]
    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "episodes": episode_path,
        "summary_csv": summary_csv_path,
        "summary_json": summary_json_path,
    }
