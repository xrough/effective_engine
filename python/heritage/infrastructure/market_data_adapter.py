# ============================================================
# 文件：market_data_adapter.py
# 职责：行情数据适配器 — 将 CSV 文件（或内置数据）翻译为 MarketDataEvent，
#       驱动整个仿真事件循环。
#
# 模式：适配器模式（Adapter Pattern）
#   领域层不感知数据来源（CSV / WebSocket / mock），
#   MarketDataAdapter 负责将外部格式转换为领域事件。
# ============================================================

from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path

from events.event_bus import EventBus
from events.events import MarketDataEvent


class MarketDataAdapter:
    """行情数据适配器（CSV 文件 → MarketDataEvent）。"""

    def __init__(self, bus: EventBus, csv_path: str = "") -> None:
        self._bus      = bus       # 事件总线（发布 MarketDataEvent）
        self._csv_path = csv_path  # CSV 文件路径（为空时使用内置数据）

    def run(self) -> None:
        """读取行情数据并逐条发布 MarketDataEvent。

        优先从 CSV 加载；文件不存在时自动使用内置硬编码数据。
        """
        ticks = self._load_from_csv(self._csv_path) if self._csv_path else []
        if not ticks:
            ticks = self._hardcoded_ticks()

        print(f"[MarketDataAdapter] 开始行情推送，共 {len(ticks)} 条 tick\n")

        for ts_str, price in ticks:
            event = MarketDataEvent(
                timestamp        = self._parse_timestamp(ts_str),
                underlying_price = price,
            )
            print("─" * 41)
            print(f"[行情] {ts_str}  AAPL = ${price:.4f}")
            self._bus.publish(event)

        print("─" * 41)
        print("[MarketDataAdapter] 行情推送完毕")

    # ── 私有辅助方法 ─────────────────────────────────────────

    def _load_from_csv(self, path: str) -> list[tuple[str, float]]:
        """从 CSV 文件读取 (timestamp, price) 列表。
        CSV 格式：第一行为标题，后续行为 timestamp,underlying_price
        """
        ticks: list[tuple[str, float]] = []
        p = Path(path)
        if not p.exists():
            print(f"[MarketDataAdapter] 警告：无法打开文件 {path}，将使用内置行情数据")
            return ticks
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过标题行
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    ticks.append((row[0].strip(), float(row[1].strip())))
        return ticks

    @staticmethod
    def _hardcoded_ticks() -> list[tuple[str, float]]:
        """内置 20 条 AAPL 模拟 tick（与 C++ 版本完全一致）。"""
        return [
            ("2024-01-01T09:30:00", 150.00),
            ("2024-01-01T09:30:01", 150.25),
            ("2024-01-01T09:30:02", 150.10),
            ("2024-01-01T09:30:03", 149.80),
            ("2024-01-01T09:30:04", 149.50),
            ("2024-01-01T09:30:05", 149.90),
            ("2024-01-01T09:30:06", 150.30),
            ("2024-01-01T09:30:07", 150.75),
            ("2024-01-01T09:30:08", 151.00),
            ("2024-01-01T09:30:09", 151.20),
            ("2024-01-01T09:30:10", 151.00),
            ("2024-01-01T09:30:11", 150.80),
            ("2024-01-01T09:30:12", 150.50),
            ("2024-01-01T09:30:13", 150.20),
            ("2024-01-01T09:30:14", 149.90),
            ("2024-01-01T09:30:15", 149.70),
            ("2024-01-01T09:30:16", 150.00),
            ("2024-01-01T09:30:17", 150.40),
            ("2024-01-01T09:30:18", 150.60),
            ("2024-01-01T09:30:19", 150.90),
        ]

    @staticmethod
    def _parse_timestamp(ts_str: str) -> datetime:
        """将 ISO-8601 字符串解析为 datetime（或使用当前时间作为回退）。"""
        try:
            return datetime.fromisoformat(ts_str)
        except ValueError:
            return datetime.now()
