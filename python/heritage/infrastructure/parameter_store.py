# ============================================================
# 文件：parameter_store.py
# 职责：参数仓库 — 订阅 ParamUpdateEvent，持久化模型参数，
#       并提供带版本查询接口。
#
# 对应 Risk_Calibration.md §2.7（Parameter Feedback）
#
# 参数反馈闭环节点：
#   BacktestCalibrationApp → ParamUpdateEvent → ParameterStore
#                                                     ↓
#                                   实时引擎查询 get_params(model_id)
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

from events.event_bus import EventBus
from events.events import ParamUpdateEvent


@dataclass
class _VersionedParams:
    """带时间戳的参数版本（支持历史回溯）。"""
    params:     dict[str, float]  # 参数键值对
    updated_at: datetime          # 更新时间戳


class ParameterStore:
    """参数仓库（订阅 ParamUpdateEvent，存储版本化参数）。"""

    def __init__(self, bus: EventBus) -> None:
        self._bus     = bus
        # 参数历史：model_id → 版本列表（按时间顺序追加）
        self._history: dict[str, list[_VersionedParams]] = {}

    def subscribe_handlers(self) -> None:
        """注册 ParamUpdateEvent 订阅器。"""
        self._bus.subscribe(ParamUpdateEvent, self._on_param_update)

    def _on_param_update(self, event: ParamUpdateEvent) -> None:
        """接收并存储新版本参数。"""
        if event.model_id not in self._history:
            self._history[event.model_id] = []

        self._history[event.model_id].append(
            _VersionedParams(params=event.params, updated_at=event.updated_at)
        )

        version = len(self._history[event.model_id])
        params_str = "  ".join(f"{k}={v:.4f}" for k, v in event.params.items())
        print(
            f"[参数仓库] 📥 接收参数更新  模型: {event.model_id}"
            f"  版本: v{version}  参数: {params_str}"
        )

    def get_params(self, model_id: str) -> dict[str, float]:
        """获取指定模型的最新参数（若未注册则返回空字典）。"""
        versions = self._history.get(model_id)
        if not versions:
            return {}
        return versions[-1].params  # 返回最新版本

    def print_all(self) -> None:
        """打印所有模型的最新参数（仿真结束时汇报）。"""
        print("\n[参数仓库] ══════ 已存储参数汇总 ══════")
        if not self._history:
            print("[参数仓库] （无已存储参数）")
            return
        for model_id, versions in self._history.items():
            print(f"  模型 [{model_id}]  共 {len(versions)} 个版本")
            latest = versions[-1]
            params_str = "  ".join(
                f"{k} = {v:.6f}" for k, v in latest.params.items()
            )
            print(f"  最新参数: {params_str}")
        print("[参数仓库] ════════════════════════════")
