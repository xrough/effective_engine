# ============================================================
# 文件：event_bus.py
# 职责：事件总线（EventBus）— 发布/订阅基础设施。
#
# 模式：观察者模式（Observer Pattern）
#   组件调用 subscribe(EventType, handler) 注册为观察者；
#   调用 publish(event) 时，EventBus 同步通知所有订阅者。
#
# Python 实现说明：
#   以 type 对象（而非 std::type_index）作为字典键，
#   直接用 type(event) 查找对应处理器列表，无需类型擦除。
# ============================================================

from __future__ import annotations
from collections import defaultdict
from typing import Callable, Any


class EventBus:
    """发布/订阅事件总线（线程安全：单线程同步分发）。"""

    def __init__(self) -> None:
        # 处理器字典：事件类型 → 处理函数列表
        self._handlers: dict[type, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable[[Any], None]) -> None:
        """注册事件处理器。

        Args:
            event_type: 要订阅的事件类型（类对象）
            handler:    处理函数，签名 handler(event) -> None
        """
        self._handlers[event_type].append(handler)

    def publish(self, event: Any) -> None:
        """向所有订阅者广播事件（同步分发）。

        Args:
            event: 任意事件实例；以 type(event) 查找订阅者
        """
        for handler in self._handlers.get(type(event), []):
            handler(event)

    def clear(self) -> None:
        """清除所有订阅（用于测试重置）。"""
        self._handlers.clear()
