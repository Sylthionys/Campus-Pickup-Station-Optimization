"""SimPy 离散事件仿真内核：模拟校园快递驿站取件流程。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import simpy
except ImportError as exc:  # pragma: no cover - 运行环境需自行安装 SimPy
    raise ImportError("请先安装 SimPy 库：pip install simpy") from exc


TIME_SLOT_PATTERN = re.compile(
    r"(?P<start_h>\d{1,2}):(?P<start_m>\d{2})\s*-\s*(?P<end_h>\d{1,2}):(?P<end_m>\d{2})"
)

# 允许的列别名，统一映射到标准名称
COLUMN_ALIASES: Dict[str, Iterable[str]] = {
    "time_slot": ("time_slot", "timeslot", "period", "timeperiod"),
    "duration_h": ("duration_h", "duration", "hours", "period_h"),
    "lambda": ("lambda", "arrival_rate", "arrivals", "lam"),
    "c": ("c", "servers", "capacity", "n_servers"),
    "mu": ("mu", "service_rate", "srv_rate"),
    "t_find": ("t_find", "find_time", "search_time", "retrieve_time"),
}

TARGET_COLUMN_NAMES: Dict[str, str] = {
    "time_slot": "time_slot",
    "duration_h": "duration_h",
    "lambda": "lambda",
    "c": "c",
    "mu": "mu",
    "t_find": "T_find",
}


def _canonicalize_column(name: str) -> str:
    """统一大小写、下划线，并去掉 _new 后缀。"""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip()).lower()
    return cleaned[:-4] if cleaned.endswith("_new") else cleaned


def normalize_param_columns(params_df: pd.DataFrame) -> pd.DataFrame:
    """将参数表中的别名列映射到标准列。"""
    rename_map: Dict[str, str] = {}
    for col in params_df.columns:
        canonical = _canonicalize_column(str(col))
        for base_name, aliases in COLUMN_ALIASES.items():
            if canonical in aliases:
                rename_map[col] = TARGET_COLUMN_NAMES[base_name]
                break
    df = params_df.rename(columns=rename_map)
    return df


def _require_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """确保关键列可被转换为数值；遇到非法值直接报错。"""
    invalid_cols: List[str] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isnull().any():
            invalid_cols.append(col)
        df[col] = series
    if invalid_cols:
        raise ValueError(f"以下列存在非数值内容，无法仿真：{', '.join(invalid_cols)}")


def parse_time_slot(slot: str) -> Optional[Tuple[int, int]]:
    """解析形如 08:00-10:00 的时间段，返回起止分钟。"""
    if not isinstance(slot, str):
        return None
    match = TIME_SLOT_PATTERN.search(slot)
    if not match:
        return None
    start = int(match.group("start_h")) * 60 + int(match.group("start_m"))
    end = int(match.group("end_h")) * 60 + int(match.group("end_m"))
    if end <= start:
        end += 24 * 60  # 跨天的情况，简单加一昼夜
    return start, end


def prepare_params_table(params_df: pd.DataFrame) -> pd.DataFrame:
    """读取参数表后，补充 slot_start/slot_end 列及辅助列。"""
    df = normalize_param_columns(params_df).copy()
    required = ["time_slot", "duration_h", "lambda", "c", "mu", "T_find"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"参数表缺少必要列：{', '.join(missing)}")
    _require_numeric_columns(df, ["duration_h", "lambda", "c", "mu", "T_find"])

    # 解析时段的起止分钟
    slot_starts: List[float] = []
    slot_ends: List[float] = []
    rolling_end = 0.0

    for slot_str, duration_h in zip(df["time_slot"], df["duration_h"]):
        parsed = parse_time_slot(str(slot_str))
        duration_min = float(duration_h) * 60.0
        if parsed:
            slot_start, slot_end = parsed
        else:
            slot_start = rolling_end
            slot_end = slot_start + duration_min
        slot_starts.append(slot_start)
        slot_ends.append(slot_end)
        rolling_end = slot_end

    df["slot_start_min"] = slot_starts
    df["slot_end_min"] = slot_ends
    df["duration_min"] = df["slot_end_min"] - df["slot_start_min"]
    df["lambda_per_min"] = df["lambda"] / 60.0

    df = df.sort_values("slot_start_min").reset_index(drop=True)
    return df


@dataclass
class StudentRecord:
    student_id: int
    arrival_time: float
    queue_enter_time: float
    wait_time: float
    service_time: float
    find_time: float
    end_time: float

    @property
    def total_time(self) -> float:
        return self.end_time - self.arrival_time


class ServerSystem:
    """封装 SimPy 资源、到达/服务进程及统计逻辑。"""

    def __init__(
        self,
        env: simpy.Environment,
        params_df: pd.DataFrame,
        rng: np.random.Generator,
        max_time: float,
    ) -> None:
        self.env = env
        self.params_df = prepare_params_table(params_df)
        initial_capacity = max(1, int(round(self.params_df.iloc[0]["c"])))
        self.server = simpy.Resource(env, capacity=initial_capacity)
        self.student_records: List[StudentRecord] = []
        self.queue_monitor: List[Dict[str, float]] = []
        self._slot_records = self.params_df.to_dict("records")
        self._target_capacity = initial_capacity
        self.arrivals_finished = False
        self.rng = rng
        self.max_time = max_time

    def get_current_slot_params(self, t: float) -> Dict[str, float]:
        """根据当前时间返回对应时段的参数。"""
        for slot in self._slot_records:
            if slot["slot_start_min"] <= t < slot["slot_end_min"]:
                return slot
        return self._slot_records[-1]

    def _set_capacity_target(self, slot_params: Dict[str, float]) -> None:
        self._target_capacity = max(1, int(round(slot_params["c"])))
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        adjusted_capacity = max(self._target_capacity, self.server.count)
        if adjusted_capacity != self.server.capacity:
            self.server.capacity = adjusted_capacity

    def is_idle(self) -> bool:
        return self.server.count == 0 and not self.server.queue

    def student_process(self, student_id: int):
        """单个学生的取件流程。"""
        arrival_time = self.env.now
        slot_params = self.get_current_slot_params(arrival_time)
        find_time = float(slot_params["T_find"])
        yield self.env.timeout(find_time)  # 确定性找件时间
        queue_enter = self.env.now
        self._set_capacity_target(slot_params)
        with self.server.request() as req:
            yield req
            wait_time = self.env.now - queue_enter
            slot_params = self.get_current_slot_params(self.env.now)
            mu_i = float(slot_params["mu"])
            if mu_i <= 0:
                raise ValueError("mu 必须为正数，表示每台服务器的服务率")
            service_mean = 60.0 / mu_i
            service_time = float(self.rng.exponential(service_mean))
            yield self.env.timeout(service_time)
        self._enforce_capacity()
        end_time = self.env.now
        self.student_records.append(
            StudentRecord(
                student_id=student_id,
                arrival_time=arrival_time,
                queue_enter_time=queue_enter,
                wait_time=wait_time,
                service_time=service_time,
                find_time=find_time,
                end_time=end_time,
            )
        )

    def arrival_process(self):
        """分时段生成到达事件。"""
        student_id = 1
        horizon = float(self.max_time)
        for slot in self._slot_records:
            slot_start = min(slot["slot_start_min"], horizon)
            slot_end = min(slot["slot_end_min"], horizon)
            if slot_start >= horizon:
                break
            lambda_min = float(slot["lambda_per_min"])

            # 若当前时间早于该时段，直接前进
            if self.env.now < slot_start:
                yield self.env.timeout(slot_start - self.env.now)

            self._set_capacity_target(slot)

            while self.env.now < slot_end:
                if lambda_min <= 0:
                    remaining = slot_end - self.env.now
                    if remaining > 0:
                        yield self.env.timeout(remaining)
                    break
                inter_arrival = float(self.rng.exponential(1.0 / lambda_min))
                if self.env.now + inter_arrival > slot_end:
                    remaining = slot_end - self.env.now
                    if remaining > 0:
                        yield self.env.timeout(remaining)
                    break
                yield self.env.timeout(inter_arrival)
                self.env.process(self.student_process(student_id))
                student_id += 1
        self.arrivals_finished = True

    def monitor_queue(self, interval: float = 1.0):
        """周期性记录队列状态，供后续可视化。"""
        while True:
            record = {
                "time": self.env.now,
                "queue_length": len(self.server.queue),
                "in_service": self.server.count,
            }
            self.queue_monitor.append(record)
            if self.arrivals_finished and self.is_idle():
                return
            yield self.env.timeout(interval)


def run_single_replication(
    params_df: pd.DataFrame,
    random_seed: Optional[int] = None,
    max_time: float = 720.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """运行一次 Q3 仿真：返回学生层统计与队列监控。

    参数:
        params_df: Excel 抽取的分时段参数表。
        random_seed: 控制 NumPy 随机数，便于复现。
        max_time: 仿真结束时间（分钟），默认覆盖 12 小时窗口。
    返回:
        stats_df: 每位学生的到达时间、等待、服务、总耗时。
        queue_monitor_df: 按固定间隔记录的队列长度与在服人数。
    """

    if random_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(random_seed)
    env = simpy.Environment()
    system = ServerSystem(env, params_df, rng=rng, max_time=max_time)
    env.process(system.arrival_process())
    env.process(system.monitor_queue())
    env.run(until=max_time)
    if not system.is_idle():
        env.run()

    stats_df = pd.DataFrame([record.__dict__ for record in system.student_records])
    if not stats_df.empty:
        stats_df["total_time"] = stats_df["end_time"] - stats_df["arrival_time"]
    queue_monitor_df = pd.DataFrame(system.queue_monitor)
    return stats_df, queue_monitor_df


def main() -> None:
    """简单演示：构造 6 个时段的参数并运行一次仿真。"""
    sample_params = pd.DataFrame(
        [
            {
                "time_slot": "08:00-10:00",
                "duration_h": 2,
                "lambda": 35,
                "c": 3,
                "mu": 10,
                "T_find": 3.0,
            },
            {
                "time_slot": "10:00-12:00",
                "duration_h": 2,
                "lambda": 45,
                "c": 4,
                "mu": 12,
                "T_find": 2.8,
            },
            {
                "time_slot": "12:00-14:00",
                "duration_h": 2,
                "lambda": 20,
                "c": 3,
                "mu": 11,
                "T_find": 3.2,
            },
            {
                "time_slot": "14:00-16:00",
                "duration_h": 2,
                "lambda": 30,
                "c": 4,
                "mu": 12,
                "T_find": 3.0,
            },
            {
                "time_slot": "16:00-18:00",
                "duration_h": 2,
                "lambda": 50,
                "c": 5,
                "mu": 13,
                "T_find": 2.5,
            },
            {
                "time_slot": "18:00-20:00",
                "duration_h": 2,
                "lambda": 25,
                "c": 3,
                "mu": 11,
                "T_find": 3.5,
            },
        ]
    )

    stats_df, queue_monitor_df = run_single_replication(
        sample_params, random_seed=42, max_time=720
    )

    if stats_df.empty:
        print("本次仿真暂无到达，无法计算指标。")
        return

    avg_total = stats_df["total_time"].mean()
    avg_wait = stats_df["wait_time"].mean()
    sl10 = (stats_df["total_time"] <= 10).mean()

    print("=== 仿真结果 ===")
    print(f"样本量: {len(stats_df)} 人")
    print(f"平均总耗时: {avg_total:.2f} 分钟")
    print(f"平均等待时间: {avg_wait:.2f} 分钟")
    print(f"SL10 (总耗时<=10分钟比例): {sl10:.1%}")

    print("\n队列监控（前 5 行）:")
    print(queue_monitor_df.head())


if __name__ == "__main__":
    main()

