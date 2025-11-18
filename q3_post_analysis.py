"""后处理脚本：读取 Q3 灵敏度/鲁棒性/预算 CSV，绘制曲线并输出摘要表。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT_CANDIDATES: Sequence[str] = (
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)

Q3_SENS_SINGLE_CSV = Path("q3_sensitivity_single_solution.csv")
Q3_ROBUST_SUMMARY_CSV = Path("q3_robustness_summary.csv")
Q3_BUDGET_CSV = Path("q3_sensitivity_budget.csv")
Q3_QUEUE_FILES: Dict[str, Path] = {
    "baseline": Path("q3_queue_baseline.csv"),
    "best_plan": Path("q3_queue_best_plan.csv"),
}
BASE_PARAMS_PATH = Path("q2_params_best.csv")

CURVE_LAMBDA_TOTAL = Path("q3_curve_total_vs_lambda_factor.png")
CURVE_C_TOTAL = Path("q3_curve_total_vs_c.png")
CURVE_SL10_LAMBDA = Path("q3_curve_SL10_vs_lambda_factor.png")
CURVE_TOTAL_BUDGET = Path("q3_curve_total_vs_budget.png")
CURVE_SL10_BUDGET = Path("q3_curve_SL10_vs_budget.png")
CURVE_QUEUE = Path("q3_queue_vs_time_compare.png")

TABLE_SENS_LAMBDA = Path("q3_table_sensitivity_lambda.csv")
TABLE_SENS_C = Path("q3_table_sensitivity_c.csv")
TABLE_BUDGET = Path("q3_table_budget_vs_performance.csv")


class DataMissingError(FileNotFoundError):
    """自定义异常，使调用者能区分缺失输入。"""


def configure_fonts() -> None:
    """尝试设置一个支持中文的字体，若失败则使用默认字体。"""

    from matplotlib import font_manager

    for name in FONT_CANDIDATES:
        try:
            font_manager.FontProperties(family=name)
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:  # pragma: no cover - 字体设置失败时退回默认值
            continue


def read_csv_with_required_columns(path: Path, required: Iterable[str]) -> pd.DataFrame:
    """读取 CSV 并校验列，供 Q3 各子问题共用。"""
    if not path.exists():
        raise DataMissingError(f"未找到输入文件：{path}")
    df = pd.read_csv(path)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"文件 {path} 缺少必要列：{', '.join(missing)}")
    return df


def infer_base_c(path: Path = BASE_PARAMS_PATH) -> float:
    """从 Q2 最优参数表估计默认窗口数，用于敏感性横轴。"""
    if not path.exists():
        return 3.0
    df = pd.read_csv(path)
    if "c" in df.columns:
        series = pd.to_numeric(df["c"], errors="coerce").dropna()
        if not series.empty:
            return float(series.median())
    return 3.0


def plot_lambda_total_curve(sens_df: pd.DataFrame) -> None:
    """绘制 λ 因子 vs 平均总耗时曲线。"""
    subset = (
        sens_df[(sens_df["param_type"] == "lambda") & sens_df["factor"].notna()]
        .copy()
        .sort_values("factor")
    )
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(subset["factor"], subset["avg_total_time_mean"], marker="o")
    ax.set_xlabel("到达率放大系数")
    ax.set_ylabel("平均总耗时（分钟）")
    ax.set_title("λ 因子对平均总耗时的影响")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_LAMBDA_TOTAL, dpi=150)
    plt.close(fig)


def plot_c_total_curve(sens_df: pd.DataFrame, base_c: float) -> None:
    """绘制窗口数调整对平均总耗时的影响。"""
    subset = sens_df[sens_df["param_type"] == "c"].copy()
    if subset.empty:
        return
    subset = subset.sort_values("delta")
    subset["c"] = (base_c + subset["delta"]).clip(lower=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(subset["c"], subset["avg_total_time_mean"], marker="o")
    ax.set_xlabel("服务器/窗口数")
    ax.set_ylabel("平均总耗时（分钟）")
    ax.set_title("窗口数变化对平均总耗时的影响")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_C_TOTAL, dpi=150)
    plt.close(fig)


def plot_robustness_sl10_curve(robust_df: pd.DataFrame) -> None:
    """绘制 baseline vs best plan 的鲁棒性 SL10 曲线。"""
    subset = robust_df[robust_df["scenario_name"].isin(["baseline", "best_plan"])]
    if subset.empty:
        return
    pivot = subset.pivot_table(
        index="lambda_factor", columns="scenario_name", values="SL10_mean"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for scenario in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[scenario],
            marker="o",
            label=scenario,
        )
    ax.set_xlabel("到达率放大系数")
    ax.set_ylabel("SL10 服务水平")
    ax.set_title("鲁棒性场景的 SL10 曲线")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CURVE_SL10_LAMBDA, dpi=150)
    plt.close(fig)


def plot_budget_curves(budget_df: pd.DataFrame) -> None:
    """预算敏感性：输出平均总耗时与 SL10 两条曲线。"""
    sorted_df = budget_df.sort_values("B")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted_df["B"], sorted_df["avg_total_time_mean"], marker="o")
    ax.set_xlabel("预算（元）")
    ax.set_ylabel("平均总耗时（分钟）")
    ax.set_title("预算 vs 平均总耗时")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_TOTAL_BUDGET, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted_df["B"], sorted_df["SL10_mean"], marker="o", color="#E24A33")
    ax.set_xlabel("预算（元）")
    ax.set_ylabel("SL10 服务水平")
    ax.set_title("预算 vs SL10")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_SL10_BUDGET, dpi=150)
    plt.close(fig)


def plot_queue_compare(queue_data: Dict[str, pd.DataFrame]) -> None:
    """对比不同方案的队列长度序列。"""
    if not queue_data:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, df in queue_data.items():
        if df.empty or "time" not in df.columns:
            continue
        ax.plot(df["time"], df["queue_length"], label=label)
    ax.set_xlabel("时间（分钟）")
    ax.set_ylabel("队列长度（人）")
    ax.set_title("不同方案的队列长度对比")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CURVE_QUEUE, dpi=150)
    plt.close(fig)


def prepare_queue_data() -> Dict[str, pd.DataFrame]:
    """按约定读取 baseline / best_plan 的排队监控 CSV。"""
    queue_data: Dict[str, pd.DataFrame] = {}
    for label, path in Q3_QUEUE_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        queue_data[label] = df
    return queue_data


def describe_sensitivity(lambda_df: pd.DataFrame, c_df: pd.DataFrame) -> None:
    """打印 λ 因子与窗口数调整的口径化描述。"""
    if not lambda_df.empty:
        base = lambda_df[np.isclose(lambda_df["factor"], 1.0)]
        target = lambda_df[np.isclose(lambda_df["factor"], 1.2)]
        if not base.empty and not target.empty:
            delta_t = target.iloc[0]["avg_total_time_mean"] - base.iloc[0]["avg_total_time_mean"]
            delta_sl = target.iloc[0]["SL10_mean"] - base.iloc[0]["SL10_mean"]
            print(
                f"当 λ 提高 20% 时，平均总耗时增加约 {delta_t:.2f} 分钟，SL10 变化 {delta_sl:+.1%}。"
            )
    if not c_df.empty:
        c_sorted = c_df.sort_values("c")
        if len(c_sorted) >= 2:
            first = c_sorted.iloc[0]
            last = c_sorted.iloc[-1]
            delta_t = last["avg_total_time_mean"] - first["avg_total_time_mean"]
            print(
                f"当窗口数从 {first['c']:.0f} 调整到 {last['c']:.0f} 时，平均总耗时变化 {delta_t:+.2f} 分钟。"
            )


def describe_budget(budget_df: pd.DataFrame) -> None:
    """给预算敏感性输出一个简明的口径。"""
    if budget_df.empty:
        return
    sorted_df = budget_df.sort_values("B")
    first = sorted_df.iloc[0]
    last = sorted_df.iloc[-1]
    delta_t = last["avg_total_time_mean"] - first["avg_total_time_mean"]
    delta_sl = last["SL10_mean"] - first["SL10_mean"]
    print(
        f"当预算从 {first['B']:.0f} 元提升到 {last['B']:.0f} 元时，平均总耗时变化 {delta_t:+.2f} 分钟，SL10 变化 {delta_sl:+.1%}。"
    )


def print_sample_rows(title: str, df: pd.DataFrame, n: int = 5) -> None:
    """辅助打印 CSV 的前几行，方便对照论文附录。"""
    if df.empty:
        return
    print(f"\n{title}（前 {n} 行）:")
    print(df.head(n).to_string(index=False))


def main() -> None:
    """Q3 后处理：绘制灵敏度曲线并输出部分摘要表。"""
    configure_fonts()
    sens_df = read_csv_with_required_columns(
        Q3_SENS_SINGLE_CSV,
        ["param_type", "factor", "delta", "avg_total_time_mean", "SL10_mean"],
    )
    robust_df = read_csv_with_required_columns(
        Q3_ROBUST_SUMMARY_CSV,
        ["scenario_name", "lambda_factor", "avg_total_time_mean", "SL10_mean"],
    )
    budget_df = read_csv_with_required_columns(
        Q3_BUDGET_CSV,
        ["B", "avg_total_time_mean", "SL10_mean"],
    )
    queue_data = prepare_queue_data()

    base_c = infer_base_c()

    plot_lambda_total_curve(sens_df)
    plot_c_total_curve(sens_df, base_c)
    plot_robustness_sl10_curve(robust_df)
    plot_budget_curves(budget_df)
    plot_queue_compare(queue_data)

    lambda_subset = sens_df[
        (sens_df["param_type"] == "lambda") & sens_df["factor"].isin([0.8, 1.0, 1.2])
    ][["factor", "avg_total_time_mean", "SL10_mean"]]
    lambda_subset.to_csv(TABLE_SENS_LAMBDA, index=False)

    c_subset = sens_df[sens_df["param_type"] == "c"].copy()
    if not c_subset.empty:
        c_subset["c"] = (base_c + c_subset["delta"]).clip(lower=1)
        c_subset = c_subset[["c", "avg_total_time_mean", "SL10_mean"]]
    c_subset.to_csv(TABLE_SENS_C, index=False)

    budget_subset = budget_df[["B", "avg_total_time_mean", "SL10_mean"]].copy()
    budget_subset.to_csv(TABLE_BUDGET, index=False)

    print_sample_rows("λ 敏感性示例", lambda_subset)
    print_sample_rows("窗口数敏感性示例", c_subset)
    print_sample_rows("预算 vs 性能示例", budget_subset)

    describe_sensitivity(lambda_subset, c_subset)
    describe_budget(budget_subset)


if __name__ == "__main__":
    main()
