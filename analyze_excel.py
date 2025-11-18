from __future__ import annotations

import math
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pickup_simulation import normalize_param_columns, run_single_replication

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - seaborn 并非硬性依赖
    sns = None


EXCEL_PATH = Path("附件一.xlsx")
Q1_MMC_METRICS_CSV = Path("q1_mmc_metrics.csv")
Q1_MMC_METRICS_XLSX = Path("q1_mmc_metrics.xlsx")
Q1_PARAMS_FROM_EXCEL_CSV = Path("q1_params_from_excel.csv")
Q1_PARAMS_FROM_EXCEL_XLSX = Path("q1_params_from_excel.xlsx")
Q1_SERVICE_LEVEL_CSV = Path("q1_time_slot_service_level.csv")
Q1_SERVICE_LEVEL_XLSX = Path("q1_time_slot_service_level.xlsx")
MISSING_Q1_DATA_MESSAGE = "需要先完成任务 Q1-Data-Prep-From-Excel 和 Q1-MMC-Metrics-By-Period"

FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Microsoft JhengHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
    "Heiti SC",
    "WenQuanYi Zen Hei",
]

TIME_COLUMN_PATTERNS: List[Sequence[str]] = [
    ("时间段",),
    ("时段",),
    ("时间",),
    ("time",),
    ("slot",),
]

T_FIND_PATTERNS: List[Sequence[str]] = [
    ("平均找件时间",),
    ("找件时间",),
    ("找件耗时",),
    ("t_find",),
    ("找件", "时间"),
]

OCCUPANCY_PATTERNS: List[Sequence[str]] = [
    ("在架件数",),
    ("货架占用率",),
    ("货架利用率",),
    ("占用率",),
    ("货架", "占用"),
    ("occupancy",),
]

CONCURRENT_PATTERNS: List[Sequence[str]] = [
    ("同时取件人数",),
    ("在场人数",),
    ("现场人数",),
    ("同时人数",),
    ("concurrent",),
]

PICK_COUNT_PATTERNS: List[Sequence[str]] = [
    ("取件人数",),
    ("取件人次",),
    ("到店人数",),
    ("人数", "取件"),
    ("pick",),
]

CHAR_TRANSLATION_TABLE = str.maketrans(
    {
        " ": "",
        "\u3000": "",
        "\xa0": "",
        "ʱ": "时",
        "ȡ": "取",
        "ˣ": "次",
        "ƽ": "平",
        "�": "",
        "\ufffd": "",
    }
)

# 针对附件一.xlsx 的乱码表头，提供 Fallback 的列名提示。
MANUAL_COLUMN_HINTS: Dict[str, Dict[str, str]] = {
    "Sheet1": {
        "time_slot": "ʱ���",
        "T_find": "�Ҽ���ʱ������ / �ˣ�",
        "occupancy": "ȡ���������ˣ�",
        "concurrent": "ƽ��ÿ��ȡ����������",
        "N_pick": "ȡ���������ˣ�",
    }
}

# 若需人工录入“同时取件人数”，请填好下方模板并将
# USE_MANUAL_CONCURRENT_TEMPLATE 设为 True。
MANUAL_CONCURRENT_TEMPLATE = pd.DataFrame(
    {
        "time_slot": [
            "08:00-10:00",
            "10:00-12:00",
            "12:00-14:00",
            "14:00-16:00",
            "16:00-18:00",
            "18:00-20:00",
        ],
        "concurrent": [np.nan] * 6,
    }
)
USE_MANUAL_CONCURRENT_TEMPLATE = False  # 设为 True 即可启用手工 concurrent

# --- Q3 多方案仿真配置 ---
SCENARIO_FILES: Dict[str, Path] = {
    # 此处文件名可按需调整，保持键为方案名称
    "baseline": Path("q1_params_from_excel.csv"),
    "planA": Path("q2_example_params_baseline.csv"),
    "planB": Path("q2_example_params_all_on.csv"),
}
Q3_REPLICATIONS_RAW_CSV = Path("q3_replications_raw.csv")
Q3_SCENARIO_SUMMARY_CSV = Path("q3_scenario_summary.csv")
TOTAL_TIME_PLOT = Path("q3_scenario_compare_total_time.png")
SL10_PLOT = Path("q3_scenario_compare_SL10.png")
LAMBDA_FACTORS = [1.0, 1.5, 2.0]
ROBUSTNESS_PEAK_ONLY = False  # True 表示仅在高峰时段放大到达率
ROBUSTNESS_N_REP = 20
Q3_ROBUSTNESS_RAW_CSV = Path("q3_robustness_raw.csv")
Q3_ROBUSTNESS_SUMMARY_CSV = Path("q3_robustness_summary.csv")
ROBUSTNESS_TOTAL_TIME_PLOT = Path("q3_robustness_total_time.png")
ROBUSTNESS_SL10_PLOT = Path("q3_robustness_SL10.png")
Q3_HIGH_DEMAND_RAW_CSV = Path("q3_high_demand_raw.csv")
Q3_HIGH_DEMAND_SUMMARY_CSV = Path("q3_high_demand_summary.csv")
HIGH_DEMAND_TOTAL_TIME_PLOT = Path("q3_high_demand_total_time.png")
HIGH_DEMAND_SL10_PLOT = Path("q3_high_demand_SL10.png")
HIGH_DEMAND_QUEUE_PLOT = Path("q3_high_demand_queue.png")
N_REP = 30
DEFAULT_BASE_SEED = 0
REQUIRED_PARAM_COLUMNS = ["time_slot", "duration_h", "lambda", "c", "mu", "T_find"]
Q3_SENS_BASE_PARAMS = Path("q2_params_best.csv")
Q3_SENS_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
Q3_SENS_C_DELTAS = [-1, 0, 1]
Q3_SENS_N_REP = 10
Q3_SENS_SINGLE_OUTPUT = Path("q3_sensitivity_single_solution.csv")
Q3_SENS_BUDGET_OUTPUT = Path("q3_sensitivity_budget.csv")
Q3_SENS_LAMBDA_PLOT = Path("q3_sens_lambda.png")
Q3_SENS_MU_PLOT = Path("q3_sens_mu.png")
Q3_SENS_TFIND_PLOT = Path("q3_sens_Tfind.png")
Q3_SENS_C_PLOT = Path("q3_sens_c.png")
Q3_SENS_BUDGET_PLOT = Path("q3_sens_budget.png")
B_LIST = [20000, 30000, 40000, 50000]
SL_MIN = 0.8
HIGH_DEMAND_CASES: List[Dict[str, Any]] = [
    {
        "case_name": "lambda_x1.5",
        "lambda_factor": 1.5,
        "mu_factor": 1.0,
        "c_cap": None,
        "note": "常规高峰：整体到达率提升 50%",
    },
    {
        "case_name": "lambda_x2_mu_0.9",
        "lambda_factor": 2.0,
        "mu_factor": 0.9,
        "c_cap": None,
        "note": "极端情况：到达率翻倍且服务效率下降 10%",
    },
    {
        "case_name": "lambda_x2_mu_0.85_c3",
        "lambda_factor": 2.0,
        "mu_factor": 0.85,
        "c_cap": 3,
        "note": "极端且窗口受限：仅允许开放 3 个窗口",
    },
]
HIGH_DEMAND_N_REP = 20

# --- Q2 方案参数生成器 ---
# 下方 MEASURES 字典集中描述了 x1~x6 的成本与影响系数。
# 这些数值来自业务访谈与工程经验，用于构造“合理但可调”的假设，
# 方便在灵敏度分析或后续校准时统一修改。
# 每个措施除了成本，还可以定义对 lambda、mu、T_find、c 的缩放/增量。
BASE_C = 3  # 基准人工窗口数
PEAK_SLOTS = ["10:00-12:00", "16:00-18:00"]
DEFAULT_T_FIND_MIN = 4.0  # 分钟
DEFAULT_MACHINE_COUNT = 2
DEFAULT_MACHINE_MU_FACTOR = 2.0

MEASURES: Dict[str, Dict[str, object]] = {
    "x1": {
        "name": "货架布局优化",
        "cost": 5000.0,
        "alpha_Tfind": 0.85,
        "note": "假设通过重新规划货架通道可平均降低 15% 的找件时间",
    },
    "x2": {
        "name": "常驻增配人手",
        "cost": 36000.0,
        "delta_staff": 1.0,
        "note": "经验表明常驻增配 1 人即可稳定多开 1 个窗口",
    },
    "x3": {
        "name": "部署自助取件机",
        "cost": 80000.0,
        "n_machine": DEFAULT_MACHINE_COUNT,
        "machine_mu_factor": DEFAULT_MACHINE_MU_FACTOR,
        "note": "来自供应商数据：单台机器服务速率约为人工 2 倍",
    },
    "x4": {
        "name": "智能提示/电子导览",
        "cost": 12000.0,
        "alpha_Tfind": 0.9,
        "note": "根据试点经验，电子导览可再降低 ~10% 找件时间",
    },
    "x5": {
        "name": "预约错峰激励",
        "cost": 15000.0,
        "lambda_peak_factor": 0.9,
        "lambda_off_factor": 1.1,
        "note": "基于客户调研：约 10% 高峰订单愿意错峰并平移到低峰",
    },
    "x6": {
        "name": "高峰临时补员",
        "cost": 8000.0,
        "peak_delta_staff": 1.0,
        "note": "运维记录显示临时补员 1 人可在高峰再开一窗口",
    },
}
B_limit = 50_000  # 预算上限（元），用于场景 B
SL_target = 0.9  # 目标服务水平
baseline_id = 0  # solution_id = 0 通常对应 x1~x6 全 0 的基准方案

TIME_RANGE_PATTERN = re.compile(
    r"(\d{1,2})(?::|：)?(\d{2})?\s*[-~～—至]\s*(\d{1,2})(?::|：)?(\d{2})?"
)


def normalize_label(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    value = str(text).strip()
    value = value.translate(CHAR_TRANSLATION_TABLE)
    value = re.sub(r"\s+", "", value.lower())
    return value


def match_columns(
    df: pd.DataFrame,
    patterns: List[Sequence[str]],
    validator: Optional[Callable[[pd.Series], bool]] = None,
) -> List[str]:
    normalized = {col: normalize_label(col) for col in df.columns}
    matches: List[str] = []
    for pattern in patterns:
        normalized_pattern = [normalize_label(p) for p in pattern if p]
        cols: List[str] = []
        for col, label in normalized.items():
            if all(keyword in label for keyword in normalized_pattern):
                if validator is None or validator(df[col]):
                    cols.append(col)
        if cols:
            matches.extend(cols)
            break
    return matches


def find_column(
    df: pd.DataFrame,
    patterns: List[Sequence[str]],
    validator: Optional[Callable[[pd.Series], bool]] = None,
) -> Optional[str]:
    candidates = match_columns(df, patterns, validator)
    return candidates[0] if candidates else None


def looks_like_numeric(series: pd.Series, min_ratio: float = 0.6) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    ratio = numeric.notna().mean()
    return bool(ratio >= min_ratio)


def has_time_range(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return False
    return bool(TIME_RANGE_PATTERN.search(text))


def extract_time_slot_bounds(value: object) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = TIME_RANGE_PATTERN.search(text)
    if not match:
        return None
    start_hour = int(match.group(1))
    start_min = int(match.group(2)) if match.group(2) else 0
    end_hour = int(match.group(3))
    end_min = int(match.group(4)) if match.group(4) else 0
    start = start_hour + start_min / 60.0
    end = end_hour + end_min / 60.0
    if end <= start:
        end += 24.0
    return start, end


def parse_time_slot_duration(value: object) -> Optional[float]:
    bounds = extract_time_slot_bounds(value)
    if bounds is None:
        return None
    start, end = bounds
    duration = end - start
    if duration <= 0:
        return None
    return duration


def looks_like_time_slot(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.astype(str).head(20)
    mask = sample.apply(has_time_range)
    return bool(mask.mean() >= 0.5)


def make_unique_columns(header_row: pd.Series) -> List[str]:
    names: List[str] = []
    seen: Dict[str, int] = {}
    for idx, value in enumerate(header_row):
        label = str(value).strip()
        if not label or label.lower() == "nan":
            label = f"col_{idx}"
        if label in seen:
            seen[label] += 1
            label = f"{label}_{seen[label]}"
        else:
            seen[label] = 0
        names.append(label)
    return names


def candidate_dataframes(df: pd.DataFrame) -> Iterator[Tuple[pd.DataFrame, str]]:
    yield df.copy(), "原始表头"
    if df.empty:
        return
    max_header_idx = max(0, min(3, len(df) - 1))
    for header_idx in range(max_header_idx):
        header_row = df.iloc[header_idx]
        trimmed = df.iloc[header_idx + 1 :].copy()
        trimmed.columns = make_unique_columns(header_row)
        trimmed.reset_index(drop=True, inplace=True)
        yield trimmed, f"使用第{header_idx + 1}行作为表头"


def choose_best_numeric_column(
    df: pd.DataFrame, columns: Sequence[str], prefer: str = "largest"
) -> Optional[str]:
    best_col: Optional[str] = None
    best_score = -math.inf if prefer == "largest" else math.inf
    for col in columns:
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            continue
        score = float(numeric.mean())
        if prefer == "largest":
            if score > best_score:
                best_score = score
                best_col = col
        else:
            if score < best_score:
                best_score = score
                best_col = col
    return best_col


def detect_time_column(df: pd.DataFrame, hint: Optional[str]) -> Optional[str]:
    if hint and hint in df.columns:
        return hint
    column = find_column(df, TIME_COLUMN_PATTERNS)
    if column:
        return column
    for col in df.columns:
        if looks_like_time_slot(df[col]):
            return col
    return None


def detect_measurement_column(
    df: pd.DataFrame,
    patterns: List[Sequence[str]],
    hint: Optional[str],
    fallback_prefer: Optional[str] = None,
) -> Optional[str]:
    if hint and hint in df.columns:
        return hint
    column = find_column(df, patterns, looks_like_numeric)
    if column:
        return column
    if fallback_prefer:
        numeric_cols = [col for col in df.columns if looks_like_numeric(df[col])]
        return choose_best_numeric_column(df, numeric_cols, prefer=fallback_prefer)
    return None


def detect_regression_source(
    sheet_frames: Dict[str, pd.DataFrame]
) -> Optional[Dict[str, object]]:
    for sheet_name, df in sheet_frames.items():
        hints = MANUAL_COLUMN_HINTS.get(sheet_name, {})
        for candidate_df, note in candidate_dataframes(df):
            time_col = detect_time_column(candidate_df, hints.get("time_slot"))
            t_find_col = detect_measurement_column(
                candidate_df, T_FIND_PATTERNS, hints.get("T_find")
            )
            occupancy_col = detect_measurement_column(
                candidate_df,
                OCCUPANCY_PATTERNS,
                hints.get("occupancy"),
                fallback_prefer="largest",
            )
            concurrent_col = detect_measurement_column(
                candidate_df,
                CONCURRENT_PATTERNS,
                hints.get("concurrent"),
                fallback_prefer="smallest",
            )

            print(
                f"[{sheet_name} | {note}] 检测列："
                f"time={time_col or '未找到'}, "
                f"T_find={t_find_col or '未找到'}, "
                f"occupancy={occupancy_col or '未找到'}, "
                f"concurrent={concurrent_col or '未找到'}"
            )

            if t_find_col and occupancy_col:
                return {
                    "sheet": sheet_name,
                    "note": note,
                    "df": candidate_df.copy(),
                    "time_col": time_col,
                    "t_find_col": t_find_col,
                    "occupancy_col": occupancy_col,
                    "concurrent_col": concurrent_col,
                }
    return None


def detect_q1_time_slot_source(
    sheet_frames: Dict[str, pd.DataFrame]
) -> Optional[Dict[str, object]]:
    candidates: List[Tuple[int, Dict[str, object]]] = []
    for sheet_name, df in sheet_frames.items():
        hints = MANUAL_COLUMN_HINTS.get(sheet_name, {})
        for candidate_df, note in candidate_dataframes(df):
            time_col = detect_time_column(candidate_df, hints.get("time_slot"))
            pick_col = detect_measurement_column(
                candidate_df,
                PICK_COUNT_PATTERNS,
                hints.get("N_pick"),
                fallback_prefer="largest",
            )
            t_find_col = detect_measurement_column(
                candidate_df,
                T_FIND_PATTERNS,
                hints.get("T_find"),
                fallback_prefer="largest",
            )
            if not (time_col and pick_col):
                continue
            priority = 0
            normalized_name = str(sheet_name)
            if "典型" in normalized_name:
                priority += 2
            if "工作日" in normalized_name:
                priority += 1
            candidates.append(
                (
                    priority,
                    {
                        "sheet": sheet_name,
                        "note": note,
                        "df": candidate_df.copy(),
                        "time_col": time_col,
                        "pick_col": pick_col,
                        "t_find_col": t_find_col,
                    },
                )
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def build_regression_dataset(
    df: pd.DataFrame, columns: Dict[str, Optional[str]]
) -> Tuple[pd.DataFrame, int]:
    use_columns = [columns["t_find_col"], columns["occupancy_col"]]
    if columns["concurrent_col"]:
        use_columns.append(columns["concurrent_col"])
    if columns["time_col"]:
        use_columns.append(columns["time_col"])

    subset = df[use_columns].copy()
    rename_map = {columns["t_find_col"]: "T_find", columns["occupancy_col"]: "occupancy"}
    if columns["concurrent_col"]:
        rename_map[columns["concurrent_col"]] = "concurrent"
    if columns["time_col"]:
        rename_map[columns["time_col"]] = "time_slot"
    subset.rename(columns=rename_map, inplace=True)

    if "time_slot" in subset.columns:
        subset["time_slot"] = subset["time_slot"].astype(str).str.strip()
        mask = subset["time_slot"].apply(has_time_range)
        if not mask.all():
            skipped = subset.loc[~mask, "time_slot"].tolist()
            if skipped:
                print("提示：以下记录的时段格式异常，已忽略：", skipped)
        subset = subset[mask].copy()

    for col in ("T_find", "occupancy", "concurrent"):
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")

    before_drop = len(subset)
    numeric_cols = ["T_find", "occupancy"] + (["concurrent"] if "concurrent" in subset.columns else [])
    subset.dropna(subset=numeric_cols, inplace=True)
    dropped = before_drop - len(subset)
    if dropped > 0:
        print(f"提示：由于缺失值共丢弃 {dropped} 条记录。")
    if subset.empty:
        raise ValueError("没有足够的数据点构建回归。")
    return subset, dropped


def build_q1_params_from_excel(
    excel_path: Path = EXCEL_PATH,
    sheet_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    """构建 Q1 的分时段参数表。

    读取附件一.xlsx，定位典型工作日的时间段与取件人数，并估计 λ、μ、c
    等字段。默认写出 ``q1_params_from_excel.(csv|xlsx)``，供 Q2/Q3 直接使用。

    参数
    ----
    excel_path:
        原始 Excel 路径，默认读取 ``EXCEL_PATH``。
    sheet_frames:
        调试用的预加载数据集，传入后将跳过 Excel 解析。
    """
    if sheet_frames is None:
        if not excel_path.exists():
            print("错误：未在当前目录找到 '附件一.xlsx'，无法生成 Q1 参数表。")
            return None
        xls = pd.ExcelFile(excel_path)
        sheet_frames = {sheet: xls.parse(sheet, header=None) for sheet in xls.sheet_names}

    selection = detect_q1_time_slot_source(sheet_frames)
    if selection is None:
        print("错误：未能从 Excel 中定位包含时段与取件人数的典型工作日数据表。")
        return None

    subset = selection["df"][
        [selection["time_col"], selection["pick_col"]]
    ].copy()
    rename_map = {
        selection["time_col"]: "time_slot",
        selection["pick_col"]: "N_pick",
    }
    if selection["t_find_col"]:
        subset[selection["t_find_col"]] = pd.to_numeric(
            selection["df"][selection["t_find_col"]], errors="coerce"
        )
        rename_map[selection["t_find_col"]] = "T_find_mean"
    subset.rename(columns=rename_map, inplace=True)
    subset["time_slot"] = subset["time_slot"].astype(str).str.strip()
    subset = subset[subset["time_slot"].apply(has_time_range)]
    subset["duration_h"] = subset["time_slot"].apply(parse_time_slot_duration)
    subset["N_pick"] = pd.to_numeric(subset["N_pick"], errors="coerce")
    subset.dropna(subset=["duration_h", "N_pick"], inplace=True)
    subset = subset[subset["duration_h"] > 0]
    if subset.empty:
        print("错误：典型工作日数据中没有有效的时段记录。")
        return None

    if "T_find_mean" not in subset.columns:
        subset["T_find_mean"] = DEFAULT_T_FIND_MIN
        print("提示：典型工作日表未提供找件时间，已使用默认值 4 分钟。")

    subset["lambda"] = subset["N_pick"] / subset["duration_h"]
    subset["c"] = BASE_C
    subset["mu_theory"] = 60.0

    total_pick = subset["N_pick"].sum()
    total_hours = subset["duration_h"].sum()
    mu_data = total_pick / (BASE_C * total_hours) if total_hours > 0 else float("nan")
    subset["mu_data"] = mu_data

    subset.sort_values("time_slot", inplace=True)
    subset.reset_index(drop=True, inplace=True)

    subset.to_csv(Q1_PARAMS_FROM_EXCEL_CSV, index=False)
    subset.to_excel(Q1_PARAMS_FROM_EXCEL_XLSX, index=False)

    print("\n=== Q1 Excel 数据预处理 ===")
    print(
        f"来源：{selection['sheet']}（{selection['note']}），共 {len(subset)} 个时段，"
        f"总取件人数 {total_pick:.0f} 人，营业时长 {total_hours:.2f} 小时。"
    )
    print(
        f"估计数据服务率 μ_data = {mu_data:.2f} 人/小时（c={BASE_C} 个窗口）。"
    )
    print(f"已写出参数表：{Q1_PARAMS_FROM_EXCEL_CSV} 与 {Q1_PARAMS_FROM_EXCEL_XLSX}")

    return subset


def build_q1_mmc_metrics(params_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """基于 Q1 参数表批量计算 M/M/c 指标。

    参数表通常来自 :func:`build_q1_params_from_excel` 输出，函数会对每个
    时间段计算 ρ、Pw、Wq 等理论/数据两套指标并写出 ``q1_mmc_metrics``。
    """

    required_cols = ["time_slot", "lambda", "c", "mu_data", "mu_theory"]
    missing = [col for col in required_cols if col not in params_df.columns]
    if missing:
        print(f"错误：Q1 参数表缺少必要列：{', '.join(missing)}")
        return None

    records: List[Dict[str, object]] = []
    print("\n=== Q1 分时段 M/M/c 指标 ===")
    for row in params_df.to_dict("records"):
        time_slot = str(row.get("time_slot", "未知时段"))
        lam = float(row.get("lambda", float("nan")))
        c_raw = row.get("c", BASE_C)
        c_value = int(round(c_raw)) if not pd.isna(c_raw) else BASE_C
        mu_theory = float(row.get("mu_theory", float("nan")))
        mu_data = float(row.get("mu_data", float("nan")))

        metrics_theory = mmc_metrics(lam, mu_theory, c_value)
        metrics_data = mmc_metrics(lam, mu_data, c_value)

        def _format_status(label: str, mu_value: float, metrics: Dict[str, Optional[float]]) -> str:
            rho_text = f"ρ={metrics['rho']:.2f}" if metrics.get("rho") is not None else "ρ=NaN"
            status = "稳定" if metrics.get("stable") else "超负荷"
            message = metrics.get("message") or ""
            if message:
                message = f"，{message}"
            return (
                f"{time_slot} | {label} μ={mu_value:.2f} 人/小时：{rho_text}，状态：{status}{message}"
            )

        print(_format_status("数据标定", mu_data, metrics_data))
        print(_format_status("理论", mu_theory, metrics_theory))

        record = {
            "time_slot": time_slot,
            "lambda": lam,
            "c": c_value,
            "mu_data": mu_data,
            "mu_theory": mu_theory,
            "rho_data": metrics_data.get("rho"),
            "rho_theory": metrics_theory.get("rho"),
            "P0_data": metrics_data.get("P0"),
            "P0_theory": metrics_theory.get("P0"),
            "Pw_data": metrics_data.get("Pw"),
            "Pw_theory": metrics_theory.get("Pw"),
            "Lq_data": metrics_data.get("Lq"),
            "Lq_theory": metrics_theory.get("Lq"),
            "L_data": metrics_data.get("L"),
            "L_theory": metrics_theory.get("L"),
            "Wq_data_min": metrics_data.get("Wq_min"),
            "Wq_theory_min": metrics_theory.get("Wq_min"),
            "W_data_min": metrics_data.get("W_min"),
            "W_theory_min": metrics_theory.get("W_min"),
            "stable_data": metrics_data.get("stable"),
            "stable_theory": metrics_theory.get("stable"),
            "message_data": metrics_data.get("message"),
            "message_theory": metrics_theory.get("message"),
        }
        records.append(record)

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(Q1_MMC_METRICS_CSV, index=False)
    metrics_df.to_excel(Q1_MMC_METRICS_XLSX, index=False)
    print(f"已写出排队指标表：{Q1_MMC_METRICS_CSV} 与 {Q1_MMC_METRICS_XLSX}")
    return metrics_df


def maybe_apply_manual_concurrent(df: pd.DataFrame) -> pd.DataFrame:
    """在缺失 ``concurrent`` 列时合并手工模板。

    Q1/Q2 的 Excel 有时未采集同时取件人数，允许通过 ``MANUAL_CONCURRENT_TEMPLATE``
    事先录入再在此处自动补齐。原 DataFrame 会被复制后返回。
    """

    if "concurrent" in df.columns or not USE_MANUAL_CONCURRENT_TEMPLATE:
        return df
    if "time_slot" not in df.columns:
        print("警告：找不到 time_slot 列，无法合并手工 concurrent。")
        return df

    manual_df = MANUAL_CONCURRENT_TEMPLATE.copy()
    manual_df["time_slot"] = manual_df["time_slot"].astype(str).str.strip()
    merged = df.merge(manual_df, on="time_slot", how="left", suffixes=("", "_manual"))
    merged["concurrent"] = merged["concurrent"].fillna(merged["concurrent_manual"])
    merged.drop(columns=["concurrent_manual"], inplace=True)
    return merged


def load_q1_params_for_q2(csv_path: Path = Q1_PARAMS_FROM_EXCEL_CSV) -> pd.DataFrame:
    """加载 Q1 的基准参数表，供 Q2 穷举使用。"""

    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 {csv_path}，请先生成问题一的基准参数表。")

    df = pd.read_csv(csv_path)
    required_columns = ["time_slot", "duration_h", "lambda", "c", "mu_data"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列：{', '.join(missing)}。")

    df["time_slot"] = df["time_slot"].astype(str).str.strip()
    numeric_cols = ["duration_h", "lambda", "c", "mu_data", "T_find_mean"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_measure_vector(**overrides: int) -> Dict[str, int]:
    vector = {f"x{i}": 0 for i in range(1, 7)}
    for key, value in overrides.items():
        if key not in vector:
            raise KeyError(f"未知措施 {key}")
        vector[key] = int(value)
    return vector


def _extract_measure_flags(x: object) -> Dict[str, int]:
    if isinstance(x, pd.Series):
        data = x.to_dict()
    elif isinstance(x, dict):
        data = x
    else:
        raise TypeError("x 需要为 dict 或 pandas Series。")

    flags: Dict[str, int] = {}
    for i in range(1, 7):
        key = f"x{i}"
        value = int(data.get(key, 0))
        flags[key] = 1 if value else 0
    return flags


def _is_baseline_vector(flags: Dict[str, int]) -> bool:
    return not any(flags.values())


def _format_numeric(value: float) -> str:
    return "NaN" if pd.isna(value) else f"{value:.2f}"


def apply_measures(base_df: pd.DataFrame, x: object) -> Tuple[pd.DataFrame, float]:
    """在 Q2 穷举中根据 x1~x6 组合调整 λ、μ、c、T_find。

    返回复制后的 DataFrame 以及方案总成本；影响系数全部来自 ``MEASURES``，
    方便在灵敏度分析中统一维护。
    """

    flags = _extract_measure_flags(x)
    df = base_df.copy()
    if "time_slot" not in df.columns:
        raise ValueError("base_df 需要包含 time_slot 列。")
    if "lambda" not in df.columns or "mu_data" not in df.columns:
        raise ValueError("base_df 需要包含 lambda 与 mu_data 列。")

    total_cost = 0.0
    for i in range(1, 7):
        key = f"x{i}"
        total_cost += MEASURES[key]["cost"] * flags[key]

    if "T_find_mean" not in df.columns:
        df["T_find_mean"] = DEFAULT_T_FIND_MIN
    df["T_find_mean"] = pd.to_numeric(df["T_find_mean"], errors="coerce").fillna(DEFAULT_T_FIND_MIN)
    tfind_factor = 1.0
    for key, active in flags.items():
        if active:
            alpha = MEASURES.get(key, {}).get("alpha_Tfind")
            if alpha:
                tfind_factor *= alpha
    df["T_find_new"] = df["T_find_mean"] * tfind_factor

    lambda_series = pd.to_numeric(df["lambda"], errors="coerce")
    lambda_new = lambda_series.copy()
    if flags["x5"]:
        peak_factor = MEASURES["x5"].get("lambda_peak_factor", 1.0)
        off_factor = MEASURES["x5"].get("lambda_off_factor", 1.0)
        peak_mask = df["time_slot"].isin(PEAK_SLOTS)
        lambda_new = np.where(
            peak_mask,
            lambda_series * peak_factor,
            lambda_series * off_factor,
        )
    df["lambda_new"] = lambda_new

    if "c" in df.columns:
        c_manual = pd.to_numeric(df["c"], errors="coerce").fillna(BASE_C)
    else:
        c_manual = pd.Series(BASE_C, index=df.index, dtype=float)
    delta_staff = sum(MEASURES[key].get("delta_staff", 0.0) * flags[key] for key in flags)
    c_manual = c_manual + delta_staff
    peak_delta = sum(MEASURES[key].get("peak_delta_staff", 0.0) * flags[key] for key in flags)
    if peak_delta:
        peak_mask = df["time_slot"].isin(PEAK_SLOTS)
        c_manual = c_manual.where(~peak_mask, c_manual + peak_delta)
    df["c_manual"] = c_manual

    mu_manual = pd.to_numeric(df["mu_data"], errors="coerce")
    n_machine = 0
    machine_factor = DEFAULT_MACHINE_MU_FACTOR
    if flags["x3"]:
        n_machine = int(MEASURES["x3"].get("n_machine", DEFAULT_MACHINE_COUNT))
        machine_factor = MEASURES["x3"].get("machine_mu_factor", DEFAULT_MACHINE_MU_FACTOR)

    if n_machine > 0:
        mu_machine = mu_manual * machine_factor
        c_eff = c_manual + n_machine
        total_capacity = c_manual * mu_manual + n_machine * mu_machine
        with np.errstate(invalid="ignore", divide="ignore"):
            mu_eff = np.where(c_eff > 0, total_capacity / c_eff, np.nan)
        df["c_new"] = c_eff
        df["mu_new"] = mu_eff
    else:
        df["c_new"] = c_manual
        df["mu_new"] = mu_manual

    return df, total_cost


def mmc_metrics(lambda_rate: float, mu: float, c: int) -> Dict[str, Optional[float]]:
    """
    经典 M/M/c 排队指标：ρ、Erlang C 等待概率、平均等待/系统时间。
    所有速率以“人/小时”为单位，返回的 Wq/W 以“分钟”为单位。
    """

    result: Dict[str, Optional[float]] = {
        "stable": False,
        "rho": None,
        "P0": None,
        "Pw": None,
        "Lq": None,
        "L": None,
        "Wq_min": None,
        "W_min": None,
        "message": None,
    }

    if any(pd.isna(value) for value in (lambda_rate, mu, c)) or mu <= 0 or c is None:
        result["message"] = "λ、μ 或 c 缺失，无法计算。"
        return result

    c_int = max(1, int(round(c)))
    if c_int <= 0:
        result["message"] = "服务窗口数量需为正整数。"
        return result

    capacity = c_int * mu
    rho = lambda_rate / capacity if capacity > 0 else float("inf")
    result["rho"] = rho
    if rho >= 1:
        result["message"] = "负载率 ρ≥1，系统不稳定。"
        return result

    a = lambda_rate / mu if mu > 0 else float("inf")
    sum_terms = sum((a**k) / math.factorial(k) for k in range(c_int))
    term_c = (a**c_int) / math.factorial(c_int)
    denom = sum_terms + term_c * (1.0 / (1.0 - rho))
    if denom <= 0:
        result["message"] = "Erlang C 分母为 0，无法计算。"
        return result

    p0 = 1.0 / denom
    pw = term_c * p0 * (1.0 / (1.0 - rho))
    pw = min(max(pw, 0.0), 1.0)
    spare_capacity = capacity - lambda_rate
    wq_hours = pw / spare_capacity if spare_capacity > 0 else float("inf")
    service_time_hours = 1.0 / mu
    w_hours = wq_hours + service_time_hours
    lq = wq_hours * lambda_rate if lambda_rate > 0 else 0.0
    l_system = lq + a

    result.update(
        {
            "stable": True,
            "P0": p0,
            "Pw": pw,
            "Lq": lq,
            "L": l_system,
            "Wq_min": wq_hours * 60.0,
            "W_min": w_hours * 60.0,
            "message": None,
        }
    )
    return result


def evaluate_solution(x: Dict[str, int], base_df: pd.DataFrame) -> Dict[str, object]:
    """评估单个 Q2 方案的成本、平均总耗时与 SL10。"""

    flags = _extract_measure_flags(x)
    df, total_cost = apply_measures(base_df, flags)
    df = df.copy()

    numeric_cols = ["duration_h", "lambda_new", "mu_new", "c_new", "T_find_new"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["duration_h"].fillna(0.0, inplace=True)
    df["lambda_new"].fillna(0.0, inplace=True)
    df["c_new"].fillna(BASE_C, inplace=True)
    df["T_find_new"].fillna(DEFAULT_T_FIND_MIN, inplace=True)

    if "mu_new" in df.columns:
        df["mu_new"] = df["mu_new"]
    else:
        df["mu_new"] = float("nan")

    if "N_pick" in df.columns:
        df["N_weight"] = pd.to_numeric(df["N_pick"], errors="coerce")
    else:
        df["N_weight"] = np.nan
    fallback_weight = (df["lambda_new"] * df["duration_h"]).round().clip(lower=0.0)
    df["N_weight"] = df["N_weight"].fillna(fallback_weight)

    penalty_total_minutes = 30.0  # 若系统不稳定，给一个惩罚性耗时避免结果崩溃。
    unstable_slots: List[str] = []
    t_total_list: List[float] = []
    sl10_list: List[float] = []

    for _, row in df.iterrows():
        time_slot = row.get("time_slot", "未知时段")
        lam = float(row["lambda_new"]) if not pd.isna(row["lambda_new"]) else 0.0
        mu = float(row["mu_new"]) if not pd.isna(row["mu_new"]) else float("nan")
        c_value = row["c_new"]
        c_int = max(1, int(round(c_value))) if not pd.isna(c_value) else BASE_C

        metrics = mmc_metrics(lam, mu, c_int)
        service_time = 60.0 / mu if mu and mu > 0 else float("nan")

        if not metrics["stable"]:
            unstable_slots.append(str(time_slot))
            print(f"警告：{time_slot} 时段系统不稳定（ρ≥1 或 μ≤0），将以惩罚值评估。")
            t_total = penalty_total_minutes
            sl10_value = 0.0
        else:
            wq = metrics["Wq_min"] or 0.0
            if not math.isfinite(service_time):
                service_time = 0.0
            # 每人总耗时 T_total_i = T_find_new + Wq_i + T_service_i
            t_total = row["T_find_new"] + wq + service_time

            t0 = 10.0 - row["T_find_new"] - service_time
            if t0 <= 0:
                sl10_value = 0.0
            else:
                rate_wait = (c_int * mu - lam) / 60.0
                if rate_wait <= 0 or metrics["Pw"] is None:
                    sl10_value = 0.0
                else:
                    # 混合指数近似：SL10_i = (1-Pw) + Pw * (1-exp(-rate_wait * t0_i))
                    decay = math.exp(-rate_wait * t0)
                    sl10_value = ((1.0 - metrics["Pw"]) + metrics["Pw"] * (1.0 - decay))
            sl10_value = float(np.clip(sl10_value, 0.0, 1.0))

        t_total_list.append(t_total)
        sl10_list.append(sl10_value)

    df["T_total_i"] = t_total_list
    df["SL10_i_eval"] = sl10_list

    weights = df["N_weight"]
    t_total_day = weighted_average(df["T_total_i"], weights)
    sl10_day = weighted_average(df["SL10_i_eval"], weights)
    f3 = 1.0 - sl10_day if pd.notna(sl10_day) else float("nan")

    return {
        "x": flags,
        "is_baseline": _is_baseline_vector(flags),
        "f1_cost": float(total_cost),
        "f2_Ttotal_mean": float(t_total_day) if pd.notna(t_total_day) else float("nan"),
        "SL10_day": float(sl10_day) if pd.notna(sl10_day) else float("nan"),
        "f3_one_minus_SL10": float(f3) if pd.notna(f3) else float("nan"),
        "has_unstable_slot": bool(unstable_slots),
        "unstable_slots": unstable_slots,
    }


def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """
    朴素 O(n^2) 帕累托判定，默认为“越小越好”的最小化目标。
    points: (n_solutions, n_objectives)
    返回每个点是否为帕累托有效点的布尔数组。
    """

    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    clean = np.where(np.isfinite(array), array, np.inf)
    n = clean.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        current = clean[i]
        # 当前点只能“淘汰”那些在所有目标上都不优于 current、且至少一项更差的点。
        dominated = np.all(current <= clean, axis=1) & np.any(current < clean, axis=1)
        efficient[dominated] = False
        efficient[i] = True
    return efficient


def analyze_q2_pareto_fronts(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    穷举 2^6 个方案，计算多目标指标与帕累托前沿，并将结果写入 CSV/XLSX。
    """

    print("\n=== Q2 全组合帕累托分析 ===")
    records: List[Dict[str, object]] = []
    for bits in product([0, 1], repeat=6):
        vector = {f"x{i+1}": bits[i] for i in range(6)}
        result = evaluate_solution(vector, base_df)
        record: Dict[str, object] = {f"x{i+1}": result["x"].get(f"x{i+1}", bits[i]) for i in range(6)}
        record.update(
            {
                "f1_cost": result.get("f1_cost"),
                "f2_Ttotal_mean": result.get("f2_Ttotal_mean"),
                "SL10_day": result.get("SL10_day"),
                "f3_one_minus_SL10": result.get("f3_one_minus_SL10"),
                "is_baseline": bool(result.get("is_baseline")),
                "has_unstable_period": bool(result.get("has_unstable_slot")),
                "unstable_slots": ", ".join(result.get("unstable_slots", [])),
            }
        )
        records.append(record)

    solutions_df = pd.DataFrame(records)
    if solutions_df.empty:
        print("未生成任何方案，无法执行帕累托分析。")
        return solutions_df

    obj2 = solutions_df[["f1_cost", "f2_Ttotal_mean"]].to_numpy()
    obj3 = solutions_df[["f1_cost", "f2_Ttotal_mean", "f3_one_minus_SL10"]].to_numpy()
    solutions_df["pareto_2obj"] = is_pareto_efficient(obj2)
    solutions_df["pareto_3obj"] = is_pareto_efficient(obj3)

    csv_path = Path("q2_all_solutions.csv")
    xlsx_path = Path("q2_all_solutions.xlsx")
    solutions_df.to_csv(csv_path, index=False)
    solutions_df.to_excel(xlsx_path, index=False)
    print(f"已保存全部方案列表至 {csv_path} 与 {xlsx_path}")

    display_cols = ["is_baseline"] + [f"x{i}" for i in range(1, 7)] + [
        "f1_cost",
        "f2_Ttotal_mean",
        "SL10_day",
    ]

    def _format_currency(value: float) -> str:
        return "NaN" if pd.isna(value) else f"{value:,.0f}"

    def _format_minutes(value: float) -> str:
        return "NaN" if pd.isna(value) else f"{value:.2f}"

    def _format_ratio(value: float) -> str:
        return "NaN" if pd.isna(value) else f"{value:.1%}"

    formatters = {
        "f1_cost": _format_currency,
        "f2_Ttotal_mean": _format_minutes,
        "SL10_day": _format_ratio,
        "is_baseline": lambda value: "是" if bool(value) else "否",
    }

    def _print_front(column: str, label: str) -> int:
        subset = solutions_df[solutions_df[column]].copy()
        subset.sort_values(["f1_cost", "f2_Ttotal_mean"], inplace=True)
        count = len(subset)
        if count == 0:
            print(f"\n{label}：暂无帕累托前沿方案。")
            return 0
        print(f"\n{label}（共 {count} 个）:")
        print(subset[display_cols].to_string(index=False, formatters=formatters))
        return count

    count_2d = _print_front("pareto_2obj", "二维目标帕累托前沿")
    count_3d = _print_front("pareto_3obj", "三维目标帕累托前沿")

    print(
        f"\n在二维目标（成本, 平均总耗时）下，共有 {count_2d} 个方案不被其它方案同时在成本和耗时两方面击败。"
    )
    if count_3d > count_2d:
        relation_desc = "方案数量略有增加"
    elif count_3d < count_2d:
        relation_desc = "方案数量有所减少"
    else:
        relation_desc = "方案数量保持一致"
    print(f"在三维目标下（附加 f3 = 1-SL10），{relation_desc}，共有 {count_3d} 个方案位于前沿。")

    return solutions_df


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    def _convert(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer, float, np.floating)) and not pd.isna(value):
            return bool(int(value))
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "t", "1", "yes", "y"}:
                return True
            if text in {"false", "f", "0", "no", "n", ""}:
                return False
        if pd.isna(value):
            return False
        return bool(value)

    return series.apply(_convert).astype(bool)


def _select_top_pareto_ids(df: pd.DataFrame, top_n: int = 5) -> List[int]:
    if "pareto_2obj" not in df.columns or "solution_id" not in df.columns:
        return []
    pareto_mask = df["pareto_2obj"]
    if pareto_mask.dtype != bool:
        pareto_mask = _coerce_bool_series(pareto_mask)
    subset = df[pareto_mask].dropna(subset=["f2_Ttotal_mean"])
    if subset.empty:
        return []
    ordered = subset.sort_values("f2_Ttotal_mean", ascending=True)
    return ordered["solution_id"].head(max(0, top_n)).astype(int).tolist()


def plot_q2_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    y_multiplier: float = 1.0,
    annotate_ids: Optional[List[int]] = None,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    pareto_mask = df.get("pareto_2obj")
    if pareto_mask is None or pareto_mask.dtype != bool:
        pareto_mask = pd.Series(False, index=df.index)

    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce") * y_multiplier

    fig, ax = plt.subplots(figsize=(8, 6))
    common_kwargs = {"alpha": 0.8, "linewidth": 1.0}
    non_pareto = (~pareto_mask) & x_values.notna() & y_values.notna()
    pareto = pareto_mask & x_values.notna() & y_values.notna()

    ax.scatter(
        x_values[non_pareto],
        y_values[non_pareto],
        s=20,
        color="#90a4ae",
        label="普通方案",
        **common_kwargs,
    )
    ax.scatter(
        x_values[pareto],
        y_values[pareto],
        s=70,
        facecolors="none",
        edgecolors="#c62828",
        label="帕累托前沿方案",
        linewidth=1.5,
    )

    if annotate_ids:
        highlight = df[df["solution_id"].isin(annotate_ids)]
        for _, row in highlight.iterrows():
            x_val = row.get(x_col)
            y_val = row.get(y_col)
            if pd.isna(x_val) or pd.isna(y_val):
                continue
            ax.annotate(
                f"{int(row['solution_id'])}",
                (float(x_val), float(y_val) * y_multiplier),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color="#37474f",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"已生成散点图：{output_path}")


def run_q2_visualization_and_recommendations(
    csv_path: Path = Path("q2_all_solutions.csv"),
) -> None:
    """根据 Q2 穷举结果绘制散点图并输出推荐方案。"""

    if not csv_path.exists():
        print(f"提示：未找到 {csv_path}，暂无法绘制 Q2 散点图和推荐。")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("提示：q2_all_solutions.csv 为空，跳过可视化与推荐。")
        return

    configure_matplotlib_fonts()

    df = df.copy()
    df.insert(0, "solution_id", np.arange(len(df), dtype=int))
    numeric_cols = ["f1_cost", "f2_Ttotal_mean", "SL10_day"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    x_cols = [f"x{i}" for i in range(1, 7)]
    existing_x_cols = [col for col in x_cols if col in df.columns]
    if existing_x_cols:
        df[existing_x_cols] = (
            df[existing_x_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        )
    bool_cols = [
        "pareto_2obj",
        "pareto_3obj",
        "has_unstable_period",
        "is_baseline",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = _coerce_bool_series(df[col])
    top_pareto_ids = _select_top_pareto_ids(df)

    plot_q2_scatter(
        df,
        x_col="f1_cost",
        y_col="f2_Ttotal_mean",
        xlabel="总成本（元）",
        ylabel="平均总耗时（分钟）",
        output_path=Path("q2_scatter_cost_vs_Ttotal.png"),
        annotate_ids=top_pareto_ids,
    )
    plot_q2_scatter(
        df,
        x_col="f1_cost",
        y_col="SL10_day",
        xlabel="总成本（元）",
        ylabel="SL10 服务水平（%）",
        output_path=Path("q2_scatter_cost_vs_SL10.png"),
        y_multiplier=100.0,
    )

    def format_minutes(value: float) -> str:
        return f"{value:.2f}" if pd.notna(value) else "NaN"

    def format_ratio(value: float) -> str:
        return f"{value:.1%}" if pd.notna(value) else "NaN"

    def format_currency(value: float) -> str:
        return f"{value:,.0f}" if pd.notna(value) else "NaN"

    baseline_mask = pd.Series(False, index=df.index)
    if "is_baseline" in df.columns:
        baseline_mask = df["is_baseline"]
    elif existing_x_cols and len(existing_x_cols) == 6:
        baseline_mask = (df[existing_x_cols] == 0).all(axis=1)
    if not baseline_mask.any() and 0 <= baseline_id < len(df):
        baseline_mask = df["solution_id"] == baseline_id
    if baseline_mask.any():
        baseline_row = df[baseline_mask].iloc[0]
    else:
        baseline_row = df.iloc[0]
        print(
            f"警告：未能自动定位 x1~x6 全 0 的基准方案，已默认使用 solution_id={int(baseline_row['solution_id'])}。"
        )
    T_baseline = baseline_row.get("f2_Ttotal_mean", np.nan)
    df["benefit_per_cost"] = np.nan
    if pd.notna(T_baseline):
        improvement = T_baseline - df["f2_Ttotal_mean"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["benefit_per_cost"] = np.where(
                df["f1_cost"] > 0, improvement / df["f1_cost"], np.nan
            )

    print("\n=== Q2 方案可视化与推荐 ===")
    print(
        f"基准方案 solution_id={int(baseline_row['solution_id'])}，平均总耗时≈{format_minutes(T_baseline)} 分钟，SL10≈{format_ratio(baseline_row.get('SL10_day', np.nan))}。"
    )

    recommendations: List[Dict[str, object]] = []

    def build_row(label: str, row: pd.Series, note: str = "") -> Dict[str, object]:
        entry: Dict[str, object] = {
            "scenario": label,
            "solution_id": int(row["solution_id"]),
            "f1_cost": row.get("f1_cost"),
            "f2_Ttotal_mean": row.get("f2_Ttotal_mean"),
            "SL10_day": row.get("SL10_day"),
            "benefit_per_cost": row.get("benefit_per_cost"),
            "notes": note,
        }
        for col in x_cols:
            entry[col] = int(row.get(col, 0))
        return entry

    valid_df = df.dropna(subset=["f2_Ttotal_mean"])
    if not valid_df.empty:
        best_a = valid_df.sort_values(
            ["f2_Ttotal_mean", "SL10_day"], ascending=[True, False]
        ).iloc[0]
        recommendations.append(build_row("A", best_a, "理论最优（无预算约束）"))
        print(
            f"场景 A（无限预算）推荐方案：solution_id={int(best_a['solution_id'])}，成本≈{format_currency(best_a['f1_cost'])} 元，平均总耗时≈{format_minutes(best_a['f2_Ttotal_mean'])} 分钟，SL10≈{format_ratio(best_a['SL10_day'])}。"
        )

    budget_df = df[df["f1_cost"] <= B_limit].copy()
    note_b = ""
    meets_target = True
    if budget_df.empty:
        budget_df = df.copy()
        note_b = "没有满足预算约束的方案，退回全量搜索。"
        meets_target = False
    sl_filtered = budget_df[budget_df["SL10_day"] >= SL_target]
    if sl_filtered.empty:
        meets_target = False
        sl_filtered = budget_df
        if note_b:
            note_b += " "
        note_b += "达不到目标 SL10。"
    if not sl_filtered.empty:
        best_b = sl_filtered.sort_values("f2_Ttotal_mean").iloc[0]
        note_text = (
            "满足 SL10 目标" if meets_target else note_b or "未达到 SL10 目标"
        )
        recommendations.append(build_row("B", best_b, note_text))
        extra_desc = "" if meets_target else "（未达到 SL10 目标）"
        print(
            f"场景 B（预算 {format_currency(B_limit)} 元）推荐方案：solution_id={int(best_b['solution_id'])}，成本≈{format_currency(best_b['f1_cost'])} 元，平均总耗时≈{format_minutes(best_b['f2_Ttotal_mean'])} 分钟，SL10≈{format_ratio(best_b['SL10_day'])}{extra_desc}。"
        )

    benefit_df = df[df["benefit_per_cost"].notna()].sort_values(
        "benefit_per_cost", ascending=False
    )
    if not benefit_df.empty:
        top_c = benefit_df.head(3)
        for _, row in top_c.iterrows():
            recommendations.append(build_row("C", row, "单位成本效益优先"))
        best_c = top_c.iloc[0]
        others = ", ".join(str(int(sid)) for sid in top_c["solution_id"].tolist()[1:])
        extra = f"，备选：{others}" if others else ""
        print(
            f"场景 C（单位成本效益最大）推荐方案：solution_id={int(best_c['solution_id'])}，单位成本效益≈{best_c['benefit_per_cost']:.6f} 分钟/元，平均总耗时≈{format_minutes(best_c['f2_Ttotal_mean'])} 分钟，SL10≈{format_ratio(best_c['SL10_day'])}{extra}。"
        )
    else:
        print("场景 C：由于基准方案数据缺失或没有非零成本方案，无法计算单位成本效益。")

    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        rec_path = Path("q2_recommended_solutions.csv")
        rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
        print(f"已输出推荐方案表：{rec_path}")


def run_single_solution_evaluator_examples(base_df: pd.DataFrame) -> None:
    """
    在控制台演示 evaluate_solution，方便论文描述单方案的三目标表现。
    """

    print("\n=== Q3 单方案评估器 ===")
    scenarios = {
        "baseline(全0)": _build_measure_vector(),
        "全量落地": _build_measure_vector(x1=1, x2=1, x3=1, x4=1, x5=1, x6=1),
        "数智提效": _build_measure_vector(x1=1, x3=1, x4=1, x6=1),
    }

    for name, vector in scenarios.items():
        result = evaluate_solution(vector, base_df)
        cost = result["f1_cost"]
        t_total = result["f2_Ttotal_mean"]
        sl10 = result["SL10_day"]
        f3 = result["f3_one_minus_SL10"]

        def fmt_value(value: Optional[float], pattern: str) -> str:
            if value is None:
                return "NaN"
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return "NaN"
            if not math.isfinite(numeric_value):
                return "NaN"
            return pattern.format(numeric_value)

        cost_text = fmt_value(cost, "{:,.0f}")
        t_total_text = fmt_value(t_total, "{:.2f}")
        sl10_text = fmt_value(sl10, "{:.1%}")
        f3_text = fmt_value(f3, "{:.3f}")
        flags_desc = ", ".join(f"{k}={v}" for k, v in result["x"].items())

        scheme_label = "基准方案" if result.get("is_baseline") else "优化方案"
        print(f"\n方案《{name}》（{scheme_label}）")
        print(f"措施组合：{flags_desc}")
        print(f"总成本 ≈ {cost_text} 元")
        print(f"全日平均总耗时 ≈ {t_total_text} 分钟")
        print(f"SL10_day ≈ {sl10_text}，因此 f3 = 1-SL10_day = {f3_text}")
        if result["has_unstable_slot"]:
            slots = "、".join(result["unstable_slots"])
            print(f"提示：{slots} 时段不稳定，相关指标按惩罚值处理。")


def run_q2_scenario_examples(base_df: pd.DataFrame) -> None:
    """演示 Q2 典型方案（baseline/组合策略）的参数变化。"""

    print("\n=== Q2 场景参数生成器示例 ===")
    scenarios = {
        "baseline": _build_measure_vector(),
        "all_on": _build_measure_vector(x1=1, x2=1, x3=1, x4=1, x5=1, x6=1),
        "精益高峰": _build_measure_vector(x1=1, x3=1, x5=1, x6=1),
        "数字化提效": _build_measure_vector(x1=1, x4=1, x5=1),
    }

    results: Dict[str, Tuple[pd.DataFrame, float]] = {}
    numeric_cols = ["lambda_new", "c_new", "mu_new", "T_find_new"]
    for name, vector in scenarios.items():
        new_df, total_cost = apply_measures(base_df, vector)
        results[name] = (new_df, total_cost)
        print(f"\n--- 方案《{name}》 ---")
        measure_desc = ", ".join(f"{k}={v}" for k, v in vector.items())
        print(f"措施组合：{measure_desc}")
        print(f"实施成本约为 {total_cost:,.0f} 元。")

        peak_subset = new_df[new_df["time_slot"].isin(PEAK_SLOTS)]
        target_subset = peak_subset if not peak_subset.empty else new_df
        summary_cols = ["time_slot"] + numeric_cols
        formatters = {"time_slot": str}
        for col in numeric_cols:
            formatters[col] = _format_numeric
        print("关键时段参数预览：")
        print(target_subset[summary_cols].to_string(index=False, formatters=formatters))

    baseline_df = results["baseline"][0]
    baseline_df.to_csv("q2_example_params_baseline.csv", index=False)
    all_on_df = results["all_on"][0]
    all_on_df.to_csv("q2_example_params_all_on.csv", index=False)
    print("\n已生成示例参数表：q2_example_params_baseline.csv, q2_example_params_all_on.csv。")


def fit_regression_model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    has_concurrent = "concurrent" in df.columns
    X_cols = ["occupancy"] + (["concurrent"] if has_concurrent else [])
    X = sm.add_constant(df[X_cols])
    model = sm.OLS(df["T_find"], X, missing="drop")
    results = model.fit()
    print("\n=== OLS 回归摘要 ===")
    print(results.summary())
    return results


def build_summary_table(results: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    params = results.params
    bse = results.bse
    t_values = results.tvalues
    p_values = results.pvalues

    rows: List[Dict[str, object]] = []
    for name in params.index:
        label = "Intercept" if name.lower() == "const" else name
        rows.append(
            {
                "term": label,
                "coef": params[name],
                "std_err": bse[name],
                "t_value": t_values[name],
                "p_value": p_values[name],
                "R_squared": results.rsquared,
                "n": int(results.nobs),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("a3_regression_summary.csv", index=False, encoding="utf-8-sig")
    print("\n已保存回归结果表：a3_regression_summary.csv")
    return summary_df


def plot_scatter_with_regression(
    df: pd.DataFrame, results: sm.regression.linear_model.RegressionResultsWrapper
) -> None:
    has_concurrent = "concurrent" in df.columns
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["occupancy"], df["T_find"], color="#1f77b4", label="观测点")

    x_range = np.linspace(df["occupancy"].min(), df["occupancy"].max(), 100)
    if has_concurrent:
        avg_concurrent = df["concurrent"].mean()
        y_range = (
            results.params.get("const", 0.0)
            + results.params.get("occupancy", 0.0) * x_range
            + results.params.get("concurrent", 0.0) * avg_concurrent
        )
        label = f"回归线（concurrent={avg_concurrent:.2f}）"
    else:
        y_range = (
            results.params.get("const", 0.0)
            + results.params.get("occupancy", 0.0) * x_range
        )
        label = "回归线"
    ax.plot(x_range, y_range, color="#ff7f0e", label=label)
    ax.set_xlabel("occupancy")
    ax.set_ylabel("T_find (min)")
    ax.set_title("T_find 与 occupancy 关系")
    ax.legend()
    fig.tight_layout()
    fig.savefig("a3_scatter_T_find_vs_occupancy.png", dpi=300)
    plt.close(fig)
    print("已生成：a3_scatter_T_find_vs_occupancy.png")


def plot_concurrent_views(
    df: pd.DataFrame, results: sm.regression.linear_model.RegressionResultsWrapper
) -> None:
    if "concurrent" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["concurrent"], df["T_find"], color="#2ca02c", label="观测点")
    x_range = np.linspace(df["concurrent"].min(), df["concurrent"].max(), 100)
    avg_occupancy = df["occupancy"].mean()
    y_range = (
        results.params.get("const", 0.0)
        + results.params.get("occupancy", 0.0) * avg_occupancy
        + results.params.get("concurrent", 0.0) * x_range
    )
    ax.plot(x_range, y_range, color="#d62728", label=f"回归线（occupancy={avg_occupancy:.0f}）")
    ax.set_xlabel("concurrent")
    ax.set_ylabel("T_find (min)")
    ax.set_title("T_find 与 concurrent 关系")
    ax.legend()
    fig.tight_layout()
    fig.savefig("a3_scatter_T_find_vs_concurrent.png", dpi=300)
    plt.close(fig)
    print("已生成：a3_scatter_T_find_vs_concurrent.png")

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(6, 4))
        ax3d = fig.add_subplot(111, projection="3d")
        ax3d.scatter(df["occupancy"], df["concurrent"], df["T_find"], color="#9467bd")
        ax3d.set_xlabel("occupancy")
        ax3d.set_ylabel("concurrent")
        ax3d.set_zlabel("T_find (min)")
        ax3d.set_title("T_find 三维散点")
        fig.tight_layout()
        fig.savefig("a3_scatter_3d.png", dpi=300)
        plt.close(fig)
        print("已生成：a3_scatter_3d.png")
    except Exception as exc:  # pragma: no cover - matplotlib 3D 可选
        print(f"警告：3D 散点图生成失败：{exc}")


def plot_residual_diagnostics(
    df: pd.DataFrame, results: sm.regression.linear_model.RegressionResultsWrapper
) -> None:
    fitted = results.fittedvalues
    residuals = results.resid

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(fitted, residuals, color="#8c564b")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("预测值")
    ax.set_ylabel("残差")
    ax.set_title("残差 vs 预测值")
    fig.tight_layout()
    fig.savefig("a3_residual_vs_fitted.png", dpi=300)
    plt.close(fig)
    print("已生成：a3_residual_vs_fitted.png")

    fig = sm.qqplot(residuals, line="45", fit=True)
    fig.suptitle("残差 QQ 图")
    fig.savefig("a3_residual_qq.png", dpi=300)
    plt.close(fig)
    print("已生成：a3_residual_qq.png")


def print_console_summary(
    df: pd.DataFrame, results: sm.regression.linear_model.RegressionResultsWrapper
) -> None:
    has_concurrent = "concurrent" in df.columns
    beta1 = results.params.get("occupancy", float("nan"))
    beta2 = results.params.get("concurrent", float("nan"))
    n = int(results.nobs)

    equation = "T_find = β0 + β1 * occupancy"
    if has_concurrent:
        equation += " + β2 * concurrent"
    print("\n=== 中文解读 ===")
    print(f"回归方程：{equation}")

    if math.isnan(beta1):
        print("未能估计 β1，请检查数据。")
    elif beta1 > 0:
        print("β1 为正，说明货架占用水平越高，平均找件时间越长。")
    else:
        print("β1 为负，说明货架占用越高可能缩短找件时间（需进一步验证）。")

    if has_concurrent:
        if math.isnan(beta2):
            print("β2 未估计成功。")
        elif beta2 > 0:
            print("β2 为正，现场同时取件人数越多，找件时间可能上升，需要关注现场拥挤度。")
        else:
            print("β2 为负，更多协同人手可能缩短找件时间。")
    else:
        print("提示：当前文件缺少“同时取件人数”列，本轮模型为单自变量；future work 可以补齐 concurrent 后再拟合。")

    print(f"样本量 n = {n}，R² = {results.rsquared:.3f}")
    if n <= 8:
        print("样本数量较少，结果仅作定性参考，请在文稿中注明样本局限。")
    else:
        print("提示：回归模型基于历史样本，解释关系时仍需结合业务判断。")


def configure_matplotlib_fonts() -> None:
    """选择一个可用的中文字体，避免图表中的文字出现方框。"""
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen_font = None
    for name in FONT_CANDIDATES:
        if name in available_fonts:
            chosen_font = name
            break

    if chosen_font:
        plt.rcParams["font.sans-serif"] = [chosen_font, "DejaVu Sans"]
        print(f"已设置 Matplotlib 中文字体：{chosen_font}")
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        print("警告：未找到常见中文字体，图表可能仍出现符号问题。")
    plt.rcParams["axes.unicode_minus"] = False


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    mask = (~values.isna()) & (~weights.isna()) & (weights > 0)
    if not mask.any():
        return float("nan")
    total_weight = weights.loc[mask].sum()
    if total_weight == 0:
        return float("nan")
    return float((values.loc[mask] * weights.loc[mask]).sum() / total_weight)


def load_params_table_for_simulation(csv_path: Path) -> pd.DataFrame:
    """读取方案参数表并统一列名，确保仿真所需字段齐全。"""

    if not csv_path.exists():
        raise FileNotFoundError(f"未找到参数文件：{csv_path}")

    # 统一各类 *_new 列名，确保字段一致
    df = pd.read_csv(csv_path)
    df = normalize_param_columns(df)
    missing = [col for col in REQUIRED_PARAM_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少必要列：{', '.join(missing)}")
    return df


def apply_lambda_factor(
    params_df: pd.DataFrame, factor: float, peak_only: bool = False
) -> pd.DataFrame:
    """
    返回新的参数表，将 λ 按照指定倍率放大；可选只在高峰时段调整。
    """

    if factor <= 0:
        raise ValueError("lambda 放大量需为正数")
    if "lambda" not in params_df.columns or "time_slot" not in params_df.columns:
        raise ValueError("参数表缺少 lambda 或 time_slot 列，无法调整到达率")

    df = params_df.copy()
    slot_mask = pd.Series(True, index=df.index)
    if peak_only:
        normalized_slots = df["time_slot"].astype(str).str.strip()
        slot_mask = normalized_slots.isin(PEAK_SLOTS)
    df.loc[slot_mask, "lambda"] = df.loc[slot_mask, "lambda"].astype(float) * float(
        factor
    )
    return df


def apply_mu_factor(params_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """返回新的参数表，将 μ 按比例缩放。"""

    if factor <= 0:
        raise ValueError("mu 放大量需为正数")
    if "mu" not in params_df.columns:
        raise ValueError("参数表缺少 mu 列，无法调整服务率")

    df = params_df.copy()
    mu_series = pd.to_numeric(df["mu"], errors="coerce")
    df["mu"] = mu_series * float(factor)
    return df


def apply_tfind_factor(params_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """返回新的参数表，将找件时间按比例缩放。"""

    if factor <= 0:
        raise ValueError("T_find 放大量需为正数")
    if "T_find" not in params_df.columns:
        raise ValueError("参数表缺少 T_find 列，无法调整找件时间")

    df = params_df.copy()
    tfind_series = pd.to_numeric(df["T_find"], errors="coerce").fillna(
        DEFAULT_T_FIND_MIN
    )
    df["T_find"] = tfind_series * float(factor)
    return df


def apply_c_delta(params_df: pd.DataFrame, delta: float) -> pd.DataFrame:
    """返回新的参数表，将窗口数在保持不低于 1 的情况下整体平移。"""

    if "c" not in params_df.columns:
        raise ValueError("参数表缺少 c 列，无法调整窗口数")

    df = params_df.copy()
    c_series = pd.to_numeric(df["c"], errors="coerce").fillna(BASE_C)
    c_new = c_series + float(delta)
    c_new = c_new.clip(lower=1.0)
    df["c"] = c_new
    return df


def apply_c_cap(params_df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """返回新的参数表，强制窗口数不超过 cap（同时不低于 1）。"""

    if cap is None:
        return params_df.copy()
    if cap <= 0:
        raise ValueError("c 上限必须为正")
    if "c" not in params_df.columns:
        raise ValueError("参数表缺少 c 列，无法限制窗口数")

    df = params_df.copy()
    c_series = pd.to_numeric(df["c"], errors="coerce").fillna(BASE_C)
    df["c"] = c_series.clip(lower=1.0, upper=float(cap))
    return df


def _prepare_params_for_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """确保参数表包含仿真所需列，优先使用 *_new 列。"""

    rename_map = {}
    for src, target in (
        ("lambda_new", "lambda"),
        ("mu_new", "mu"),
        ("c_new", "c"),
        ("T_find_new", "T_find"),
    ):
        if src in df.columns:
            rename_map[src] = target
    prepared = df.rename(columns=rename_map)
    prepared = normalize_param_columns(prepared)
    missing = [col for col in REQUIRED_PARAM_COLUMNS if col not in prepared.columns]
    if missing:
        raise ValueError(f"参数表缺少必要列：{', '.join(missing)}")
    return prepared


def simulate_params_multiple_times(
    params_df: pd.DataFrame, n_rep: int, base_seed: int = DEFAULT_BASE_SEED
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """对同一参数表重复仿真，返回每次复制与均值摘要。"""

    if n_rep <= 0:
        raise ValueError("n_rep 需为正整数")

    rep_records: List[Dict[str, float]] = []
    for rep in range(n_rep):
        stats_df, queue_monitor_df = run_single_replication(
            params_df, random_seed=base_seed + rep
        )
        metrics = compute_replication_metrics(stats_df, queue_monitor_df)
        metrics["rep"] = rep
        rep_records.append(metrics)

    rep_df = pd.DataFrame(rep_records)
    summary = summarize_replication_metrics(rep_df)
    summary["n_rep"] = len(rep_df)
    return rep_df, summary


def summarize_replication_metrics(rep_df: pd.DataFrame) -> Dict[str, float]:
    """对复制级指标取均值/标准差，便于灵敏度分析使用。"""

    summary = {
        "avg_total_time_mean": np.nan,
        "avg_total_time_std": np.nan,
        "avg_wait_time_mean": np.nan,
        "avg_wait_time_std": np.nan,
        "SL10_mean": np.nan,
        "SL10_std": np.nan,
    }

    if rep_df.empty:
        return summary

    for col, mean_key, std_key in (
        ("avg_total_time", "avg_total_time_mean", "avg_total_time_std"),
        ("avg_wait_time", "avg_wait_time_mean", "avg_wait_time_std"),
        ("SL10", "SL10_mean", "SL10_std"),
    ):
        if col in rep_df.columns:
            summary[mean_key] = float(rep_df[col].mean())
            summary[std_key] = float(rep_df[col].std(ddof=0))
    return summary


def compute_replication_metrics(
    stats_df: pd.DataFrame, queue_monitor_df: pd.DataFrame
) -> Dict[str, float]:
    """对单次复制的输出计算关键指标。"""

    metrics = {
        "avg_total_time": np.nan,
        "avg_wait_time": np.nan,
        "total_time_p50": np.nan,
        "total_time_p90": np.nan,
        "wait_time_p50": np.nan,
        "wait_time_p90": np.nan,
        "SL10": np.nan,
        "max_queue": np.nan,
        "avg_queue": np.nan,
    }

    if not stats_df.empty:
        # 分别计算不同耗时指标
        if "total_time" not in stats_df.columns:
            stats_df = stats_df.assign(
                total_time=stats_df["end_time"] - stats_df["arrival_time"]
            )
        metrics["avg_total_time"] = float(stats_df["total_time"].mean())
        if "wait_time" in stats_df.columns:
            metrics["avg_wait_time"] = float(stats_df["wait_time"].mean())
            non_na_wait = stats_df["wait_time"].dropna().values
            if non_na_wait.size:
                metrics["wait_time_p50"] = float(np.percentile(non_na_wait, 50))
                metrics["wait_time_p90"] = float(np.percentile(non_na_wait, 90))
        non_na_total = stats_df["total_time"].dropna().values
        if non_na_total.size:
            metrics["total_time_p50"] = float(np.percentile(non_na_total, 50))
            metrics["total_time_p90"] = float(np.percentile(non_na_total, 90))
        metrics["SL10"] = float((stats_df["total_time"] <= 10).mean())

    if not queue_monitor_df.empty:
        queue_col = None
        if "queue_len" in queue_monitor_df.columns:
            queue_col = "queue_len"
        elif "queue_length" in queue_monitor_df.columns:
            queue_col = "queue_length"
        if queue_col is not None:
            metrics["max_queue"] = float(queue_monitor_df[queue_col].max())
            metrics["avg_queue"] = float(queue_monitor_df[queue_col].mean())

    return metrics


def build_replication_summary(
    rep_df: pd.DataFrame, scenario_order: Sequence[str]
) -> pd.DataFrame:
    """按方案聚合复制结果，输出均值与标准差。"""

    summary_columns = [
        "scenario_name",
        "avg_total_time_mean",
        "avg_total_time_std",
        "total_time_p50_mean",
        "total_time_p50_std",
        "total_time_p90_mean",
        "total_time_p90_std",
        "avg_wait_time_mean",
        "avg_wait_time_std",
        "wait_time_p50_mean",
        "wait_time_p50_std",
        "wait_time_p90_mean",
        "wait_time_p90_std",
        "SL10_mean",
        "SL10_std",
        "max_queue_mean",
        "max_queue_std",
        "avg_queue_mean",
        "avg_queue_std",
    ]

    if rep_df.empty:
        return pd.DataFrame(columns=summary_columns)

    summary_df = (
        rep_df.groupby("scenario_name", sort=False)
        .agg(
            avg_total_time_mean=("avg_total_time", "mean"),
            avg_total_time_std=("avg_total_time", "std"),
            total_time_p50_mean=("total_time_p50", "mean"),
            total_time_p50_std=("total_time_p50", "std"),
            total_time_p90_mean=("total_time_p90", "mean"),
            total_time_p90_std=("total_time_p90", "std"),
            avg_wait_time_mean=("avg_wait_time", "mean"),
            avg_wait_time_std=("avg_wait_time", "std"),
            wait_time_p50_mean=("wait_time_p50", "mean"),
            wait_time_p50_std=("wait_time_p50", "std"),
            wait_time_p90_mean=("wait_time_p90", "mean"),
            wait_time_p90_std=("wait_time_p90", "std"),
            SL10_mean=("SL10", "mean"),
            SL10_std=("SL10", "std"),
            max_queue_mean=("max_queue", "mean"),
            max_queue_std=("max_queue", "std"),
            avg_queue_mean=("avg_queue", "mean"),
            avg_queue_std=("avg_queue", "std"),
        )
        .reset_index()
    )

    summary_df = (
        summary_df.set_index("scenario_name")
        .reindex(scenario_order)
        .reset_index()
        .dropna(how="all")
    )
    summary_df.columns = summary_columns
    return summary_df


def build_lambda_factor_summary(
    rep_df: pd.DataFrame,
    scenario_order: Sequence[str],
    lambda_factors: Sequence[float],
) -> pd.DataFrame:
    """
    针对 (方案, λ 倍率) 组合聚合复制结果，并计算相对基准的变化量。
    """

    summary_columns = [
        "scenario_name",
        "lambda_factor",
        "avg_total_time_mean",
        "avg_total_time_std",
        "avg_wait_time_mean",
        "avg_wait_time_std",
        "SL10_mean",
        "SL10_std",
        "max_queue_mean",
        "max_queue_std",
        "avg_queue_mean",
        "avg_queue_std",
        "delta_total_time",
        "delta_SL10",
        "delta_max_queue",
        "delta_avg_queue",
    ]

    if rep_df.empty:
        return pd.DataFrame(columns=summary_columns)

    lambda_order = list(dict.fromkeys(float(f) for f in lambda_factors))
    grouped = (
        rep_df.groupby(["scenario_name", "lambda_factor"], sort=False)
        .agg(
            avg_total_time_mean=("avg_total_time", "mean"),
            avg_total_time_std=("avg_total_time", "std"),
            avg_wait_time_mean=("avg_wait_time", "mean"),
            avg_wait_time_std=("avg_wait_time", "std"),
            SL10_mean=("SL10", "mean"),
            SL10_std=("SL10", "std"),
            max_queue_mean=("max_queue", "mean"),
            max_queue_std=("max_queue", "std"),
            avg_queue_mean=("avg_queue", "mean"),
            avg_queue_std=("avg_queue", "std"),
        )
        .reset_index()
    )

    scenario_cat = pd.Categorical(
        grouped["scenario_name"], categories=list(scenario_order), ordered=True
    )
    factor_cat = pd.Categorical(
        grouped["lambda_factor"], categories=lambda_order, ordered=True
    )
    grouped = (
        grouped.assign(_scenario=scenario_cat, _factor=factor_cat)
        .sort_values(by=["_scenario", "_factor"])
        .drop(columns=["_scenario", "_factor"])
        .reset_index(drop=True)
    )
    summary_df = grouped.copy()

    baseline_factor = next(
        (factor for factor in lambda_order if math.isclose(factor, 1.0)), None
    )
    delta_cols = {
        "avg_total_time_mean": "delta_total_time",
        "SL10_mean": "delta_SL10",
        "max_queue_mean": "delta_max_queue",
        "avg_queue_mean": "delta_avg_queue",
    }
    for col in delta_cols.values():
        summary_df[col] = np.nan

    if baseline_factor is not None:
        baseline_mask = np.isclose(
            summary_df["lambda_factor"].astype(float), float(baseline_factor)
        )
        baseline_df = summary_df.loc[
            baseline_mask,
            [
                "scenario_name",
                "avg_total_time_mean",
                "SL10_mean",
                "max_queue_mean",
                "avg_queue_mean",
            ],
        ].rename(
            columns={
                "avg_total_time_mean": "avg_total_time_mean_baseline",
                "SL10_mean": "SL10_mean_baseline",
                "max_queue_mean": "max_queue_mean_baseline",
                "avg_queue_mean": "avg_queue_mean_baseline",
            }
        )
        summary_df = summary_df.merge(
            baseline_df, on="scenario_name", how="left", copy=False
        )
        summary_df["delta_total_time"] = (
            summary_df["avg_total_time_mean"]
            - summary_df["avg_total_time_mean_baseline"]
        )
        summary_df["delta_SL10"] = (
            summary_df["SL10_mean"] - summary_df["SL10_mean_baseline"]
        )
        summary_df["delta_max_queue"] = (
            summary_df["max_queue_mean"] - summary_df["max_queue_mean_baseline"]
        )
        summary_df["delta_avg_queue"] = (
            summary_df["avg_queue_mean"] - summary_df["avg_queue_mean_baseline"]
        )
        summary_df = summary_df.drop(
            columns=[
                "avg_total_time_mean_baseline",
                "SL10_mean_baseline",
                "max_queue_mean_baseline",
                "avg_queue_mean_baseline",
            ]
        )

    summary_df = summary_df.reindex(columns=summary_columns)
    return summary_df


def plot_metric_with_error(
    summary_df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """画出带误差棒的对比柱状图。"""

    if summary_df.empty or mean_col not in summary_df.columns:
        return

    x_pos = np.arange(len(summary_df))
    means = summary_df[mean_col].astype(float)
    errors = summary_df[std_col].fillna(0.0).astype(float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_pos, means, yerr=errors, capsize=4, color="#4C72B0")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary_df["scenario_name"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_single_solution_param_sensitivity(
    base_params_path: Path = Q3_SENS_BASE_PARAMS,
    factors: Sequence[float] = Q3_SENS_FACTORS,
    c_deltas: Sequence[float] = Q3_SENS_C_DELTAS,
    n_rep: int = Q3_SENS_N_REP,
    base_seed: int = DEFAULT_BASE_SEED,
) -> pd.DataFrame:
    """围绕单个方案进行 λ/μ/T_find/c 的参数敏感性仿真。"""

    if not base_params_path.exists():
        print(
            f"提示：未找到 {base_params_path}，暂无法执行单方案敏感性分析。"
        )
        return pd.DataFrame()

    base_df = load_params_table_for_simulation(base_params_path)
    records: List[Dict[str, object]] = []
    for factor in factors:
        variant_df = apply_lambda_factor(base_df, float(factor))
        _, summary = simulate_params_multiple_times(variant_df, n_rep, base_seed)
        records.append(
            {
                "param_type": "lambda",
                "factor": float(factor),
                "delta": np.nan,
                **summary,
            }
        )

    for factor in factors:
        variant_df = apply_mu_factor(base_df, float(factor))
        _, summary = simulate_params_multiple_times(variant_df, n_rep, base_seed)
        records.append(
            {
                "param_type": "mu",
                "factor": float(factor),
                "delta": np.nan,
                **summary,
            }
        )

    for factor in factors:
        variant_df = apply_tfind_factor(base_df, float(factor))
        _, summary = simulate_params_multiple_times(variant_df, n_rep, base_seed)
        records.append(
            {
                "param_type": "T_find",
                "factor": float(factor),
                "delta": np.nan,
                **summary,
            }
        )

    for delta in c_deltas:
        variant_df = apply_c_delta(base_df, float(delta))
        _, summary = simulate_params_multiple_times(variant_df, n_rep, base_seed)
        records.append(
            {
                "param_type": "c",
                "factor": np.nan,
                "delta": float(delta),
                **summary,
            }
        )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        print("单方案敏感性结果为空。")
        return result_df

    result_df.to_csv(Q3_SENS_SINGLE_OUTPUT, index=False)
    print(f"已输出单方案敏感性数据：{Q3_SENS_SINGLE_OUTPUT}")

    configure_matplotlib_fonts()
    plot_param_sensitivity_lines(
        result_df,
        param_type="lambda",
        x_col="factor",
        x_label="λ 因子",
        output_path=Q3_SENS_LAMBDA_PLOT,
        title="λ 因子敏感性",
    )
    plot_param_sensitivity_lines(
        result_df,
        param_type="mu",
        x_col="factor",
        x_label="μ 因子",
        output_path=Q3_SENS_MU_PLOT,
        title="μ 因子敏感性",
    )
    plot_param_sensitivity_lines(
        result_df,
        param_type="T_find",
        x_col="factor",
        x_label="T_find 因子",
        output_path=Q3_SENS_TFIND_PLOT,
        title="T_find 因子敏感性",
    )
    plot_param_sensitivity_lines(
        result_df,
        param_type="c",
        x_col="delta",
        x_label="窗口数增量",
        output_path=Q3_SENS_C_PLOT,
        title="窗口数敏感性",
    )

    _print_param_sensitivity_comments(result_df)
    return result_df


def _print_param_sensitivity_comments(result_df: pd.DataFrame) -> None:
    """在控制台输出几个典型因子变化的中文解读。"""

    def _describe(
        df: pd.DataFrame,
        param_type: str,
        label: str,
        base_value: float,
        target_value: float,
        value_col: str = "factor",
    ) -> None:
        subset = df[df["param_type"] == param_type]
        if subset.empty or value_col not in subset.columns:
            return
        base_row = subset[np.isclose(subset[value_col], base_value)]
        target_row = subset[np.isclose(subset[value_col], target_value)]
        if base_row.empty or target_row.empty:
            return
        base_row = base_row.iloc[0]
        target_row = target_row.iloc[0]
        delta_total = target_row["avg_total_time_mean"] - base_row["avg_total_time_mean"]
        delta_sl10 = target_row["SL10_mean"] - base_row["SL10_mean"]
        if value_col == "delta":
            target_label = f"{target_value:+.1f}"
        else:
            target_label = f"{target_value:.1f}x"
        print(
            f"当 {label} 调整至 {target_label} 时，"
            f"平均总耗时约 {target_row['avg_total_time_mean']:.2f} 分钟（变化 {delta_total:+.2f}），"
            f"SL10≈{target_row['SL10_mean']:.1%}（变化 {delta_sl10:+.1%}）。"
        )

    print("\n=== Q3 单方案参数敏感性结论 ===")
    _describe(result_df, "lambda", "λ", base_value=1.0, target_value=1.2)
    _describe(result_df, "mu", "μ", base_value=1.0, target_value=1.2)
    _describe(result_df, "T_find", "T_find", base_value=1.0, target_value=1.2)
    _describe(result_df, "c", "c", base_value=0.0, target_value=1.0, value_col="delta")


def run_budget_sensitivity_analysis(
    solutions_csv: Path = Path("q2_all_solutions.csv"),
    budgets: Sequence[float] = B_LIST,
    sl_min: float = SL_MIN,
    n_rep: int = Q3_SENS_N_REP,
    base_seed: int = DEFAULT_BASE_SEED,
) -> pd.DataFrame:
    """在不同预算约束下选择方案并进行仿真验证。"""

    if not solutions_csv.exists():
        print(f"提示：未找到 {solutions_csv}，无法执行预算灵敏度分析。")
        return pd.DataFrame()
    if not Q1_PARAMS_FROM_EXCEL_CSV.exists():
        print(f"提示：缺少 {Q1_PARAMS_FROM_EXCEL_CSV}，无法重构方案参数表。")
        return pd.DataFrame()

    solutions_df = pd.read_csv(solutions_csv)
    if solutions_df.empty:
        print("q2_all_solutions.csv 为空，跳过预算灵敏度分析。")
        return solutions_df

    if "solution_id" not in solutions_df.columns:
        solutions_df.insert(0, "solution_id", np.arange(len(solutions_df), dtype=int))

    numeric_cols = ["f1_cost", "f2_Ttotal_mean", "SL10_day"]
    for col in numeric_cols:
        if col in solutions_df.columns:
            solutions_df[col] = pd.to_numeric(solutions_df[col], errors="coerce")

    x_cols = [f"x{i}" for i in range(1, 7)]
    for col in x_cols:
        if col in solutions_df.columns:
            solutions_df[col] = (
                pd.to_numeric(solutions_df[col], errors="coerce").fillna(0).astype(int)
            )

    base_df = load_q1_params_for_q2()
    records: List[Dict[str, object]] = []
    for budget in budgets:
        feasible = solutions_df[solutions_df["f1_cost"] <= budget]
        feasible = feasible[feasible["SL10_day"] >= sl_min]
        if feasible.empty:
            print(
                f"[Q3 预算敏感性] 预算 {budget:,.0f} 元下没有满足稳定性/服务水平约束的可行方案。"
            )
            records.append(
                {
                    "budget": budget,
                    "solution_id": None,
                    "avg_total_time_mean": None,
                    "SL10_mean": None,
                    "has_feasible_solution": False,
                }
            )
            continue

        best_row = feasible.sort_values("f2_Ttotal_mean").iloc[0]
        vector = {col: int(best_row.get(col, 0)) for col in x_cols}
        params_df, _ = apply_measures(base_df, vector)
        sim_df = _prepare_params_for_simulation(params_df)
        _, summary = simulate_params_multiple_times(sim_df, n_rep, base_seed)
        record = {
            "budget": budget,
            "solution_id": int(best_row["solution_id"]),
            "f1_cost": best_row.get("f1_cost"),
            "f2_Ttotal_mean": best_row.get("f2_Ttotal_mean"),
            "SL10_day": best_row.get("SL10_day"),
            "avg_total_time_mean": summary.get("avg_total_time_mean"),
            "SL10_mean": summary.get("SL10_mean"),
            "has_feasible_solution": True,
        }
        for col in x_cols:
            record[col] = vector.get(col, 0)
        records.append(record)
        print(
            f"预算 {budget:,.0f} 元：推荐方案 solution_id={record['solution_id']}，"
            f"仿真平均总耗时≈{record['avg_total_time_mean']:.2f} 分钟，SL10≈{record['SL10_mean']:.1%}。"
        )

    result_df = pd.DataFrame(records)
    if "has_feasible_solution" not in result_df.columns:
        # 当所有预算都无可行方案时，通过前面的 append 也会带出该列；此处仅为安全兜底。
        result_df["has_feasible_solution"] = False
    result_df.to_csv(Q3_SENS_BUDGET_OUTPUT, index=False)
    print(f"已输出预算灵敏度数据：{Q3_SENS_BUDGET_OUTPUT}")

    feasible_result = result_df[result_df["has_feasible_solution"] == True]
    if feasible_result.empty:
        print("[Q3 预算敏感性] 所有预算档位都没有可行方案，图像中不会显示任何点。")
    else:
        configure_matplotlib_fonts()
        plot_param_sensitivity_lines(
            feasible_result.assign(param_type="budget", factor=feasible_result["budget"]),
            param_type="budget",
            x_col="factor",
            x_label="预算（元）",
            output_path=Q3_SENS_BUDGET_PLOT,
            title="预算 vs 仿真表现",
        )
        infeasible_budgets = result_df[~result_df["has_feasible_solution"]]["budget"].tolist()
        if infeasible_budgets:
            budget_list = "、".join(f"{b:,.0f}" for b in infeasible_budgets)
            print(
                f"[Q3 预算敏感性] 注：预算 {budget_list} 元 无可行方案，因此未在图中出现。"
            )

    _print_budget_comments(result_df)
    return result_df


def _print_budget_comments(result_df: pd.DataFrame) -> None:
    if "has_feasible_solution" in result_df.columns:
        valid = result_df[result_df["has_feasible_solution"] == True]
    else:
        valid = result_df
    valid = valid.dropna(subset=["avg_total_time_mean"])
    if valid.empty:
        return
    valid = valid.sort_values("budget")
    first = valid.iloc[0]
    last = valid.iloc[-1]
    print(
        "随着预算从 {low:.0f} 元增加到 {high:.0f} 元，平均总耗时由 {t_low:.2f} 分钟 "
        "降至 {t_high:.2f} 分钟，SL10 从 {sl_low:.1%} 提升至 {sl_high:.1%}，边际改善逐渐收敛。".format(
            low=first["budget"],
            high=last["budget"],
            t_low=first["avg_total_time_mean"],
            t_high=last["avg_total_time_mean"],
            sl_low=first["SL10_mean"],
            sl_high=last["SL10_mean"],
        )
    )


def plot_lambda_factor_metric(
    summary_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """绘制 λ 放大因子与指标之间的折线图。"""

    if summary_df.empty or value_col not in summary_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for scenario_name, group_df in summary_df.groupby("scenario_name", sort=False):
        group_sorted = group_df.sort_values("lambda_factor")
        ax.plot(
            group_sorted["lambda_factor"],
            group_sorted[value_col],
            marker="o",
            label=scenario_name,
        )
    ax.set_xlabel("λ 放大因子")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_high_demand_stress_tests(
    scenario_files: Dict[str, Path],
    stress_cases: Sequence[Dict[str, Any]] = HIGH_DEMAND_CASES,
    n_rep: int = HIGH_DEMAND_N_REP,
    base_seed: int = DEFAULT_BASE_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """在多个极端高负荷场景下仿真不同方案。"""

    if not stress_cases:
        raise ValueError("stress_cases 不能为空")
    if n_rep <= 0:
        raise ValueError("n_rep 必须为正整数")

    scenario_order = list(scenario_files.keys())
    case_order: List[str] = []
    rep_records: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(stress_cases):
        case_name = str(case.get("case_name") or f"stress_{case_idx + 1}")
        if case_name not in case_order:
            case_order.append(case_name)
        lambda_factor = float(case.get("lambda_factor", 1.0))
        mu_factor = float(case.get("mu_factor", 1.0))
        c_cap = case.get("c_cap")
        peak_only = bool(case.get("peak_only", False))
        case_note = str(case.get("note", ""))

        for scenario_name, csv_path in scenario_files.items():
            params_df = load_params_table_for_simulation(Path(csv_path))
            stress_df = params_df.copy()
            if not math.isclose(lambda_factor, 1.0):
                stress_df = apply_lambda_factor(
                    stress_df, lambda_factor, peak_only=peak_only
                )
            if not math.isclose(mu_factor, 1.0):
                stress_df = apply_mu_factor(stress_df, mu_factor)
            if c_cap is not None:
                stress_df = apply_c_cap(stress_df, float(c_cap))

            for rep in range(n_rep):
                stats_df, queue_monitor_df = run_single_replication(
                    stress_df, random_seed=base_seed + rep
                )
                metrics = compute_replication_metrics(stats_df, queue_monitor_df)
                metrics.update(
                    {
                        "scenario_name": scenario_name,
                        "case_name": case_name,
                        "lambda_factor": lambda_factor,
                        "mu_factor": mu_factor,
                        "c_cap": float(c_cap) if c_cap is not None else np.nan,
                        "case_note": case_note,
                        "peak_only": peak_only,
                        "rep": rep,
                    }
                )
                rep_records.append(metrics)

    rep_df = pd.DataFrame(rep_records)
    summary_df = build_high_demand_summary(rep_df, scenario_order, case_order)
    return rep_df, summary_df


def build_high_demand_summary(
    rep_df: pd.DataFrame, scenario_order: Sequence[str], case_order: Sequence[str]
) -> pd.DataFrame:
    """对 (方案, 高负荷场景) 组合聚合复制结果。"""

    summary_columns = [
        "scenario_name",
        "case_name",
        "lambda_factor",
        "mu_factor",
        "c_cap",
        "case_note",
        "avg_total_time_mean",
        "avg_total_time_std",
        "avg_wait_time_mean",
        "avg_wait_time_std",
        "SL10_mean",
        "SL10_std",
        "max_queue_mean",
        "max_queue_std",
        "avg_queue_mean",
        "avg_queue_std",
    ]

    if rep_df.empty:
        return pd.DataFrame(columns=summary_columns)

    grouped = (
        rep_df.groupby(["scenario_name", "case_name"], sort=False)
        .agg(
            lambda_factor=("lambda_factor", "mean"),
            mu_factor=("mu_factor", "mean"),
            c_cap=("c_cap", "mean"),
            case_note=("case_note", "first"),
            avg_total_time_mean=("avg_total_time", "mean"),
            avg_total_time_std=("avg_total_time", "std"),
            avg_wait_time_mean=("avg_wait_time", "mean"),
            avg_wait_time_std=("avg_wait_time", "std"),
            SL10_mean=("SL10", "mean"),
            SL10_std=("SL10", "std"),
            max_queue_mean=("max_queue", "mean"),
            max_queue_std=("max_queue", "std"),
            avg_queue_mean=("avg_queue", "mean"),
            avg_queue_std=("avg_queue", "std"),
        )
        .reset_index()
    )

    scenario_cat = pd.Categorical(
        grouped["scenario_name"], categories=list(scenario_order), ordered=True
    )
    case_cat = pd.Categorical(
        grouped["case_name"], categories=list(case_order), ordered=True
    )
    summary_df = (
        grouped.assign(_scenario=scenario_cat, _case=case_cat)
        .sort_values(["_case", "_scenario"])
        .drop(columns=["_scenario", "_case"])
        .reset_index(drop=True)
    )
    summary_df = summary_df.reindex(columns=summary_columns)
    return summary_df


def plot_high_demand_metric(
    summary_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    case_order: Sequence[str],
) -> None:
    """绘制高负荷场景下方案的折线比较。"""

    if summary_df.empty or value_col not in summary_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for scenario_name, group_df in summary_df.groupby("scenario_name", sort=False):
        group_sorted = group_df.copy()
        if case_order:
            category = pd.Categorical(
                group_sorted["case_name"], categories=list(case_order), ordered=True
            )
            group_sorted = (
                group_sorted.assign(_case=category)
                .sort_values("_case")
                .drop(columns="_case")
            )
        ax.plot(
            group_sorted["case_name"],
            group_sorted[value_col],
            marker="o",
            label=scenario_name,
        )

    ax.set_xlabel("高负荷场景")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_high_demand_conclusion(
    summary_df: pd.DataFrame, case_order: Sequence[str]
) -> None:
    """输出高负荷鲁棒性中文摘要。"""

    if summary_df.empty:
        print("高负荷鲁棒性结果为空，无法输出结论。")
        return

    order = list(case_order) if case_order else summary_df["case_name"].unique()
    note_map = (
        summary_df.dropna(subset=["case_name"])
        .drop_duplicates("case_name")
        .set_index("case_name")
        ["case_note"]
    )
    if note_map.empty:
        note_map = pd.Series(dtype=str)

    print("\n=== Q3 高负荷鲁棒性小结 ===")
    for case_name in order:
        subset = summary_df[summary_df["case_name"] == case_name]
        if subset.empty:
            continue
        note = ""
        if isinstance(note_map, pd.Series) and case_name in note_map.index:
            raw_note = note_map.loc[case_name]
            note = f"（{raw_note}）" if isinstance(raw_note, str) and raw_note else ""
        scenario_series = subset["scenario_name"].astype(str)
        baseline_row = subset[scenario_series.str.lower() == "baseline"]
        baseline_metrics = baseline_row.iloc[0] if not baseline_row.empty else None
        best_row = subset.sort_values("avg_total_time_mean").iloc[0]
        print(
            f"场景 {case_name}{note}: {best_row['scenario_name']} 平均总耗时"
            f"≈{best_row['avg_total_time_mean']:.2f} 分钟，SL10≈{best_row['SL10_mean']:.1%}。"
        )
        if baseline_metrics is None:
            continue
        delta_total = best_row["avg_total_time_mean"] - baseline_metrics[
            "avg_total_time_mean"
        ]
        delta_sl10 = best_row["SL10_mean"] - baseline_metrics["SL10_mean"]
        delta_queue = best_row["avg_queue_mean"] - baseline_metrics["avg_queue_mean"]
        if baseline_metrics["scenario_name"] == best_row["scenario_name"]:
            print(
                "   baseline 方案依旧最稳健，平均总耗时变化 {dt:+.2f} 分钟，"
                "SL10 {dsl:+.1%}，平均队长 {dq:+.1f} 人。".format(
                    dt=delta_total, dsl=delta_sl10, dq=delta_queue
                )
            )
        else:
            print(
                "   与 baseline 相比：平均总耗时 {dt:+.2f} 分钟，SL10 {dsl:+.1%}，"
                "平均队长 {dq:+.1f} 人。".format(
                    dt=delta_total, dsl=delta_sl10, dq=delta_queue
                )
            )

def run_replications_for_scenarios(
    scenario_files: Dict[str, Path],
    n_rep: int = N_REP,
    base_seed: int = DEFAULT_BASE_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    针对多个方案执行多次复制仿真。

    返回:
        rep_df: 每次复制的原始指标
        summary_df: 按方案聚合后的统计
    """

    if n_rep <= 0:
        raise ValueError("n_rep 必须为正整数")

    rep_records: List[Dict[str, Any]] = []
    scenario_order = list(scenario_files.keys())

    for scenario_name, csv_path in scenario_files.items():
        params_df = load_params_table_for_simulation(Path(csv_path))
        # 循环不同随机种子，保证复制间独立
        for rep in range(n_rep):
            stats_df, queue_monitor_df = run_single_replication(
                params_df, random_seed=base_seed + rep
            )
            metrics = compute_replication_metrics(stats_df, queue_monitor_df)
            metrics["scenario_name"] = scenario_name
            metrics["rep"] = rep
            rep_records.append(metrics)

    rep_df = pd.DataFrame(rep_records)
    summary_df = build_replication_summary(rep_df, scenario_order)
    return rep_df, summary_df


def run_lambda_robustness_analysis(
    scenario_files: Dict[str, Path],
    lambda_factors: Sequence[float],
    n_rep: int = ROBUSTNESS_N_REP,
    base_seed: int = DEFAULT_BASE_SEED,
    peak_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """在不同 λ 放大倍率下重复仿真，衡量方案鲁棒性。"""

    if n_rep <= 0:
        raise ValueError("n_rep 必须为正整数")
    if not lambda_factors:
        raise ValueError("lambda_factors 不能为空")

    rep_records: List[Dict[str, Any]] = []
    scenario_order = list(scenario_files.keys())

    for scenario_name, csv_path in scenario_files.items():
        params_df = load_params_table_for_simulation(Path(csv_path))
        for factor in lambda_factors:
            factor_df = apply_lambda_factor(
                params_df, float(factor), peak_only=peak_only
            )
            for rep in range(n_rep):
                stats_df, queue_monitor_df = run_single_replication(
                    factor_df, random_seed=base_seed + rep
                )
                metrics = compute_replication_metrics(stats_df, queue_monitor_df)
                metrics.update(
                    {
                        "scenario_name": scenario_name,
                        "lambda_factor": float(factor),
                        "rep": rep,
                        "peak_only": bool(peak_only),
                    }
                )
                rep_records.append(metrics)

    rep_df = pd.DataFrame(rep_records)
    summary_df = build_lambda_factor_summary(rep_df, scenario_order, lambda_factors)
    return rep_df, summary_df


def print_scenario_summary_table(summary_df: pd.DataFrame) -> None:
    """输出整洁的控制台表格，便于人工核对。"""

    if summary_df.empty:
        print("Q3 仿真无有效结果，跳过表格输出。")
        return

    display_df = summary_df.copy()
    format_cols = {
        "avg_total_time_mean": "{:.2f}",
        "avg_total_time_std": "{:.2f}",
        "total_time_p50_mean": "{:.2f}",
        "total_time_p50_std": "{:.2f}",
        "total_time_p90_mean": "{:.2f}",
        "total_time_p90_std": "{:.2f}",
        "avg_wait_time_mean": "{:.2f}",
        "avg_wait_time_std": "{:.2f}",
        "wait_time_p50_mean": "{:.2f}",
        "wait_time_p50_std": "{:.2f}",
        "wait_time_p90_mean": "{:.2f}",
        "wait_time_p90_std": "{:.2f}",
        "SL10_mean": "{:.1%}",
        "SL10_std": "{:.1%}",
        "max_queue_mean": "{:.1f}",
        "max_queue_std": "{:.1f}",
        "avg_queue_mean": "{:.1f}",
        "avg_queue_std": "{:.1f}",
    }

    for col, fmt in format_cols.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: fmt.format(x) if pd.notna(x) else "NaN"
            )

    print("\n=== Q3 多方案仿真汇总 ===")
    print(display_df.to_string(index=False))


def print_lambda_robustness_conclusion(
    summary_df: pd.DataFrame, lambda_factors: Sequence[float]
) -> None:
    """输出中文小结，说明不同方案在大流量下的表现差异。"""

    if summary_df.empty:
        print("λ 放大鲁棒性结果为空，无法输出结论。")
        return

    max_factor = max(lambda_factors)
    print("\n=== λ 放大鲁棒性小结 ===")
    scenario_stats = []
    for scenario_name, group_df in summary_df.groupby("scenario_name", sort=False):
        group_sorted = group_df.sort_values("lambda_factor")
        base_row = group_sorted[np.isclose(group_sorted["lambda_factor"], 1.0)]
        peak_row = group_sorted[np.isclose(group_sorted["lambda_factor"], max_factor)]
        if base_row.empty or peak_row.empty:
            continue
        base_row = base_row.iloc[0]
        peak_row = peak_row.iloc[0]
        delta_total = peak_row["avg_total_time_mean"] - base_row["avg_total_time_mean"]
        delta_sl10 = peak_row["SL10_mean"] - base_row["SL10_mean"]
        scenario_stats.append(
            {
                "scenario_name": scenario_name,
                "delta_sl10": delta_sl10,
                "delta_total": delta_total,
            }
        )
        print(
            f"{scenario_name}: λ={max_factor:.1f} 时平均总耗时 {peak_row['avg_total_time_mean']:.2f} 分钟"
            f"（变化 {delta_total:+.2f}），SL10={peak_row['SL10_mean']:.1%}"
            f"（变化 {delta_sl10:+.1%}）。"
        )

    if len(scenario_stats) >= 2:
        best = max(scenario_stats, key=lambda x: x["delta_sl10"])
        worst = min(scenario_stats, key=lambda x: x["delta_sl10"])
        print(
            f"对比 λ={max_factor:.1f}: {best['scenario_name']} 的 SL10 变化 {best['delta_sl10']:+.1%}，"
            f"{worst['scenario_name']} 变化 {worst['delta_sl10']:+.1%}，说明前者在高峰下更稳健。"
        )


def plot_param_sensitivity_lines(
    df: pd.DataFrame,
    param_type: str,
    x_col: str,
    x_label: str,
    output_path: Path,
    title: str,
) -> None:
    """绘制单方案参数敏感性曲线，一图展示平均总耗时与 SL10。"""

    subset = df[df["param_type"] == param_type].copy()
    if subset.empty or x_col not in subset.columns:
        return

    subset = subset.dropna(subset=[x_col]).copy()
    if subset.empty:
        return

    subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
    subset.sort_values(x_col, inplace=True)
    if subset[x_col].isna().all():
        return

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    x_values = subset[x_col].astype(float)
    total_time = subset["avg_total_time_mean"].astype(float)
    sl10_percent = subset["SL10_mean"].astype(float) * 100.0

    ax1.plot(x_values, total_time, marker="o", color="#1b9e77", label="平均总耗时")
    ax1.set_ylabel("平均总耗时 (分钟)")
    ax2.plot(x_values, sl10_percent, marker="s", color="#d95f02", label="SL10")
    ax2.set_ylabel("SL10 (%)")
    ax1.set_xlabel(x_label)
    ax1.set_title(title)
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"已生成敏感性图：{output_path}")


def compute_service_level_metrics() -> None:
    """根据 Q1 输出估算各时段 T_total 与 SL10。"""

    missing_files = [
        path.name
        for path in (Q1_MMC_METRICS_CSV, Q1_PARAMS_FROM_EXCEL_CSV)
        if not path.exists()
    ]
    if missing_files:
        print(f"错误：{MISSING_Q1_DATA_MESSAGE}")
        return

    try:
        metrics_df = pd.read_csv(Q1_MMC_METRICS_CSV)
        params_df = pd.read_csv(Q1_PARAMS_FROM_EXCEL_CSV)
    except Exception as exc:  # pragma: no cover - 只在文件损坏时触发
        print(f"错误：读取 Q1 CSV 失败：{exc}")
        return

    params_columns = ["time_slot", "N_pick", "T_find_mean"]
    metrics_columns = ["time_slot", "lambda", "c", "mu_data", "Pw_data", "Wq_data_min"]
    if not set(metrics_columns).issubset(metrics_df.columns) or not set(
        params_columns
    ).issubset(params_df.columns):
        print(f"错误：{MISSING_Q1_DATA_MESSAGE}")
        return

    params_view = params_df.loc[:, params_columns].copy()
    metrics_view = metrics_df.loc[:, metrics_columns].copy()
    merged_df = params_view.merge(metrics_view, on="time_slot", how="inner")
    if merged_df.empty:
        print("错误：按 time_slot 合并后没有记录，请检查输入文件。")
        return

    numeric_columns = [
        "N_pick",
        "lambda",
        "c",
        "mu_data",
        "Pw_data",
        "Wq_data_min",
        "T_find_mean",
    ]
    for col in numeric_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    merged_df["T_service_min"] = np.where(
        merged_df["mu_data"] > 0, 60.0 / merged_df["mu_data"], np.nan
    )
    merged_df["T_total_mean"] = (
        merged_df["T_find_mean"] + merged_df["Wq_data_min"] + merged_df["T_service_min"]
    )

    t0 = 10.0 - merged_df["T_find_mean"] - merged_df["T_service_min"]
    rate_wait = (merged_df["c"] * merged_df["mu_data"] - merged_df["lambda"]) / 60.0

    sl10 = pd.Series(0.0, index=merged_df.index, dtype=float)
    feasible = (
        (t0 > 0)
        & (rate_wait > 0)
        & (~merged_df["Pw_data"].isna())
        & (~t0.isna())
        & (~rate_wait.isna())
    )
    if feasible.any():
        decay = np.exp(-rate_wait.loc[feasible] * t0.loc[feasible])
        sl10.loc[feasible] = (
            (1.0 - merged_df.loc[feasible, "Pw_data"])
            + merged_df.loc[feasible, "Pw_data"] * (1.0 - decay)
        )
    merged_df["SL10_i"] = sl10.clip(0.0, 1.0)

    weights = merged_df["N_pick"]
    t_total_day = weighted_average(merged_df["T_total_mean"], weights)
    sl10_day = weighted_average(merged_df["SL10_i"], weights)

    output_df = merged_df.copy()
    output_df["T_total_mean_day"] = np.nan
    output_df["SL10_day"] = np.nan

    summary_row = {col: np.nan for col in output_df.columns}
    summary_row["time_slot"] = "整日加权"
    summary_row["N_pick"] = weights.sum()
    summary_row["T_total_mean_day"] = t_total_day
    summary_row["SL10_day"] = sl10_day

    output_df = pd.concat([output_df, pd.DataFrame([summary_row])], ignore_index=True)

    output_df.to_csv(Q1_SERVICE_LEVEL_CSV, index=False)
    output_df.to_excel(Q1_SERVICE_LEVEL_XLSX, index=False)

    print("\n=== Q1 分时段平均总耗时与 SL10 ===")
    for row in merged_df.itertuples(index=False):
        t_total_text = (
            f"{row.T_total_mean:.2f}" if pd.notna(row.T_total_mean) else "NaN"
        )
        sl10_text = f"{row.SL10_i:.1%}" if pd.notna(row.SL10_i) else "NaN"
        print(f"{row.time_slot}: T_total_mean={t_total_text} 分钟, SL10={sl10_text}")

    def fmt_day_value(value: float) -> str:
        return f"{value:.2f}" if pd.notna(value) else "NaN"

    def fmt_day_ratio(value: float) -> str:
        return f"{value:.1%}" if pd.notna(value) else "NaN"

    print("\n整日加权平均：")
    print(f"T_total_mean_day = {fmt_day_value(t_total_day)} 分钟")
    print(f"SL10_day = {fmt_day_ratio(sl10_day)}")
    print("SL10_day 表示全日范围内，总耗时不超过 10 分钟的学生比例。")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    if sns is not None:
        sns.set_theme(style="whitegrid")

    configure_matplotlib_fonts()
    if not EXCEL_PATH.exists():
        print("错误：未在当前目录找到 '附件一.xlsx'，请确认文件已放置。")
        return

    xls = pd.ExcelFile(EXCEL_PATH)
    sheet_frames: Dict[str, pd.DataFrame] = {
        sheet_name: xls.parse(sheet_name, header=None) for sheet_name in xls.sheet_names
    }

    q1_params_df: Optional[pd.DataFrame] = None
    q1_metrics_df: Optional[pd.DataFrame] = None
    try:
        q1_params_df = build_q1_params_from_excel(
            excel_path=EXCEL_PATH, sheet_frames=sheet_frames
        )
    except Exception as exc:
        print(f"Q1 Excel 数据预处理失败：{exc}")

    if q1_params_df is not None:
        try:
            q1_metrics_df = build_q1_mmc_metrics(q1_params_df)
        except Exception as exc:
            print(f"Q1 M/M/c 指标计算失败：{exc}")

    if Q1_MMC_METRICS_CSV.exists() and Q1_PARAMS_FROM_EXCEL_CSV.exists():
        compute_service_level_metrics()
    else:
        print(f"提示：{MISSING_Q1_DATA_MESSAGE}")

    if Q1_PARAMS_FROM_EXCEL_CSV.exists():
        try:
            q2_base_df = load_q1_params_for_q2()
            run_q2_scenario_examples(q2_base_df)
            run_single_solution_evaluator_examples(q2_base_df)
            analyze_q2_pareto_fronts(q2_base_df)
        except Exception as exc:
            print(f"Q2 场景参数生成失败：{exc}")
    else:
        print(f"提示：{Q1_PARAMS_FROM_EXCEL_CSV} 不存在，暂无法生成 Q2 场景。")

    try:
        run_q2_visualization_and_recommendations()
    except Exception as exc:
        print(f"Q2 可视化及推荐输出失败：{exc}")

    try:
        # Q3：批量运行多方案多复制仿真
        rep_df, summary_df = run_replications_for_scenarios(
            SCENARIO_FILES, n_rep=N_REP, base_seed=DEFAULT_BASE_SEED
        )
        rep_df.to_csv(Q3_REPLICATIONS_RAW_CSV, index=False)
        summary_df.to_csv(Q3_SCENARIO_SUMMARY_CSV, index=False)
        print(
            f"Q3 仿真结果已写入 {Q3_REPLICATIONS_RAW_CSV} 与 {Q3_SCENARIO_SUMMARY_CSV}"
        )
        print_scenario_summary_table(summary_df)
        plot_metric_with_error(
            summary_df,
            "avg_total_time_mean",
            "avg_total_time_std",
            "平均总耗时 (分钟)",
            "Q3 方案平均总耗时对比",
            TOTAL_TIME_PLOT,
        )
        plot_metric_with_error(
            summary_df,
            "SL10_mean",
            "SL10_std",
            "SL10 (<=10分钟比例)",
            "Q3 方案 SL10 对比",
            SL10_PLOT,
        )
    except FileNotFoundError as exc:
        print(f"Q3 多方案仿真所需文件缺失：{exc}")
    except Exception as exc:
        print(f"Q3 多方案仿真失败：{exc}")

    try:
        mode_text = "仅高峰乘以因子" if ROBUSTNESS_PEAK_ONLY else "全时段乘以因子"
        robustness_rep_df, robustness_summary_df = run_lambda_robustness_analysis(
            SCENARIO_FILES,
            lambda_factors=LAMBDA_FACTORS,
            n_rep=ROBUSTNESS_N_REP,
            base_seed=DEFAULT_BASE_SEED,
            peak_only=ROBUSTNESS_PEAK_ONLY,
        )
        robustness_rep_df.to_csv(Q3_ROBUSTNESS_RAW_CSV, index=False)
        robustness_summary_df.to_csv(Q3_ROBUSTNESS_SUMMARY_CSV, index=False)
        print(
            f"Q3 λ 放大鲁棒性仿真（{mode_text}）结果已写入 "
            f"{Q3_ROBUSTNESS_RAW_CSV} 与 {Q3_ROBUSTNESS_SUMMARY_CSV}"
        )
        plot_lambda_factor_metric(
            robustness_summary_df,
            "avg_total_time_mean",
            "平均总耗时 (分钟)",
            "λ 放大因子 vs 平均总耗时",
            ROBUSTNESS_TOTAL_TIME_PLOT,
        )
        plot_lambda_factor_metric(
            robustness_summary_df,
            "SL10_mean",
            "SL10 (<=10分钟比例)",
            "λ 放大因子 vs SL10",
            ROBUSTNESS_SL10_PLOT,
        )
        print_lambda_robustness_conclusion(robustness_summary_df, LAMBDA_FACTORS)
    except FileNotFoundError as exc:
        print(f"Q3 λ 放大鲁棒性仿真所需文件缺失：{exc}")
    except Exception as exc:
        print(f"Q3 λ 放大鲁棒性仿真失败：{exc}")

    try:
        case_order = [case.get("case_name", f"stress_{idx + 1}") for idx, case in enumerate(HIGH_DEMAND_CASES)]
        stress_rep_df, stress_summary_df = run_high_demand_stress_tests(
            SCENARIO_FILES,
            stress_cases=HIGH_DEMAND_CASES,
            n_rep=HIGH_DEMAND_N_REP,
            base_seed=DEFAULT_BASE_SEED,
        )
        stress_rep_df.to_csv(Q3_HIGH_DEMAND_RAW_CSV, index=False)
        stress_summary_df.to_csv(Q3_HIGH_DEMAND_SUMMARY_CSV, index=False)
        print(
            "Q3 高负荷鲁棒性仿真结果已写入"
            f" {Q3_HIGH_DEMAND_RAW_CSV} 与 {Q3_HIGH_DEMAND_SUMMARY_CSV}"
        )
        plot_high_demand_metric(
            stress_summary_df,
            value_col="avg_total_time_mean",
            ylabel="平均总耗时 (分钟)",
            title="高负荷场景 vs 平均总耗时",
            output_path=HIGH_DEMAND_TOTAL_TIME_PLOT,
            case_order=case_order,
        )
        plot_high_demand_metric(
            stress_summary_df,
            value_col="SL10_mean",
            ylabel="SL10 (<=10分钟比例)",
            title="高负荷场景 vs SL10",
            output_path=HIGH_DEMAND_SL10_PLOT,
            case_order=case_order,
        )
        plot_high_demand_metric(
            stress_summary_df,
            value_col="avg_queue_mean",
            ylabel="平均队列长度 (人)",
            title="高负荷场景 vs 平均队长",
            output_path=HIGH_DEMAND_QUEUE_PLOT,
            case_order=case_order,
        )
        print_high_demand_conclusion(stress_summary_df, case_order)
    except FileNotFoundError as exc:
        print(f"Q3 高负荷鲁棒性仿真所需文件缺失：{exc}")
    except Exception as exc:
        print(f"Q3 高负荷鲁棒性仿真失败：{exc}")

    try:
        run_single_solution_param_sensitivity()
    except Exception as exc:
        print(f"Q3 单方案敏感性分析失败：{exc}")

    try:
        run_budget_sensitivity_analysis()
    except Exception as exc:
        print(f"Q3 预算敏感性分析失败：{exc}")

    print("=== Excel 工作表概览 ===")
    print(f"检测到 {len(xls.sheet_names)} 个工作表：{', '.join(xls.sheet_names)}")

    for sheet_name in xls.sheet_names:
        df_preview = xls.parse(sheet_name)
        print(f"\n--- Sheet: {sheet_name} ---")
        print("列名：", list(df_preview.columns))
        print("前几行：")
        print(df_preview.head())

    selection = detect_regression_source(sheet_frames)
    if selection is None or selection["t_find_col"] is None:
        print("当前附件未包含找件时间相关列，无法自动进行 A3 回归，请手工在 Excel 中补充后再运行。")
        return
    if selection["occupancy_col"] is None:
        print("错误：未能定位“货架占用率/在架件数”相关列，请检查表头命名。")
        return

    print(
        f"\n已选定工作表：{selection['sheet']}（{selection['note']}）"
        f"\n使用列：T_find='{selection['t_find_col']}', "
        f"occupancy='{selection['occupancy_col']}', "
        f"concurrent='{selection['concurrent_col'] or '无'}'"
    )

    regression_df, _ = build_regression_dataset(selection["df"], selection)
    regression_df = maybe_apply_manual_concurrent(regression_df)

    print("\n=== 回归样本 ===")
    print(regression_df.head())
    print(f"样本量：{len(regression_df)} 条")

    results = fit_regression_model(regression_df)
    build_summary_table(results)
    plot_scatter_with_regression(regression_df, results)
    plot_concurrent_views(regression_df, results)
    plot_residual_diagnostics(regression_df, results)
    print_console_summary(regression_df, results)


if __name__ == "__main__":
    main()

