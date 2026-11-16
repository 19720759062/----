# -*- coding: utf-8 -*-
"""
generate_signals.py —— signals.csv 初版 + 调仓排名表初版 生成器（11/18 交付）

用法（在项目根目录下执行）：
    python src/factors/generate_signals.py

前提：数据负责人已交付 data/cleaned/stock_data.csv
     （字段：date, code, open, high, low, close, volume；前复权）

输出（写到 reports/）：
    signals.csv            date, code, score, selected      —— 时间轴要求的初版格式
    rebalance_ranking.csv  date, code, close, momentum, ma_bias, low_vol,
                           momentum_01, ma_bias_01, low_vol_01, score, rank, selected
                           —— 调仓排名表初版（答辩时解释"为什么买了这些股票"）

规则（与全组统一标准一致）：
    每月最后一个交易日收盘后计算综合得分，选 Top 5 等权（selected=1）；
    候选不足 5 只时按实际数量全选并打印警告；
    窗口不足（动量 60 日未满）的月份自动跳过。
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_calc import (  # noqa: E402
    calc_all_factors,
    calc_scores,
    month_end_dates,
    select_top_k,
    TOP_K,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT, "data", "cleaned", "stock_data.csv")
REPORTS = os.path.join(ROOT, "reports")

SIGNAL_COLS = ["date", "code", "score", "selected"]
RANK_COLS = [
    "date", "code", "close", "momentum", "ma_bias", "low_vol",
    "momentum_01", "ma_bias_01", "low_vol_01", "score", "rank", "selected",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到数据文件: {path}\n"
            "请先确认数据负责人已把 data/cleaned/stock_data.csv 推到仓库，"
            "并在本地执行 git pull 拉取最新数据。"
        )
    df = pd.read_csv(path)
    required = {"date", "code", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必需字段: {missing}，请与数据负责人核对字段规范。")
    return df


def main() -> None:
    df = load_data()
    print(f"读取数据: {DATA_PATH}")
    print(f"股票数 {df['code'].nunique()}，日期范围 "
          f"{pd.to_datetime(df['date']).min().date()} ~ "
          f"{pd.to_datetime(df['date']).max().date()}")

    score_df = calc_scores(calc_all_factors(df))
    me_dates = month_end_dates(df["date"])

    signal_rows, rank_rows = [], []
    skipped = []
    for d in me_dates:
        day = score_df[score_df["date"] == d].dropna(subset=["score"])
        if day.empty:
            skipped.append(d)
            continue
        picked = select_top_k(score_df, d, k=TOP_K)
        sel_codes = set(picked["code"])
        if len(sel_codes) < TOP_K:
            print(f"[警告] {d.date()} 有效候选仅 {len(sel_codes)} 只（< {TOP_K}），"
                  "按实际数量全选，需与回测负责人确认资金分配规则。")
        ranked = day.sort_values("score", ascending=False).reset_index(drop=True)
        for i, row in ranked.iterrows():
            is_sel = row["code"] in sel_codes
            signal_rows.append({
                "date": d, "code": row["code"],
                "score": round(row["score"], 6),
                "selected": 1 if is_sel else 0,
            })
            rank_rows.append({
                "date": d, "code": row["code"], "close": row["close"],
                "momentum": round(row["momentum"], 6),
                "ma_bias": round(row["ma_bias"], 6),
                "low_vol": round(row["low_vol"], 6),
                "momentum_01": round(row["momentum_01"], 6),
                "ma_bias_01": round(row["ma_bias_01"], 6),
                "low_vol_01": round(row["low_vol_01"], 6),
                "score": round(row["score"], 6),
                "rank": i + 1,
                "selected": 1 if is_sel else 0,
            })

    signals = pd.DataFrame(signal_rows)
    ranking = pd.DataFrame(rank_rows)

    # 输出前自检
    sel = signals[signals["selected"] == 1]
    assert sel["score"].notna().all(), "selected 行存在 NaN 得分"
    per_month = sel.groupby(sel["date"].dt.to_period("M")).size()
    bad = per_month[per_month > TOP_K]
    assert bad.empty, f"存在月份选中数超过 {TOP_K}: {bad.to_dict()}"

    os.makedirs(REPORTS, exist_ok=True)
    p1 = os.path.join(REPORTS, "signals.csv")
    p2 = os.path.join(REPORTS, "rebalance_ranking.csv")
    signals.to_csv(p1, index=False, encoding="utf-8-sig")
    ranking.to_csv(p2, index=False, encoding="utf-8-sig")

    print(f"\n信号日 {len(me_dates)} 个，其中 {len(skipped)} 个无有效得分已跳过"
          f"（动量窗口 60 日未满属正常）: "
          + (str([x.date().isoformat() for x in skipped]) if skipped else "无"))
    print(f"每次调仓选中数分布:\n{per_month.to_string()}")
    print(f"\n已生成:\n  {p1}\n  {p2}")
    print("下一步：按《小样本人工检查计划》步骤 2-3 抽 2 只股票用 Excel 手工核对，"
          "并填写因子小样本真实数据检查记录，然后 push 到仓库。")


if __name__ == "__main__":
    main()
