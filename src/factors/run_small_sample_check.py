# -*- coding: utf-8 -*-
"""
run_small_sample_check.py —— 小样本人工检查脚本（11/16 交付：任务③的可执行部分）

作用：
    在真实数据（数据负责人 11/17 交付）到位之前，用 5 只"模式已知"的
    合成股票验证因子计算方向、打分与选股逻辑是否正确；
    真实数据到位后，本脚本可直接改为读取 data/cleaned/stock_data.csv
    的前几只股票，按《小样本人工检查计划.md》逐项人工核对。

运行：
    cd quant-trading-project
    python src/factors/run_small_sample_check.py

5 只合成股票的设计（便于人工判断"谁该排第一"）：
    SH600001  稳定上涨   → 动量最高、价格在均线上方、波动低
    SH600002  稳定下跌   → 动量最低、价格在均线下方
    SH600003  横盘低波动 → 波动最小、动量≈0
    SH600004  大幅震荡   → 波动最大、动量≈0
    SH600005  温和上涨   → 各因子介于 600001 与 600003 之间
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "factors"))
from factor_calc import (  # noqa: E402
    calc_all_factors,
    calc_scores,
    month_end_dates,
    select_top_k,
)

N_DAYS = 130  # > 60 + 20，保证动量窗口能取满


def make_synthetic_data() -> pd.DataFrame:
    """生成 5 只模式已知的合成股票日频数据（不复权不影响检查逻辑）。"""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2026-01-05", periods=N_DAYS)  # 工作日序列
    specs = {
        "SH600001": {"start": 10.0, "drift": 0.0020, "noise": 0.002},  # 稳定上涨
        "SH600002": {"start": 20.0, "drift": -0.0020, "noise": 0.002},  # 稳定下跌
        "SH600003": {"start": 15.0, "drift": 0.0000, "noise": 0.001},  # 横盘低波动
        "SH600004": {"start": 8.0, "drift": 0.0000, "noise": 0.020},   # 大幅震荡
        "SH600005": {"start": 12.0, "drift": 0.0010, "noise": 0.003},  # 温和上涨
    }
    rows = []
    for code, s in specs.items():
        rets = rng.normal(s["drift"], s["noise"], N_DAYS)
        close = s["start"] * np.cumprod(1 + rets)
        for i in range(N_DAYS):
            rows.append(
                {
                    "date": dates[i],
                    "code": code,
                    "open": close[i] * 0.998,
                    "high": close[i] * 1.01,
                    "low": close[i] * 0.99,
                    "close": close[i],
                    "volume": 1_000_000 + i,
                }
            )
    return pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)


def check_no_future_leakage(df: pd.DataFrame) -> bool:
    """未来数据泄露的程序化检查：

    把数据截断到某日 t，重算因子，t 日因子值必须与全样本计算完全一致
    （rolling/shift 只用过去数据时应严格成立）。
    """
    t = pd.bdate_range("2026-01-05", periods=N_DAYS)[N_DAYS - 40]
    full = calc_all_factors(df)
    trunc = calc_all_factors(df[df["date"] <= t])
    cols = ["momentum", "ma_bias", "low_vol"]
    left = full[full["date"] == t].set_index("code")[cols]
    right = trunc[trunc["date"] == t].set_index("code")[cols]
    return bool(np.allclose(left.values, right.values, equal_nan=True))


def main() -> None:
    df = make_synthetic_data()
    factor_df = calc_all_factors(df)
    score_df = calc_scores(factor_df)

    last_day = df["date"].max()
    final = score_df[score_df["date"] == last_day].set_index("code")

    pd.set_option("display.float_format", lambda v: f"{v: .4f}")
    print("=" * 78)
    print(f"小样本因子检查表（{last_day.date()}，共 {final.shape[0]} 只股票）")
    print("=" * 78)
    cols = [
        "close", "momentum", "ma_bias", "low_vol",
        "momentum_01", "ma_bias_01", "low_vol_01", "score",
    ]
    print(final[cols].sort_values("score", ascending=False))

    print("\n[检查1] 均线因子方向：上涨股应为正、下跌股应为负")
    print("  SH600001 ma_bias =", round(final.loc["SH600001", "ma_bias"], 4), "(预期 > 0)")
    print("  SH600002 ma_bias =", round(final.loc["SH600002", "ma_bias"], 4), "(预期 < 0)")

    print("\n[检查2] 动量因子方向：稳定上涨 > 温和上涨 > 横盘/震荡 > 稳定下跌")
    order = (
        final["momentum"].sort_values(ascending=False).index.tolist()
    )
    print("  动量降序:", " > ".join(order))

    print("\n[检查3] 低波动因子方向：横盘低波动股得分应最高、大幅震荡股最低")
    vol_order = final["low_vol"].sort_values(ascending=False).index.tolist()
    print("  低波动因子降序:", " > ".join(vol_order))

    print("\n[检查4] 窗口不足处理：前 60 个交易日 momentum 应为 NaN")
    first60 = factor_df[factor_df["date"] <= pd.bdate_range("2026-01-05", periods=59)[-1]]
    all_nan = first60["momentum"].isna().all()
    print("  前59个交易日 momentum 全为 NaN:", all_nan)

    print("\n[检查5] 未来数据泄露：截断重算因子值不变")
    print("  通过:", check_no_future_leakage(df))

    print("\n[检查6] 打分范围：所有 0-1 分应落在 [0,1]，综合得分 ∈ [0,1]")
    c01 = ["momentum_01", "ma_bias_01", "low_vol_01", "score"]
    in_range = final[c01].dropna().apply(lambda s: s.between(0, 1).all()).all()
    print("  通过:", bool(in_range))

    print("\n[检查7] 月末信号日识别（供 11/18 signals.csv 使用）")
    me_dates = month_end_dates(df["date"])
    print("  最后3个月末交易日:", [d.date().isoformat() for d in me_dates[-3:]])

    print("\n[检查8] 选股逻辑：综合得分 Top 5（等权）")
    print(select_top_k(score_df, last_day)[["date", "code", "score"]].to_string(index=False))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, "factor_check_small_sample.csv"))
    out_df = final.reset_index()[["date", "code"] + cols]
    num_cols = out_df.select_dtypes("number").columns
    out_df[num_cols] = out_df[num_cols].round(6)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n检查表已保存: {out_path}")


if __name__ == "__main__":
    main()
