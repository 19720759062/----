# -*- coding: utf-8 -*-
"""
factor_calc.py —— 三因子计算与标准化打分（策略与信号负责人模块）

统一标准（全组必须遵守，见 README 第 2 节）：
    动量因子   过去 60 个交易日收益率，越高越好
    均线因子   当前价 / 20日均线 - 1，越高越好
    低波动因子 -（过去 20 日收益率标准差），波动越低得分越高
    打分方法   横截面排名法转 0-1 分；综合得分 = 50%动量 + 30%均线 + 20%低波动
    调仓       每月最后一个交易日计算信号，下月第一个交易日成交

数据输入约定（与数据负责人 11/17 交付的 data/cleaned/stock_data.csv 一致）：
    长表格式，每行一个 (date, code)：
    date, code, open, high, low, close, volume
    其中 date 为升序排列的交易日，价格为前复权价。

防未来数据泄露说明：
    本模块所有因子均只用"当日及以前"的数据（rolling/shift 均向后看），
    横截面排名也只使用同一交易日当天的因子值。
    run_small_sample_check.py 中包含"截断数据后因子值不变"的程序化验证。
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 参数集中定义（后续参数敏感性分析只改这里）
# ---------------------------------------------------------------------------
MOMENTUM_WINDOW = 60      # 动量因子回看窗口（交易日）
MA_WINDOW = 20            # 均线因子均线窗口（交易日）
VOL_WINDOW = 20           # 低波动因子波动窗口（交易日）
WEIGHT_MOMENTUM = 0.50    # 动量权重
WEIGHT_MA = 0.30          # 均线权重
WEIGHT_VOL = 0.20         # 低波动权重
TOP_K = 5                 # 每期选股数量（等权）

FACTOR_COLUMNS = ["momentum", "ma_bias", "low_vol"]


# ---------------------------------------------------------------------------
# 单因子计算（输入：单只股票按日期升序的收盘价 Series；输出：同期因子 Series）
# ---------------------------------------------------------------------------
def calc_momentum(close: pd.Series) -> pd.Series:
    """动量因子：过去 60 个交易日收益率 = close[t] / close[t-60] - 1。

    前 60 个交易日窗口不足，返回 NaN（不参与当期打分，
    相当于自动排除上市/数据时间过短的股票）。
    """
    return close / close.shift(MOMENTUM_WINDOW) - 1.0


def calc_ma_bias(close: pd.Series) -> pd.Series:
    """均线因子：当前价 / 20日均线 - 1（价格偏离均线的程度）。

    价格高于 20 日均线 → 值为正 → 趋势较强；前 20 日窗口不足返回 NaN。
    """
    ma = close.rolling(MA_WINDOW).mean()
    return close / ma - 1.0


def calc_low_vol(close: pd.Series) -> pd.Series:
    """低波动因子：-（过去 20 日日收益率的标准差）。

    波动越小 → 值越大（越接近 0）→ 得分越高；
    统一"越大越好"方向后三个因子可直接加权。前 21 日窗口不足返回 NaN。
    """
    daily_ret = close.pct_change()
    return -daily_ret.rolling(VOL_WINDOW).std()


# ---------------------------------------------------------------------------
# 横截面标准化：排名法转 0-1
# ---------------------------------------------------------------------------
def rank_normalize(scores: pd.Series) -> pd.Series:
    """把当日横截面因子值按排名映射到 [0, 1]。

    规则：
      1. 只对当日非 NaN 的股票排名（窗口不足的股票不进入排名）；
      2. score_01 = (rank - 1) / (n - 1)，n 为当日有效股票数；
      3. n == 1 时记 0.5（中性分，避免除零，并在文档中说明）；
      4. 并列名次取平均名次（method="average"）。

    为什么用排名法而不是 z-score：
      因子分布有厚尾、量纲不同，排名法对极端值稳健，
      且天然落在 [0,1]，三个因子加权后量纲可比。
    """
    valid = scores.dropna()
    n = len(valid)
    if n == 0:
        return pd.Series(np.nan, index=scores.index)
    if n == 1:
        return scores.where(scores.isna(), 0.5)
    ranks = scores.rank(method="average")
    return (ranks - 1.0) / (n - 1.0)


# ---------------------------------------------------------------------------
# 批量计算：长表 → 因子表
# ---------------------------------------------------------------------------
def calc_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """对长表数据计算全部三因子原始值。

    参数
    ----
    df : 包含 date, code, close（可含 open/high/low/volume）的 DataFrame。

    返回
    ----
    在原表基础上追加 momentum / ma_bias / low_vol 三列原始因子值，
    按 (date, code) 排序。同日停牌缺数据的股票当日自然无因子值。
    """
    required = {"date", "code", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必需字段: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["code", "date"]).reset_index(drop=True)

    # 逐股票计算：rolling/shift 都只向后看，不引入未来数据
    grouped = out.groupby("code", sort=False)["close"]
    out["momentum"] = grouped.transform(calc_momentum)
    out["ma_bias"] = grouped.transform(calc_ma_bias)
    out["low_vol"] = grouped.transform(calc_low_vol)
    return out


def calc_scores(factor_df: pd.DataFrame) -> pd.DataFrame:
    """在因子表基础上做横截面排名打分并计算综合得分。

    综合得分 = 0.5*momentum_01 + 0.3*ma_bias_01 + 0.2*low_vol_01
    三个因子任一为 NaN，则综合得分 NaN（该股票当期不参与选股）。
    """
    out = factor_df.copy()
    for col, score_col in zip(FACTOR_COLUMNS, [f"{c}_01" for c in FACTOR_COLUMNS]):
        # 每个交易日做一次横截面排名
        out[score_col] = out.groupby("date")[col].transform(rank_normalize)
    out["score"] = (
        WEIGHT_MOMENTUM * out["momentum_01"]
        + WEIGHT_MA * out["ma_bias_01"]
        + WEIGHT_VOL * out["low_vol_01"]
    )
    return out


# ---------------------------------------------------------------------------
# 调仓信号日：每月最后一个交易日
# ---------------------------------------------------------------------------
def month_end_dates(dates: pd.Series) -> list:
    """返回"每月最后一个交易日"列表（升序去重）。

    用途：在这些日子收盘后计算综合得分、生成下一交易日成交的信号。
    """
    d = pd.to_datetime(pd.Series(sorted(set(dates))))
    grouped = d.groupby(d.dt.to_period("M"))
    return [g.iloc[-1] for _, g in grouped if len(g) > 0]


def select_top_k(score_df: pd.DataFrame, date, k: int = TOP_K) -> pd.DataFrame:
    """取某信号日综合得分最高的 k 只股票（等权候选池）。

    返回按得分降序的 date/code/各因子/score 明细，
    供调仓排名表（解释"为什么买了这些股票"）使用。
    """
    day = pd.to_datetime(date)
    day_scores = score_df[score_df["date"] == day].dropna(subset=["score"])
    picked = day_scores.sort_values("score", ascending=False).head(k)
    return picked.reset_index(drop=True)
