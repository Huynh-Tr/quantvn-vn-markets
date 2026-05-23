"""
Chiến lược: EMA Cross (20/60) + RSI(14) lọc tín hiệu

Logic:
- Mua  (+1): EMA20 cắt lên EMA60 VÀ RSI < 70 (chưa vào vùng quá mua)
- Bán  (-1): EMA20 cắt xuống EMA60 VÀ RSI > 30 (chưa vào vùng quá bán)
- Giữ   (0): không có điều kiện crossover hoặc RSI lọc bỏ
"""

import os

import numpy as np
import pandas as pd

def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng chiến lược EMA Cross + RSI vào DataFrame OHLCV.

    Parameters
    ----------
    df : DataFrame có ít nhất cột 'Close'

    Returns
    -------
    DataFrame gốc được bổ sung các cột:
      ema_20    : EMA chu kỳ 20
      ema_60   : EMA chu kỳ 60
      rsi_14   : RSI chu kỳ 14
      signal   : tín hiệu tại điểm crossover (1 / -1 / 0)
      position : vị thế duy trì đến tín hiệu tiếp theo (1 / -1 / 0)
    """
    df = df.copy()

    # Bước 1 — tính chỉ báo
    df["ema_20"] = _ema(df["Close"], span=20)
    df["ema_60"] = _ema(df["Close"], span=60)
    df["rsi_14"] = _rsi(df["Close"], period=14)

    # Bước 2 — phát hiện crossover
    ema_above = df["ema_20"] > df["ema_60"]
    cross_up   = ema_above  & (~ema_above.shift(1).fillna(False))   # EMA20 cắt lên
    cross_down = (~ema_above) & (ema_above.shift(1).fillna(False))  # EMA20 cắt xuống

    # Bước 3 — tạo tín hiệu (RSI lọc bỏ các crossover ở vùng cực đoan)
    df["signal"] = 0
    df.loc[cross_up   & (df["rsi_14"] < 70), "signal"] = 1   # mua
    df.loc[cross_down & (df["rsi_14"] > 30), "signal"] = -1  # bán

    # Bước 4 — chuyển sang position (giữ trạng thái đến khi có tín hiệu mới)
    df["position"] = (
        df["signal"]
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
        .astype(int)
    )

    return df

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))