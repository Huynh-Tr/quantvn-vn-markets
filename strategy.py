"""
Chiến lược: EMA Cross (9/21) + RSI(14) lọc tín hiệu

Logic:
- Mua  (+1): EMA9 cắt lên EMA21 VÀ RSI < 70 (chưa vào vùng quá mua)
- Bán  (-1): EMA9 cắt xuống EMA21 VÀ RSI > 30 (chưa vào vùng quá bán)
- Giữ   (0): không có điều kiện crossover hoặc RSI lọc bỏ

Cách dùng:
  python strategy.py
"""

import os

import numpy as np
import pandas as pd


# ── Tải API key từ .env (không cần python-dotenv) ──────────────────────────
def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()

_API_KEY = os.getenv("QUANTVN_API_KEY", "")
if _API_KEY:
    from quantvn.vn.data.utils import client
    client(apikey=_API_KEY)


# ── Chỉ báo kỹ thuật ───────────────────────────────────────────────────────
def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (EMA) dùng hệ số span."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI theo phương pháp Wilder (EWM với com = period-1).
    Trả về giá trị trong [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Hàm chiến lược chính ───────────────────────────────────────────────────
def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng chiến lược EMA Cross + RSI vào DataFrame OHLCV.

    Parameters
    ----------
    df : DataFrame có ít nhất cột 'Close'

    Returns
    -------
    DataFrame gốc được bổ sung các cột:
      ema_9    : EMA chu kỳ 9
      ema_21   : EMA chu kỳ 21
      rsi_14   : RSI chu kỳ 14
      signal   : tín hiệu tại điểm crossover (1 / -1 / 0)
      position : vị thế duy trì đến tín hiệu tiếp theo (1 / -1 / 0)
    """
    df = df.copy()

    # Bước 1 — tính chỉ báo
    df["ema_9"] = _ema(df["Close"], span=9)
    df["ema_21"] = _ema(df["Close"], span=21)
    df["rsi_14"] = _rsi(df["Close"], period=14)

    # Bước 2 — phát hiện crossover
    ema_above = df["ema_9"] > df["ema_21"]
    cross_up   = ema_above  & (~ema_above.shift(1).fillna(False))   # EMA9 cắt lên
    cross_down = (~ema_above) & (ema_above.shift(1).fillna(False))  # EMA9 cắt xuống

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


# ── Chạy thử chiến lược ────────────────────────────────────────────────────
if __name__ == "__main__":
    from quantvn.vn.data import get_stock_hist

    SYMBOL     = "VIC"
    RESOLUTION = "1H"

    print(f"Lấy dữ liệu {SYMBOL} ({RESOLUTION})...")
    df_raw = get_stock_hist(SYMBOL, resolution=RESOLUTION)
    print(f"  → {df_raw.shape[0]} nến, cột: {list(df_raw.columns)}")

    result = gen_position(df_raw)

    buy_n  = (result["signal"] == 1).sum()
    sell_n = (result["signal"] == -1).sum()
    print(f"\nTín hiệu mua  (+1): {buy_n}")
    print(f"Tín hiệu bán  (-1): {sell_n}")
    print(f"Tổng giao dịch    : {buy_n + sell_n}")

    cols = ["Date", "time", "Close", "ema_9", "ema_21", "rsi_14", "signal", "position"]
    print("\n10 nến gần nhất:")
    print(result[cols].tail(10).to_string(index=False))
