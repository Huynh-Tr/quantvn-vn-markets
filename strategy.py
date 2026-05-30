import numpy as np
import pandas as pd

def gen_position(df: pd.DataFrame,
                 n: int = 40,
                 ema_span: int = 200,
                 atr_period: int = 14,
                 atr_mult: float = 1.5,
                 rr: float = 2.0) -> pd.DataFrame:
    """Chiến lược Donchian Breakout + lọc EMA + SL/TP với R:R = 1:2.

    Ý tưởng
    -------
    - Breakout: vào lệnh khi giá phá kênh giá cao/ thấp nhất trong ``n`` bar gần nhất.
    - Lọc EMA: chỉ **Long khi giá > EMA**, chỉ **Short khi giá < EMA**
      => chỉ giao dịch thuận xu hướng, loại bỏ breakout ngược trend (giảm tín hiệu giả).
    - Quản trị rủi ro: mỗi lệnh đặt Stop-Loss và Take-Profit theo tỷ lệ R:R = 1:2,
      với khoảng rủi ro ``R = atr_mult * ATR`` đo tại thời điểm vào lệnh.
        * Long : SL = entry - R,  TP = entry + rr*R
        * Short: SL = entry + R,  TP = entry - rr*R
      Khi giá chạm SL hoặc TP thì đóng vị thế về 0 và chờ tín hiệu breakout hợp lệ kế tiếp.

    Parameters
    ----------
    df : pd.DataFrame
        Dữ liệu OHLCV, cần có các cột 'High', 'Low', 'Close'.
    n : int, optional
        Số bar của kênh Donchian (đỉnh/ đáy), mặc định 40.
    ema_span : int, optional
        Chu kỳ EMA dùng làm bộ lọc xu hướng, mặc định 200.
    atr_period : int, optional
        Chu kỳ ATR để đo rủi ro, mặc định 14.
    atr_mult : float, optional
        Hệ số nhân ATR để xác định khoảng rủi ro R, mặc định 1.5.
    rr : float, optional
        Tỷ lệ Reward:Risk (TP = rr * R), mặc định 2.0 (tức 1:2).

    Returns
    -------
    pd.DataFrame
        DataFrame gốc bổ sung các cột:
          - 'ema_trend', 'atr', 'don_upper', 'don_lower': chỉ báo trung gian
          - 'position'   : vị thế (1 long / -1 short / 0 đứng ngoài)
          - 'stop_loss'  : mức SL hiện hành của vị thế
          - 'take_profit': mức TP hiện hành của vị thế

    Notes
    -----
    - Kênh Donchian dùng ``shift(1)`` để tránh nhìn trộm (look-ahead) bar hiện tại.
    - Nếu trong cùng một bar cả SL và TP đều bị chạm, giả định **kịch bản xấu nhất**
      (chạm SL trước) để không đánh giá lạc quan quá mức.
    """
    # --- Chỉ báo ---
    df['ema_trend'] = ema(df['Close'], ema_span)          # EMA lọc xu hướng
    df['atr'] = atr(df, atr_period)                        # độ biến động -> đo R
    df['don_upper'] = df['High'].rolling(n).max().shift(1) # đỉnh N bar (loại bar hiện tại)
    df['don_lower'] = df['Low'].rolling(n).min().shift(1)  # đáy N bar (loại bar hiện tại)

    # Chuyển sang numpy để vòng lặp nhanh và rõ ràng hơn
    close = df['Close'].to_numpy()
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    ema_t = df['ema_trend'].to_numpy()
    atr_v = df['atr'].to_numpy()
    up = df['don_upper'].to_numpy()
    lo = df['don_lower'].to_numpy()

    m = len(df)
    position = np.zeros(m, dtype=int)   # vị thế ghi lại theo từng bar
    sl_arr = np.full(m, np.nan)          # mức SL theo từng bar
    tp_arr = np.full(m, np.nan)          # mức TP theo từng bar

    pos = 0                  # vị thế hiện hành: 1 / -1 / 0
    entry = sl = tp = np.nan # giá vào lệnh, mức SL, mức TP của vị thế đang mở

    for i in range(m):
        # --- 1) Nếu đang giữ lệnh: kiểm tra SL/TP trước ---
        if pos == 1:                      # đang Long
            hit_sl = low[i] <= sl         # giá thấp nhất chạm SL
            hit_tp = high[i] >= tp        # giá cao nhất chạm TP
            if hit_sl or hit_tp:          # xấu nhất: ưu tiên SL nếu cả hai cùng chạm
                pos = 0
                entry = sl = tp = np.nan
        elif pos == -1:                   # đang Short
            hit_sl = high[i] >= sl        # giá cao nhất chạm SL
            hit_tp = low[i] <= tp         # giá thấp nhất chạm TP
            if hit_sl or hit_tp:
                pos = 0
                entry = sl = tp = np.nan

        # --- 2) Nếu đang đứng ngoài: tìm tín hiệu breakout đã lọc EMA ---
        if pos == 0 and not np.isnan(up[i]) and not np.isnan(ema_t[i]) and not np.isnan(atr_v[i]):
            risk = atr_mult * atr_v[i]    # khoảng rủi ro R
            if risk > 0:
                # Long: phá đỉnh kênh VÀ giá trên EMA
                long_sig = (close[i] > up[i]) and (close[i] > ema_t[i])
                # Short: thủng đáy kênh VÀ giá dưới EMA
                short_sig = (close[i] < lo[i]) and (close[i] < ema_t[i])
                if long_sig:
                    pos = 1
                    entry = close[i]
                    sl = entry - risk         # SL cách entry 1R
                    tp = entry + rr * risk    # TP cách entry rr*R (mặc định 2R)
                elif short_sig:
                    pos = -1
                    entry = close[i]
                    sl = entry + risk
                    tp = entry - rr * risk

        # Ghi lại trạng thái của bar i
        position[i] = pos
        sl_arr[i] = sl
        tp_arr[i] = tp

    df['position'] = position
    df['stop_loss'] = sl_arr
    df['take_profit'] = tp_arr
    return df

def ema(s: pd.Series, span: int) -> pd.Series:
    """Tính Exponential Moving Average (EMA) của một chuỗi giá.

    EMA đặt trọng số lớn hơn cho dữ liệu gần đây nên phản ứng nhanh hơn SMA,
    dùng làm bộ lọc xu hướng (giá nằm trên/ dưới EMA => xu hướng tăng/ giảm).

    Parameters
    ----------
    s : pd.Series
        Chuỗi giá (thường là giá đóng cửa 'Close').
    span : int
        Chu kỳ EMA (số bar).

    Returns
    -------
    pd.Series
        Giá trị EMA theo từng bar.
    """
    return s.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Tính Average True Range (ATR) — thước đo độ biến động.

    ATR là trung bình động của True Range (TR), với
    TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|).
    Dùng để định lượng rủi ro (R) khi đặt stop-loss/ take-profit.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame có các cột 'High', 'Low', 'Close'.
    period : int, optional
        Chu kỳ làm mượt ATR (mặc định 14).

    Returns
    -------
    pd.Series
        Giá trị ATR theo từng bar.
    """
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    # True Range = biên độ lớn nhất trong 3 cách đo
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Làm mượt TR theo kiểu Wilder (alpha = 1/period)
    return tr.ewm(alpha=1 / period, adjust=False).mean()
