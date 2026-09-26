"""特徵、triple-barrier 標籤、規則排序、ML 分類器。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import config as C

PRICE_FEATURES = ["ret_5", "ret_10", "ret_20", "ret_60", "exc_20", "exc_60",
                  "ma20_gap", "ma60_gap", "ma20_slope", "vol_20d", "vol_ratio",
                  "high60_gap", "regime"]

# 籌碼面。買賣超一律除以同期成交量 -> 「法人吃掉幾 % 的成交量」，
# 否則台積電的絕對股數永遠輾壓小型股，橫斷面比較沒有意義。
CHIP_FEATURES = ["fgn_5", "fgn_20", "tru_5", "tru_20", "tru_streak",
                 "margin_chg_20", "short_ratio"]

FEATURES = PRICE_FEATURES + CHIP_FEATURES


def build_features(df, bench):
    """df: 長表 [symbol,date,ohlcv]；bench: 大盤 close 的 Series(index=date)。

    全部特徵只用到當日（含）以前的資料，沒有未來資訊。
    """
    bench_ret = {n: (bench / bench.shift(n) - 1).rename(f"b{n}") for n in (20, 60)}
    bench_regime = (bench > bench.rolling(60).mean()).astype(float).rename("regime")

    out = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        c, v = g.close, g.volume
        for n in (5, 10, 20, 60):
            g[f"ret_{n}"] = c / c.shift(n) - 1
        ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
        g["ma20_gap"] = c / ma20 - 1
        g["ma60_gap"] = c / ma60 - 1
        g["ma20_slope"] = ma20 / ma20.shift(5) - 1
        g["vol_20d"] = c.pct_change().rolling(20).std()
        g["vol_ratio"] = v.rolling(5).mean() / v.rolling(60).mean()
        g["high60_gap"] = c / c.rolling(60).max() - 1
        g["turnover20"] = (c * v).rolling(20).mean()
        # 顯示用，不是特徵：這檔開盤會落在什麼價位區間。
        # 用 p10/p90 的實際分位數而非平均絕對值 —— 平均值沒辦法換算成價位。
        gap = g.open / c.shift(1) - 1
        g["gap_lo"] = gap.rolling(250, min_periods=60).quantile(0.10)
        g["gap_hi"] = gap.rolling(250, min_periods=60).quantile(0.90)
        out.append(g)

    f = pd.concat(out, ignore_index=True)
    idx = f.date.map
    for n in (20, 60):
        f[f"exc_{n}"] = f[f"ret_{n}"] - idx(bench_ret[n])
    f["regime"] = idx(bench_regime)
    return f


def add_chips(df, chips):
    """把籌碼面併進特徵表。chips 缺的日子留 NaN，HistGradientBoosting 吃得下。"""
    if chips is None or chips.empty:
        for c in CHIP_FEATURES:
            df[c] = np.nan
        return df

    d = df.merge(chips, on=["symbol", "date"], how="left")
    out = []
    for _, g in d.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        vol = g.volume.rolling(20).sum()
        for tag, col in (("fgn", "foreign_net"), ("tru", "trust_net")):
            g[f"{tag}_5"] = g[col].rolling(5).sum() / g.volume.rolling(5).sum()
            g[f"{tag}_20"] = g[col].rolling(20).sum() / vol
        pos = (g.trust_net > 0).astype(int)          # 投信連續買超天數
        g["tru_streak"] = pos.groupby((pos == 0).cumsum()).cumsum()
        g["margin_chg_20"] = g.margin_bal / g.margin_bal.shift(20) - 1
        g["short_ratio"] = g.short_bal / g.margin_bal.replace(0, np.nan)
        out.append(g)
    d = pd.concat(out, ignore_index=True)
    return d.replace([np.inf, -np.inf], np.nan)


def settle_one(entry, high, low, close, horizon, up_r=None, dn_r=None, etf=False):
    """單一部位的 triple-barrier 結算，high/low/close 是進場日起算的未來 K 線。

    up_r / dn_r 必須顯式傳入訊號「產生當時」的門檻，不能讀當下的環境變數 ——
    並行追蹤多個版本時，用 A 版的設定去結算 B 版的訊號會全錯而且不會報錯。
    未給則退回目前設定（單一版本時的方便用法）。

    回傳 (outcome, net, days)，outcome 為 win / loss / expire / open。
    與 label() 必須給出相同判定（test_run.py 有斷言守著），差別只在
    這支能處理「還沒到期」的部位，供 verify 用。
    """
    up_r = C.raw_target(etf=etf) if up_r is None else up_r
    dn_r = C.STOP_LOSS if dn_r is None else dn_r
    up, dn = entry * (1 + up_r), entry * (1 - dn_r)
    n = len(close)
    for i in range(min(n, horizon)):
        if low[i] <= dn:                       # 同根同時觸及 -> 保守判停損
            return "loss", C.net_return(-dn_r, etf=etf), i + 1
        if high[i] >= up:
            return "win", C.net_return(up_r, etf=etf), i + 1
    if n >= horizon:
        return "expire", C.net_return(close[horizon - 1] / entry - 1, etf=etf), horizon
    return "open", np.nan, n


def entry_price(nxt_open, nxt_low, prev_close, mode="open", cap=None):
    """決定進場價，回傳 None 代表沒成交。

    open   次日開盤市價。回測基準 —— 無論開在哪都買。
    cap    同上，但開盤漲幅超過 cap 就放棄（測「跳空追高是否該濾掉」）。
    limit  掛前一日收盤。已經開在限價之下就用開盤成交（買得更便宜，不是限價），
           否則要盤中低點觸及限價才成交（測「等好價格」這個直覺）。
    """
    if mode == "open":
        return nxt_open
    if mode == "cap":
        if not prev_close > 0:
            return None
        return nxt_open if nxt_open / prev_close - 1 <= cap else None
    if mode == "limit":
        if nxt_open <= prev_close:
            return nxt_open
        return prev_close if nxt_low <= prev_close else None
    raise ValueError(f"未知的進場模式 {mode!r}")


def label(df, horizon, net=None, entry="open", cap=None, stop=None):
    """Triple barrier。訊號在 t 日收盤產生，隔日進場（無前視偏誤）。

    y=1 先觸停利、y=0 先觸停損或到期平倉。同根 K 同時觸及 -> 判停損（保守）。
    未成交的訊號 y=NaN，會被下游的 dropna 自然排除。
    net 可覆寫停利目標，讓單一 process 內能掃不同目標。
    回傳增加 y / net_ret / hold_days / stopped 四欄。
    """
    out = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        etf = C.is_etf(sym)
        up_r = C.raw_target(net, etf=etf)
        dn_r = C.STOP_LOSS if stop is None else stop
        o, hi, lo, cl = (g[k].values for k in ("open", "high", "low", "close"))
        n = len(g)
        y = np.full(n, np.nan)
        ret = np.full(n, np.nan)
        days = np.full(n, np.nan)
        stopped = np.full(n, np.nan)   # 1=觸停損出場，用來分辨「沒賺到」是停損還是到期

        # ponytail: 每檔一個 python 迴圈（~1250 圈 x 300 檔）。先觸哪個 barrier 難以
        # 純向量化；實測數秒內完成。若股票池擴到數千檔再考慮 numba。
        for i in range(n - horizon - 1):
            px = entry_price(o[i + 1], lo[i + 1], cl[i], entry, cap)
            if px is None or not px > 0:
                continue
            up, dn = px * (1 + up_r), px * (1 - dn_r)
            w_hi, w_lo = hi[i + 1:i + 1 + horizon], lo[i + 1:i + 1 + horizon]
            t_up = np.argmax(w_hi >= up) if (w_hi >= up).any() else n
            t_dn = np.argmax(w_lo <= dn) if (w_lo <= dn).any() else n
            if t_dn <= t_up and t_dn < n:          # 停損先到（或同日，保守）
                y[i], raw, days[i], stopped[i] = 0, -dn_r, t_dn + 1, 1
            elif t_up < n:                          # 停利達陣
                y[i], raw, days[i], stopped[i] = 1, up_r, t_up + 1, 0
            else:                                   # 到期平倉
                y[i], raw, days[i], stopped[i] = 0, cl[i + horizon] / px - 1, horizon, 0
            ret[i] = C.net_return(raw, etf=etf)

        g["y"], g["net_ret"], g["hold_days"], g["stopped"] = y, ret, days, stopped
        out.append(g)
    return pd.concat(out, ignore_index=True)


def rule_score(df):
    """引擎 A：規則排序基準線。每日橫斷面百分位加權，無任何參數擬合。"""
    g = df.groupby("date")
    pct = lambda c: g[c].rank(pct=True)
    return (0.5 * pct("exc_60") + 0.3 * pct("exc_20")
            + 0.1 * (df.ma20_gap > 0).astype(float) + 0.1 * pct("vol_ratio"))


def fit_predict(train, test, feats=None):
    """引擎 B：ML。回傳 test 的停利機率。feats 未給則用全部特徵。"""
    feats = FEATURES if feats is None else feats
    tr = train.dropna(subset=["y"])
    if tr.y.nunique() < 2 or len(tr) < 500:
        return np.full(len(test), np.nan)
    m = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4,
        min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    m.fit(tr[feats], tr.y.astype(int))
    return m.predict_proba(test[feats])[:, 1]


def vol_bucket(df, n=10):
    """每日橫斷面的波動率分層（0=最低波動）。用 rank 再切，避免重複邊界。"""
    return df.groupby("date").vol_20d.transform(
        lambda s: pd.qcut(s.rank(method="first"), min(n, max(1, s.notna().sum())),
                          labels=False, duplicates="drop"))


def neutralize(df, score, against="vol_20d"):
    """每日橫斷面把分數對波動率做迴歸，回傳殘差。

    對「波動率的百分位」而非原始值迴歸：波動率分布右尾很長，
    直接用原始值會被少數暴走股主導迴歸線。

    殘差的意思是「在相同波動率水準下，這檔比同儕高多少分」——
    模型若只是在挑高波動股，中性化後分數會趨於雜訊。
    """
    x = df.groupby("date")[against].rank(pct=True)
    y = df[score]
    d = df.date
    dx = x - x.groupby(d).transform("mean")
    dy = y - y.groupby(d).transform("mean")
    var = (dx * dx).groupby(d).transform("sum")
    beta = (dx * dy).groupby(d).transform("sum") / var.replace(0, np.nan)
    return (dy - beta.fillna(0) * dx).values


def tradable(df):
    """流動性過濾：避免「預測成功但無人交易」。"""
    return df[df.turnover20 >= C.MIN_TURNOVER]
