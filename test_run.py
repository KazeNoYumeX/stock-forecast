#!/usr/bin/env python3
"""守住兩個會安靜出錯的地方：金錢計算、資料洩漏。"""
import numpy as np
import pandas as pd

import backtest as B
import config as C
import model as M


def test_cost_math():
    for etf in (False, True):
        raw = C.raw_target(etf=etf)
        assert abs(C.net_return(raw, etf=etf) - C.NET_TARGET) < 1e-12, "毛/淨反解不一致"
        assert raw > C.NET_TARGET, "毛門檻必須高於淨目標，否則費用沒扣到"
    # 手算驗證：買 100 賣 105.8287，未折扣個股
    raw = C.raw_target()
    buy, sell = 100 * (1 + C.FEE_RATE + C.SLIPPAGE), 100 * (1 + raw) * (1 - C.FEE_RATE - C.TAX_STOCK - C.SLIPPAGE)
    assert abs(sell / buy - 1 - 0.05) < 1e-12


def _series(closes):
    n = len(closes)
    return pd.DataFrame({"symbol": "9999.TW", "date": pd.date_range("2024-01-01", periods=n, freq="B"),
                         "open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": 1e6})


def test_barrier_label():
    H = 5
    up = C.raw_target() + 0.01
    # 先觸停利 -> 1
    hit_up = M.label(_series([100.0, 100, 100, 100 * (1 + up), 100, 100, 100]), H).y.iloc[0]
    # 先觸停損 -> 0
    hit_dn = M.label(_series([100.0, 100, 100, 100 * (1 - C.STOP_LOSS - 0.01), 100, 100, 100]), H).y.iloc[0]
    # 盤整到期 -> 0
    flat = M.label(_series([100.0] * 7), H).y.iloc[0]
    assert (hit_up, hit_dn, flat) == (1.0, 0.0, 0.0), (hit_up, hit_dn, flat)
    # 同根 K 同時觸及 -> 保守判停損
    d = _series([100.0] * 7)
    d.loc[3, ["high", "low"]] = [100 * (1 + up), 100 * (1 - C.STOP_LOSS - 0.01)]
    assert M.label(d, H).y.iloc[0] == 0.0, "同時觸及應保守判停損"
    # 進場價是隔日開盤，不是當日收盤（無前視偏誤）
    d = _series([100.0] * 7)
    d.loc[1, "open"] = 50.0                      # 隔日開盤腰斬
    d.loc[2:, ["high", "low", "close"]] = 50.0 * (1 + up)
    assert M.label(d, H).y.iloc[0] == 1.0, "進場價應取隔日開盤"


def test_chip_features():
    n = 10
    d = pd.DataFrame({"symbol": "9999.TW", "date": pd.date_range("2024-01-01", periods=n, freq="B"),
                      "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0})
    c = pd.DataFrame({"symbol": "9999.TW", "date": d.date,
                      "foreign_net": [100.0] * n,
                      "trust_net": [10, 10, 10, -5, 10, 10, 10, 10, 0, 10],
                      "inst_net": 0.0, "margin_bal": [100.0] * 5 + [150.0] * 5,
                      "short_bal": [20.0] * n})
    g = M.add_chips(d, c).sort_values("date")
    # 買賣超正規化成「吃掉幾 % 成交量」：5 日 500 股 / 5 日 5000 股 = 0.1
    assert abs(g.fgn_5.iloc[-1] - 0.1) < 1e-9, g.fgn_5.iloc[-1]
    # 投信連買天數：最後 4 天是 10,10,0,10 -> streak=1
    assert list(g.tru_streak) == [1, 2, 3, 0, 1, 2, 3, 4, 0, 1], list(g.tru_streak)
    assert abs(g.short_ratio.iloc[-1] - 20 / 150) < 1e-9
    # 籌碼缺資料時留 NaN 而非 0（0 會被模型當成「法人剛好沒買」的真實訊號）
    empty = M.add_chips(d.copy(), pd.DataFrame())
    assert empty[M.CHIP_FEATURES].isna().all().all()


def test_neutralize():
    rng = np.random.default_rng(0)
    n_day, n_sym = 40, 80
    rows = []
    for d in pd.date_range("2024-01-01", periods=n_day, freq="B"):
        vol = rng.lognormal(-3, 0.6, n_sym)                 # 右尾很長，跟真實波動率一樣
        rows.append(pd.DataFrame({
            "date": d, "symbol": [f"s{i}" for i in range(n_sym)], "vol_20d": vol,
            # 分數幾乎完全由波動率驅動 —— 中性化後應該只剩雜訊
            "score": 3.0 * vol + 0.05 * rng.standard_normal(n_sym)}))
    df = pd.concat(rows, ignore_index=True)

    before = df.groupby("date").apply(
        lambda g: g.score.corr(g.vol_20d.rank(pct=True)), include_groups=False).mean()
    df["resid"] = M.neutralize(df, "score")
    after = df.groupby("date").apply(
        lambda g: g.resid.corr(g.vol_20d.rank(pct=True)), include_groups=False).mean()
    # 0.8 而非 1.0：分數是波動率的線性函數，但這裡比對的是波動率百分位，
    # 對數常態分布跟自己的百分位之間本來就只有 ~0.8 的 Pearson 相關。
    assert before > 0.7, before
    assert abs(after) < 1e-9, f"中性化後仍與波動率相關: {after}"

    # 分層：每日切 10 層，標號 0..9，每層檔數相近
    df["q"] = M.vol_bucket(df)
    assert set(df.q.unique()) == set(range(10)), sorted(df.q.unique())
    sizes = df.groupby(["date", "q"]).size()
    assert sizes.min() >= n_sym // 10 - 1 and sizes.max() <= n_sym // 10 + 1, (sizes.min(), sizes.max())


def test_embargo():
    dates = pd.date_range("2020-01-01", periods=C.TRAIN_DAYS + C.TEST_DAYS + 200, freq="B").values
    for h in C.HORIZONS:
        got = list(B.folds(dates, h))
        assert got, "切不出任何一折"
        for tr, te in got:
            gap = np.busday_count(tr[-1].astype("datetime64[D]"), te[0].astype("datetime64[D]"))
            assert gap > h, f"embargo 不足：訓練尾端到測試起點只有 {gap} 天，需 > {h}"
        starts = [te[0] for _, te in got]
        assert len(set(starts)) == len(starts), "測試期重疊"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("全部通過")
