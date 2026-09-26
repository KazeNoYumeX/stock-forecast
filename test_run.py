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


def test_settle_matches_label():
    """verify 的結算與回測的標籤必須給出相同判定，否則線上表現跟回測不可比。"""
    rng = np.random.default_rng(7)
    n, H = 120, 7
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.03))
    d = pd.DataFrame({
        "symbol": "9999.TW", "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "close": close, "open": close * (1 + rng.standard_normal(n) * 0.004),
        "high": close * (1 + abs(rng.standard_normal(n)) * 0.012),
        "low": close * (1 - abs(rng.standard_normal(n)) * 0.012), "volume": 1e6})
    d[["high", "low"]] = np.c_[d[["high", "open", "close"]].max(axis=1),
                               d[["low", "open", "close"]].min(axis=1)]

    lab = M.label(d, H)
    checked = 0
    for i in range(n - H - 1):
        if not np.isfinite(lab.y.iloc[i]):
            continue
        w = d.iloc[i + 1:i + 1 + H]
        out, net, _ = M.settle_one(d.open.iloc[i + 1], w.high.values, w.low.values,
                                   w.close.values, H,
                                   up_r=C.raw_target(), dn_r=C.STOP_LOSS)
        assert (out == "win") == (lab.y.iloc[i] == 1), (i, out, lab.y.iloc[i])
        assert abs(net - lab.net_ret.iloc[i]) < 1e-12, (i, net, lab.net_ret.iloc[i])
        checked += 1
    assert checked > 80, checked

    # 還沒到期的部位要回報 open，不能當成到期平倉（平盤 3 根，兩邊都沒觸到）
    flat = np.full(3, 100.0)
    out, net, days = M.settle_one(100.0, flat, flat, flat, H)
    assert (out, days) == ("open", 3) and np.isnan(net), (out, days, net)

    # 門檻必須吃顯式參數而非環境變數：同一段 K 線用不同門檻要給出不同判定。
    # 並行追蹤多版本時，用 A 版設定結算 B 版訊號會全錯且不會報錯。
    hi = np.array([100.0, 108.0, 100.0])
    lo = np.array([100.0, 100.0, 100.0])
    assert M.settle_one(100.0, hi, lo, hi, H, up_r=0.05, dn_r=0.05)[0] == "win"
    assert M.settle_one(100.0, hi, lo, hi, H, up_r=0.12, dn_r=0.05)[0] == "open"
    lo2 = np.array([100.0, 92.0, 100.0])
    assert M.settle_one(100.0, hi, lo2, hi, H, up_r=0.05, dn_r=0.12)[0] == "win"
    assert M.settle_one(100.0, hi, lo2, hi, H, up_r=0.20, dn_r=0.05)[0] == "loss"


def test_entry_modes():
    """限價成交邏輯最容易寫錯，三種情形各驗一次。"""
    # open：無論開在哪都成交
    assert M.entry_price(110, 105, 100, "open") == 110

    # cap：開盤漲幅超過上限就放棄
    assert M.entry_price(102, 99, 100, "cap", cap=0.04) == 102
    assert M.entry_price(106, 99, 100, "cap", cap=0.04) is None

    # limit 掛前收 100：
    #   開盤已在限價之下 -> 用開盤成交（買得更便宜），不是用限價
    assert M.entry_price(97, 95, 100, "limit") == 97
    #   開高但盤中觸及限價 -> 以限價成交
    assert M.entry_price(103, 99, 100, "limit") == 100
    #   開高且全天沒回到限價 -> 沒成交
    assert M.entry_price(103, 101, 100, "limit") is None

    # 未成交的訊號在 label() 要留 NaN，不能當成虧損
    n, H = 12, 5
    d = pd.DataFrame({"symbol": "9999.TW",
                      "date": pd.date_range("2024-01-01", periods=n, freq="B"),
                      "open": 110.0, "high": 112.0, "low": 109.0, "close": 100.0,
                      "volume": 1e6})
    lab = M.label(d, H, entry="limit")          # 天天開高且不回頭 -> 全部沒成交
    assert lab.y.isna().all(), lab.y.tolist()


def test_label_stop_param_multi_symbol():
    """stop 參數曾被迴圈內的同名陣列遮蔽，第二檔起就拿到上一檔的陣列。
    單一股票的測試抓不到，所以這裡一定要用多檔。"""
    n, H = 40, 5
    rng = np.random.default_rng(1)
    frames = []
    for s in ("1111.TW", "2222.TW", "3333.TW"):
        cl = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
        frames.append(pd.DataFrame({
            "symbol": s, "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": cl, "high": cl * 1.01, "low": cl * 0.99, "close": cl, "volume": 1e6}))
    d = pd.concat(frames, ignore_index=True)

    wide = M.label(d, H, stop=0.30)        # 停損極寬 -> 幾乎不會觸停損
    tight = M.label(d, H, stop=0.005)      # 停損極窄 -> 幾乎每筆都停損
    assert wide.stopped.mean() < tight.stopped.mean(), (wide.stopped.mean(), tight.stopped.mean())
    assert tight.stopped.mean() > 0.9, tight.stopped.mean()


def test_trail_exit():
    """移動停損取代固定停利停損，邏輯錯了整個實驗就白做。"""
    a = np.array

    # 立刻下跌：峰值＝進場價，跌破 entry x (1-trail) 出場
    raw, days, st = M.trail_exit(100.0, a([100.0, 100]), a([85.0, 80]), a([88.0, 82]), 0.10)
    assert (round(raw, 10), days, st) == (-0.10, 1, 1), (raw, days, st)

    # 先漲後回檔：停損線隨峰值往上鎖，出場價是 peak x (1-trail) 而非進場價
    raw, days, st = M.trail_exit(100.0, a([110.0, 150, 140]), a([100.0, 130, 130]),
                                 a([108.0, 145, 132]), 0.10)
    assert (round(raw, 10), days, st) == (0.35, 3, 1), (raw, days, st)

    # 一路上漲沒回檔：抱到期，以最後收盤平倉，不算被掃出場
    raw, days, st = M.trail_exit(100.0, a([110.0, 120, 130]), a([105.0, 115, 125]),
                                 a([108.0, 118, 128]), 0.10)
    assert (round(raw, 10), days, st) == (0.28, 3, 0), (raw, days, st)

    # 停損線只能往上鎖，不能跟著回檔往下 —— 跌完又反彈不該重設起算點
    raw, _, _ = M.trail_exit(100.0, a([200.0, 120, 120]), a([190.0, 115, 115]),
                             a([195.0, 118, 118]), 0.10)
    assert round(raw, 10) == 0.80, raw          # 峰值鎖在 200 -> 出場 180

    # 同一根的最高價不得用來設自己的停損線（偏樂觀）
    raw, days, _ = M.trail_exit(100.0, a([150.0, 100]), a([130.0, 100]),
                                a([135.0, 100]), 0.10)
    assert days == 2 and round(raw, 10) == 0.35, (raw, days)


def test_random_has_no_lift():
    """虛無假設：隨機選股的同波動 lift 必須是 1.0。不是的話量尺本身有偏誤。"""
    rng = np.random.default_rng(3)
    n_day, n_sym = 60, 200
    rows = []
    for d in pd.date_range("2024-01-01", periods=n_day, freq="B"):
        vol = rng.lognormal(-3, 0.6, n_sym)
        # 命中率刻意與波動率正相關 —— 若量尺沒扣掉波動，隨機選股也會跑出 lift
        p = 0.15 + 0.5 * (vol.argsort().argsort() / n_sym)
        rows.append(pd.DataFrame({
            "date": d, "symbol": [f"s{i}" for i in range(n_sym)], "vol_20d": vol,
            "y": (rng.random(n_sym) < p).astype(float),
            "stopped": 0.0, "net_ret": 0.0, "hold_days": 5.0,
            "rnd": rng.random(n_sym)}))
    test = B.vol_matched(pd.concat(rows, ignore_index=True))
    base = test.y.mean()

    # 隨機選股：兩種 lift 都該是 1.0
    m = B.metrics(B.pick(test, "rnd", 20), base)
    assert abs(m["vlift"] - 1.0) < 0.05, f"隨機選股的同波動 lift = {m['vlift']:.3f}"
    assert abs(m["lift"] - 1.0) < 0.10, f"隨機選股的全市場 lift = {m['lift']:.3f}"

    # 按波動率選股：這份合成資料的命中率與波動率正相關，所以全市場 lift 會虛高，
    # 而同波動 lift 必須把這個假優勢扣回 1.0 —— 這正是量尺存在的理由。
    v = B.metrics(B.pick(test, "vol_20d", 20), base)
    assert v["lift"] > 1.4, f"照理該虛高，實測 {v['lift']:.3f}"
    assert abs(v["vlift"] - 1.0) < 0.10, f"同波動 lift 沒扣乾淨：{v['vlift']:.3f}"


def test_equity_matches_turnover():
    """資金曲線與週轉率公式是兩條獨立路徑，必須相符 —— 這正是抓到 equity() 錯誤的方式。"""
    # 持有天數要有變化，出場日才會散落到週末 —— 否則測不出「日曆日 vs 交易日」的差異
    top_n, ret = 4, 0.01
    dates = pd.date_range("2024-01-01", periods=600, freq="B").repeat(top_n)
    holds = np.tile([3.0, 4.0, 5.0, 6.0, 7.0], len(dates) // 5 + 1)[:len(dates)]
    picks = pd.DataFrame({"engine": "e", "net_ret": ret,
                          "hold_days": holds, "date": dates})
    hold = holds.mean()
    eq = B.equity(picks, "e", top_n=top_n)
    c_turn = B.cagr_of(ret, hold)
    # 用曲線的點數當分母是錯的：出場日是日曆日（含週末），而 252 是交易日。
    wrong = eq.iloc[-1] ** (252 / len(eq)) - 1
    assert abs(B.curve_cagr(eq) - c_turn) / c_turn < 0.20, (B.curve_cagr(eq), c_turn)
    assert abs(wrong - c_turn) / c_turn > 0.20, "這份合成資料要能凸顯兩種分母的差異"


def test_grid_selection_rejects_one_sided():
    """錄取條件必須拒絕「靠單邊撐起來」的格子 ——
    集中度 TOP_N=1 與跳空過濾 4% 都是這樣騙過我的。"""
    base = {"vlift": 1.3, "vlift_sd": 0.15, "prec": 0.3, "ret": 0.002,
            "rnd_cagr": -0.14, "h": 7, "t": 0.12, "s": 0.05, "folds": 13}
    cells = pd.DataFrame([
        {**base, "cagr": 0.80, "h1": -0.05, "h2": 1.70},   # 年化最高，但前半是負的
        {**base, "cagr": 0.60, "h1": 0.05, "h2": 1.20},    # 兩半都正但差 24 倍
        {**base, "cagr": 0.25, "h1": 0.22, "h2": 0.28},    # 略低但兩半一致 <- 應選這個
        {**base, "cagr": 0.30, "h1": 0.28, "h2": 0.32, "rnd_cagr": 0.25},  # 贏不過隨機
        {**base, "cagr": 0.40, "h1": 0.35, "h2": 0.45, "vlift_sd": 0.55},  # 逐折太散
    ])
    w = B.select(cells)
    assert w is not None and abs(w["cagr"] - 0.25) < 1e-9, w["cagr"]

    # 每一條規則都要真的會擋
    for i, expect in enumerate(["兩半年化都為正", "兩半差距 < 2x", None,
                                "贏隨機 > 10pp", "逐折 std < 0.30"]):
        bad = B._judge(cells.iloc[i])
        if expect is None:
            assert not bad, bad
        else:
            assert expect in bad, (i, bad)

    # 全部落選時要回 None，不能偷偷放寬
    assert B.select(cells[cells.h1 < 0]) is None


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
