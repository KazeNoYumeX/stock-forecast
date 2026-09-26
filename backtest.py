"""Walk-forward 回測：規則 vs ML 同場比較。"""
import numpy as np
import pandas as pd

import config as C
import model as M


def folds(dates, horizon):
    """訓練 -> embargo -> 測試，測試期不重疊。

    embargo 是關鍵：label 用到未來 horizon 天的價格，訓練集尾端不砍掉這段，
    測試期資訊會滲進訓練集，回測數字會漂亮到假。
    """
    emb = horizon + 1
    i = 0
    while i + C.TRAIN_DAYS + emb + C.TEST_DAYS <= len(dates):
        tr = dates[i:i + C.TRAIN_DAYS - emb]
        te = dates[i + C.TRAIN_DAYS + emb:i + C.TRAIN_DAYS + emb + C.TEST_DAYS]
        yield tr, te
        i += C.TEST_DAYS


ENGINES = [("rule", "規則排序"), ("ml", "ML 純價量"),
           ("ml_chip", "ML 價量+籌碼"), ("ml_neut", "ML 波動率中性化")]


def vol_matched(test):
    """加上 vol_q / vol_base / vol_stop_base。

    同日同波動分層的命中率＝正確的對照組。用 leave-one-out，
    否則每檔都會把自己算進自己的基準裡。
    """
    test["vol_q"] = M.vol_bucket(test)
    cnt = test.groupby(["date", "vol_q"]).y.transform("count")
    for col, dst in (("y", "vol_base"), ("stopped", "vol_stop_base")):
        s = test.groupby(["date", "vol_q"])[col].transform("sum")
        test[dst] = ((s - test[col]) / (cnt - 1)).where(cnt > 1)
    return test


def pick(test, col, top_n):
    """每日取分數前 N 名。"""
    return test[test.groupby("date")[col].rank(ascending=False, method="first") <= top_n]


def metrics(sel, base):
    vb, sb = sel.vol_base.mean(), sel.vol_stop_base.mean()
    p = sel.y.mean()
    return {"prec": p, "lift": p / base if base else np.nan,
            "vbase": vb, "vlift": p / vb if vb else np.nan, "vq": sel.vol_q.mean(),
            "stop": sel.stopped.mean(), "stopbase": sel.vol_stop_base.mean(),
            "stoplift": sel.stopped.mean() / sb if sb else np.nan,
            "ret": sel.net_ret.mean(), "hold": sel.hold_days.mean(), "n": len(sel)}


def curve_cagr(eq):
    """資金曲線的年化。用實際跨越的日曆天數，不是曲線的點數。"""
    span = (eq.index[-1] - eq.index[0]).days
    return eq.iloc[-1] ** (365.25 / span) - 1 if span > 0 else np.nan


def cagr_of(ret, hold):
    """滿倉輪轉年化：持有 d 天賺 r，一年輪轉 252/d 次。"""
    return (1 + ret) ** (252 / hold) - 1 if hold and hold > 0 else np.nan


def run(df, horizon, top_n=None, last_only=False):
    """df 需含特徵 + y + net_ret。回傳 (每折摘要, 逐筆選股)。

    籌碼欄位刻意不 dropna：HistGradientBoosting 原生吃 NaN，丟掉反而會讓
    三個引擎的樣本數不一致，比較就不公平了。
    """
    top_n = top_n or C.TOP_N
    d = M.tradable(df).dropna(subset=["y"] + M.PRICE_FEATURES).copy()
    dates = np.sort(d.date.unique())
    rows, picks = [], []

    fs = list(folds(dates, horizon))
    for tr_d, te_d in (fs[-1:] if last_only else fs):      # daily 只需要最後一折做校準
        train = d[d.date.isin(tr_d)]
        test = d[d.date.isin(te_d)].copy()
        if test.empty:
            continue
        test["rule"] = M.rule_score(test)
        test["ml"] = M.fit_predict(train, test, M.PRICE_FEATURES)
        test["ml_chip"] = M.fit_predict(train, test, M.FEATURES)
        test["ml_neut"] = M.neutralize(test, "ml_chip")

        # 同日同波動率分層的命中率＝正確的對照組。用 leave-one-out，
        # 否則每檔都會把自己算進自己的基準裡。
        test = vol_matched(test)

        base = test.y.mean()
        r = {"test_start": str(pd.Timestamp(te_d[0]).date()),
             "test_end": str(pd.Timestamp(te_d[-1]).date()),
             "n": len(test), "base": base}
        for eng, _ in ENGINES:
            sel = pick(test, eng, top_n)
            r.update({f"{eng}_{k}": v for k, v in metrics(sel, base).items()})
            picks.append(sel.assign(engine=eng, fold=r["test_start"]))
        rows.append(r)

    return pd.DataFrame(rows), (pd.concat(picks, ignore_index=True) if picks else pd.DataFrame())


def equity(picks, engine, top_n=None):
    """滿倉輪轉的資金曲線。

    每天開 N 檔、平均持有 h 天 -> 同時壓著 N*h 個部位，每筆權重 1/(N*h)。
    報酬認列在「出場日」而非進場日，否則回撤會被提早算到。
    交易日轉日曆日用 7/5 近似，只影響曲線的時間軸不影響報酬。
    """
    p = picks[picks.engine == engine].dropna(subset=["net_ret", "hold_days"])
    if p.empty:
        return pd.Series(dtype=float)
    w = 1.0 / max(1e-9, (top_n or C.TOP_N) * p.hold_days.mean())
    exit_d = p.date + pd.to_timedelta((p.hold_days * 7 / 5).round(), unit="D")
    return (1 + p.assign(x=exit_d).groupby("x").net_ret.sum() * w).sort_index().cumprod()


def summarize(report, picks, horizon):
    out = [f"\n=== Walk-forward 回測  H={horizon}d  毛門檻 {C.raw_target() * 100:.2f}% "
           f"(淨 {C.NET_TARGET * 100:.1f}%)  停損 {C.STOP_LOSS * 100:.1f}% ===",
           f"{len(report)} 折 | 測試期 {report.test_start.iloc[0]} ~ {report.test_end.iloc[-1]}",
           f"基礎機率 base rate: {report.base.mean() * 100:.2f}%  "
           f"（隨機挑一檔，{horizon} 天內先觸停利的機率）\n",
           f"{'引擎':<16}{'precision':>10}{'全市場lift':>12}{'同波動lift':>12}"
           f"{'picks波動分層':>14}{'淨報酬':>9}{'持有天':>8}{'年化(滿倉)':>12}{'最大回撤':>10}"]

    for eng, label in ENGINES:
        eq = equity(picks, eng)
        mdd = (eq / eq.cummax() - 1).min() if len(eq) else np.nan
        # 年化用資金週轉率算：持有 d 天賺 r，滿倉輪轉一年就是 (1+r)^(252/d)-1。
        # 與 equity() 曲線是兩條獨立路徑，兩者不符就代表其中一邊算錯了。
        ret, hold = report[f"{eng}_ret"].mean(), report[f"{eng}_hold"].mean()
        cagr = cagr_of(ret, hold)
        out.append(f"{label:<16}{report[f'{eng}_prec'].mean()*100:>9.2f}%"
                   f"{report[f'{eng}_lift'].mean():>11.2f}x"
                   f"{report[f'{eng}_vlift'].mean():>11.2f}x"
                   f"{report[f'{eng}_vq'].mean():>13.1f}"
                   f"{report[f'{eng}_ret'].mean()*100:>+8.2f}%"
                   f"{hold:>8.1f}"
                   f"{cagr*100:>+11.1f}%"
                   f"{mdd*100:>9.1f}%")
    out.append("  同波動lift = 跟「同一天、同波動率分層」的股票比。這才是真正的選股能力，"
               "\n  全市場lift 會把「專挑高波動股」也算成本事。picks波動分層 0=最低波動 9=最高。"
               "\n  年化假設資金滿倉輪轉（持有到期就換下一批），未計入等待訊號的閒置期。")

    out.append("\n觸停損機率 vs 同波動同儕（檢驗模型是在預測『方向』還是只在預測『會大動』）:")
    for eng, label in ENGINES:
        s, sb = report[f"{eng}_stop"].mean(), report[f"{eng}_stopbase"].mean()
        out.append(f"  {label:<16} 停損率 {s*100:5.2f}%  同儕 {sb*100:5.2f}%  "
                   f"比值 {s/sb if sb else float('nan'):.2f}x")
    out.append("  比值 > 1 代表 picks 也比同儕更容易被掃出場 -> 抓到的是波動而非方向。")

    out.append("\n逐折表現（檢查是不是只靠某一段行情撐起來）:")
    t = report[["test_start", "base"]].copy()
    t["lift"] = report.ml_neut_vlift.round(2)
    t["每筆%"] = (report.ml_neut_ret * 100).round(2)
    t["年化%"] = [round(cagr_of(r, h) * 100, 1) for r, h in
                 zip(report.ml_neut_ret, report.ml_neut_hold)]
    t["base"] = t.base.round(3)
    out.append(t.to_string(index=False))

    chip_gain = report.ml_chip_vlift.mean() - report.ml_vlift.mean()
    out.append(f"\n籌碼面貢獻（同波動基準）：{chip_gain:+.3f}"
               f"（{(report.ml_chip_vlift > report.ml_vlift).mean()*100:.0f}% 的折數贏過純價量）")
    neut = report.ml_neut_vlift.mean()
    out.append(f"中性化後：同波動lift {neut:.2f}x，"
               f"{(report.ml_neut_vlift > 1).mean()*100:.0f}% 的折數 > 1.0")

    best = max((e for e, _ in ENGINES), key=lambda e: report[f"{e}_vlift"].mean())
    if report[f"{best}_vlift"].mean() < 1.15:
        out.append("\n判定：扣掉波動率後 lift < 1.15，模型沒有真正的選股能力，"
                   "先前的高 precision 來自「挑波動大的股票」。")
    else:
        out.append(f"\n判定：{best} 扣掉波動率後仍有 "
                   f"{report[f'{best}_vlift'].mean():.2f}x，是真的選股能力。")
    return "\n".join(out)


# --- 穩定性審核 --------------------------------------------------------

def _folds_of(feat, horizon, top_ns, net=None, entry="open", cap=None, stop=None,
              exit="barrier", trail=None, seed=0):
    """標籤一次、每折訓練一次，然後一次評估所有 top_n。

    top_n 只影響「取前幾名」，不影響標籤與訓練 —— 每個 top_n 重跑整個回測是浪費。
    回傳 (逐折逐 top_n 的長表, 選股明細)。
    """
    d = M.tradable(M.label(feat, horizon, net=net, entry=entry, cap=cap, stop=stop,
                           exit=exit, trail=trail))
    d = d.dropna(subset=["y"] + M.PRICE_FEATURES).copy()
    n_sig = len(d)
    dates = np.sort(d.date.unique())
    rng = np.random.default_rng(seed)

    rows, picks = [], []
    for tr_d, te_d in folds(dates, horizon):
        train, test = d[d.date.isin(tr_d)], d[d.date.isin(te_d)].copy()
        if test.empty:
            continue
        test["ml_chip"] = M.fit_predict(train, test, M.FEATURES)
        test["ml_neut"] = M.neutralize(test, "ml_chip")
        test["rnd"] = rng.random(len(test))          # 查驗 1 的虛無假設
        test = vol_matched(test)
        base = test.y.mean()
        start = str(pd.Timestamp(te_d[0]).date())
        for n in top_ns:
            for eng in ("ml_neut", "rnd"):
                sel = pick(test, eng, n)
                rows.append({"fold": start, "top_n": n, "engine": eng,
                             "base": base, **metrics(sel, base)})
                if eng == "ml_neut":
                    picks.append(sel.assign(engine=eng, top_n=n))
    return pd.DataFrame(rows), (pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()), n_sig


def _agg(r):
    """一組逐折紀錄 -> 摘要。"""
    ret, hold = r.ret.mean(), r.hold.mean()
    return {"vlift": r.vlift.mean(), "vlift_sd": r.vlift.std(), "vlift_min": r.vlift.min(),
            "vlift_max": r.vlift.max(), "win": (r.vlift > 1).mean(),
            "prec": r.prec.mean(), "ret": ret, "hold": hold, "cagr": cagr_of(ret, hold)}


def audit(feat, horizon, targets, tops, bench=None):
    out = [f"\n{'=' * 78}\n穩定性審核  H={horizon}d  停損 {C.STOP_LOSS * 100:.1f}%\n{'=' * 78}"]
    base_target = C.NET_TARGET
    cache = {}
    for t in targets:
        cache[t] = _folds_of(feat, horizon, tops, net=t)

    r0, p0, _ = cache[base_target] if base_target in cache else cache[targets[0]]
    ref_t = base_target if base_target in cache else targets[0]

    # --- 1. 隨機基準（量尺本身有沒有偏誤）---
    rnd = r0[r0.engine == "rnd"]
    rv = rnd.vlift.mean()
    ok1 = abs(rv - 1.0) <= 0.10
    ra = _agg(rnd)
    out += ["\n[1] 隨機基準　隨機選股的同波動 lift 應該等於 1.00",
            f"    實測 {rv:.3f}（各 top_n 範圍 {rnd.groupby('top_n').vlift.mean().min():.3f}"
            f" ~ {rnd.groupby('top_n').vlift.mean().max():.3f}）",
            f"    {'通過 — 量尺沒有偏誤' if ok1 else '不通過 — 量尺有偏誤，以下結論全部作廢'}",
            f"    隨機選股在同一套停利停損下的年化：{ra['cagr']*100:+.1f}%"
            f"（每筆 {ra['ret']*100:+.2f}%）",
            "    ^ 這是「規則本身 + 市場 beta」的貢獻。模型的年化要明顯超過這個數字才算有用。"]
    if not ok1:
        return "\n".join(out)

    # --- 2. equity 曲線與週轉率公式是否相符 ---
    out.append("\n[2] equity 自我一致性　曲線 CAGR vs 週轉率公式（兩條獨立路徑）")
    for n in tops:
        g = r0[(r0.engine == "ml_neut") & (r0.top_n == n)]
        eq = equity(p0[p0.top_n == n], "ml_neut", top_n=n)
        if len(eq) < 20:
            continue
        # 曲線索引是出場日（日曆日，含週末），不能拿 252 交易日當分母 ——
        # 819 個交易日散在約 1,100 個日曆日上，分母高估三成會讓數字一律偏保守。
        c_curve = curve_cagr(eq)
        c_turn = cagr_of(g.ret.mean(), g.hold.mean())
        rel = abs(c_curve - c_turn) / max(1e-9, abs(c_turn))
        out.append(f"    top_n={n:<3} 曲線 {c_curve*100:+7.1f}%  公式 {c_turn*100:+7.1f}%  "
                   f"差異 {rel*100:4.0f}%  {'ok' if rel < 0.20 else '不符'}")

    # --- 3+4+5. 目標 x 檔數格點、趨勢斷點、逐折離散度 ---
    out.append(f"\n[3-5] 目標 x 檔數格點（同波動 lift ／ 年化 ／ 逐折 std）")
    out.append(f"    {'目標':<8}" + "".join(f"{'N=' + str(n):>22}" for n in tops))
    for t in targets:
        rt = cache[t][0]
        cells = []
        for n in tops:
            a = _agg(rt[(rt.engine == "ml_neut") & (rt.top_n == n)])
            cells.append(f"{a['vlift']:.2f}x/{a['cagr']*100:+.0f}%/±{a['vlift_sd']:.2f}".rjust(22))
        out.append(f"    {t*100:>5.0f}%  " + "".join(cells))
    out.append("    每格＝同波動lift / 年化(滿倉) / 逐折標準差。lift 若靠單折撐起，標準差會很大。")

    # --- 3b. 分半穩定性 ---
    out.append("\n[3b] 分半穩定性　同一組參數要在前後兩半都站得住")
    folds_all = sorted(r0.fold.unique())
    half = len(folds_all) // 2
    for t in targets:
        rt = cache[t][0]
        rt = rt[rt.engine == "ml_neut"]
        line = []
        for n in tops:
            s = rt[rt.top_n == n]
            h1 = s[s.fold.isin(folds_all[:half])].vlift.mean()
            h2 = s[s.fold.isin(folds_all[half:])].vlift.mean()
            line.append(f"{h1:.2f}/{h2:.2f}".rjust(22))
        out.append(f"    {t*100:>5.0f}%  " + "".join(line))
    out.append(f"    前半 {folds_all[0]}~ ／ 後半 {folds_all[half]}~。兩個數字差很多＝過擬合。")

    # --- 5b. 最佳組合的逐折離散度 ---
    best = None
    for t in targets:
        rt = cache[t][0]
        for n in tops:
            a = _agg(rt[(rt.engine == "ml_neut") & (rt.top_n == n)])
            if best is None or a["cagr"] > best[0]["cagr"]:
                best = (a, t, n)
    a, bt, bn = best
    out += [f"\n[5b] 年化最高的組合：目標 {bt*100:.0f}%　TOP_N={bn}",
            f"    同波動 lift  平均 {a['vlift']:.2f}  標準差 {a['vlift_sd']:.2f}  "
            f"最差折 {a['vlift_min']:.2f}  最好折 {a['vlift_max']:.2f}  勝折 {a['win']*100:.0f}%",
            f"    年化 {a['cagr']*100:+.1f}%　每筆淨報酬 {a['ret']*100:+.2f}%　平均持有 {a['hold']:.1f} 天"]

    # --- 6. 市場相依性 ---
    if bench is not None:
        rt = cache[bt][0]
        s = rt[(rt.engine == "ml_neut") & (rt.top_n == bn)].set_index("fold")
        mk = {}
        for f in s.index:
            w = bench[bench.index >= pd.Timestamp(f)].head(63)
            mk[f] = w.iloc[-1] / w.iloc[0] - 1 if len(w) > 1 else np.nan
        m = pd.Series(mk)
        up, dn = m > 0, m <= 0
        out += ["\n[6] 市場相依性　優勢是否只存在於上漲季",
                f"    上漲季 {up.sum()} 折：lift {s.vlift[up.values].mean():.2f}",
                f"    下跌季 {dn.sum()} 折：lift {s.vlift[dn.values].mean():.2f}"
                + ("" if dn.sum() else "　← 測試期沒有下跌季，等於完全沒被測過")]

    # --- 7+8. 進場方式 ---
    out.append(f"\n[7-8] 進場方式（目標 {ref_t*100:.0f}%，TOP_N={bn}）　成交率會影響資金利用率")
    _, _, n_base = cache[ref_t]
    variants = [("開盤市價（回測基準）", "open", None)] + \
               [(f"開盤漲逾 {c*100:.0f}% 就跳過", "cap", c) for c in (0.02, 0.04, 0.06)] + \
               [("掛前一日收盤等成交", "limit", None)]
    for name, mode, cap in variants:
        rr, _, n_sig = _folds_of(feat, horizon, [bn], net=ref_t, entry=mode, cap=cap)
        a = _agg(rr[(rr.engine == "ml_neut") & (rr.top_n == bn)])
        fill = n_sig / n_base
        out.append(f"    {name:<22} 成交率 {fill*100:5.1f}%  lift {a['vlift']:.2f}  "
                   f"每筆 {a['ret']*100:+.2f}%  年化 {a['cagr']*100:+7.1f}%  "
                   f"折算閒置後 {a['cagr']*fill*100:+7.1f}%")
    out.append("    「折算閒置後」＝年化 x 成交率。跳過的單子讓資金空著，不折算就是拿蘋果比橘子。")
    return "\n".join(out)



# --- 三維格點搜尋 ------------------------------------------------------

# 錄取條件，事前講定。先前的教訓：集中度(TOP_N=1)與跳空過濾(4%)都是掃出來的
# 最佳格，樣本放大後全部消失。所以挑的是「兩半都站得住」而不是「年化最高」。
RULES = [
    ("兩半年化都為正", lambda c: c["h1"] > 0 and c["h2"] > 0),
    # 比值檢定在數字接近零時沒有意義（+0.4% vs +0.9% 是 2.25 倍，但只差 0.5pp），
    # 所以補一個絕對容差。這是修正數學缺陷，不是為了讓某一格過關而放寬 ——
    # 實際套用後只多錄取一格，而那格年化 +0.7%，經濟意義約等於零。
    ("兩半差距 < 2x 或 5pp 內",
     lambda c: min(abs(c["h1"]), abs(c["h2"])) * 2 >= max(abs(c["h1"]), abs(c["h2"]))
     or abs(c["h1"] - c["h2"]) <= 0.05),
    ("贏隨機 > 10pp", lambda c: c["cagr"] - c["rnd_cagr"] > 0.10),
    ("逐折 std < 0.30", lambda c: c["vlift_sd"] < 0.30),
    # 只在多頭驗證過的策略不算驗證過。2022 年台股 -22.6%，測試期起點必須涵蓋它。
    # 實例：H=7/停利5%/停損20% 通過上面四條（年化 +0.7%），納入 2022 後變成 -10.3%、
    # 回撤 -58.0%。預設 TRAIN_DAYS=504 會讓第一折從 2023-02 才開始，要用 252。
    ("測試期涵蓋 2022 空頭", lambda c: c.get("first_fold", "9999") <= "2022-07-01"),
]


def _half(r, folds_all, lo, hi):
    s = r[r.fold.isin(folds_all[lo:hi])]
    return cagr_of(s.ret.mean(), s.hold.mean()) if len(s) else np.nan


def grid(feat, horizons, targets, stops, top_n=None, trails=None):
    """逐格重新標籤 + 訓練。top_n 只影響取前幾名，所以固定一個值就好。

    trails 有給就跑移動停損（沒有停利天花板，一個參數取代停利與停損），
    此時 targets / stops 不適用。
    """
    top_n = top_n or C.TOP_N
    combos = ([(t, None, "trail", t) for t in trails] if trails else
              [(t, s, "barrier", None) for t in targets for s in stops])
    cells = []
    for h in horizons:
        for t, s, mode, tr in combos:
            if True:
                r, _, _ = _folds_of(feat, h, [top_n], net=t if mode == "barrier" else None,
                                    stop=s, exit=mode, trail=tr)
                if r.empty:
                    continue
                mine = r[r.engine == "ml_neut"]
                rnd = r[r.engine == "rnd"]
                fa = sorted(mine.fold.unique())
                half = len(fa) // 2
                a = _agg(mine)
                cells.append({"h": h, "t": t, "s": s, "mode": mode, "folds": len(fa),
                              "first_fold": fa[0],
                              "vlift": a["vlift"], "vlift_sd": a["vlift_sd"],
                              "prec": a["prec"], "ret": a["ret"], "cagr": a["cagr"],
                              "rnd_cagr": _agg(rnd)["cagr"],
                              "h1": _half(mine, fa, 0, half),
                              "h2": _half(mine, fa, half, len(fa))})
                c = cells[-1]
                tag = (f"回檔{tr*100:>3.0f}%      " if mode == "trail"
                       else f"停利{t*100:>3.0f}% 停損{s*100:>3.0f}%")
                print(f"  H={h:<3} {tag}  "
                      f"lift {c['vlift']:.2f}  年化 {c['cagr']*100:+7.1f}%  "
                      f"(前半 {c['h1']*100:+6.1f}% 後半 {c['h2']*100:+6.1f}%)  "
                      f"隨機 {c['rnd_cagr']*100:+6.1f}%", flush=True)
    return pd.DataFrame(cells)


def _judge(c):
    """回傳落選的規則名稱清單，空的代表錄取。"""
    return [n for n, f in RULES if not f(c)]


def select(df, target=None):
    """錄取條件全過的格子裡取年化最高。target 給定時只看該停利目標。"""
    d = df if target is None else df[df.t == target]
    ok = [c for _, c in d.iterrows() if not _judge(c)]
    return max(ok, key=lambda c: c["cagr"]) if ok else None


def grid_report(df):
    out = [f"\n{'=' * 96}", "格點搜尋結果（TOP_N 固定，錄取條件事前講定）", "=" * 96,
           f"{'H':>3}{'停利/回檔':>9}{'停損':>7}{'折':>4}{'lift':>7}{'std':>7}"
           f"{'命中':>7}{'年化':>9}{'前半':>9}{'後半':>9}{'隨機':>9}  判定"]
    for _, c in df.sort_values("cagr", ascending=False).iterrows():
        bad = _judge(c)
        s_txt = "  移動" if c.get("mode") == "trail" else f"{c.s*100:>6.0f}%"
        out.append(f"{c.h:>3}{c.t*100:>8.0f}%{s_txt:>7}{c.folds:>4}"
                   f"{c.vlift:>7.2f}{c.vlift_sd:>7.2f}{c.prec*100:>6.1f}%"
                   f"{c.cagr*100:>+8.1f}%{c.h1*100:>+8.1f}%{c.h2*100:>+8.1f}%"
                   f"{c.rnd_cagr*100:>+8.1f}%  "
                   + ("錄取" if not bad else "落選：" + "、".join(bad)))
    out.append("\n錄取條件：" + "　".join(n for n, _ in RULES))

    for label, tgt in (("推薦版（不限停利）", None), ("使用者版（停利固定 5%）", 0.05)):
        w = select(df, tgt)
        if w is None:
            out.append(f"\n{label}：沒有任何格子通過錄取條件。不放寬條件 —— "
                       "這代表在這個搜尋範圍內找不到可信的設定。")
        else:
            out.append(f"\n{label}：H={w['h']}d　停利 {w['t']*100:.0f}%　停損 {w['s']*100:.0f}%"
                       f"　年化 {w['cagr']*100:+.1f}%（前半 {w['h1']*100:+.1f}%／"
                       f"後半 {w['h2']*100:+.1f}%）　贏隨機 "
                       f"{(w['cagr'] - w['rnd_cagr'])*100:+.0f}pp")
    return "\n".join(out)
