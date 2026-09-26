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


def run(df, horizon, top_n=None):
    """df 需含特徵 + y + net_ret。回傳 (每折摘要, 逐筆選股)。

    籌碼欄位刻意不 dropna：HistGradientBoosting 原生吃 NaN，丟掉反而會讓
    三個引擎的樣本數不一致，比較就不公平了。
    """
    top_n = top_n or C.TOP_N
    d = M.tradable(df).dropna(subset=["y"] + M.PRICE_FEATURES).copy()
    dates = np.sort(d.date.unique())
    rows, picks = [], []

    for tr_d, te_d in folds(dates, horizon):
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
        test["vol_q"] = M.vol_bucket(test)
        cnt = test.groupby(["date", "vol_q"]).y.transform("count")
        for col, dst in (("y", "vol_base"), ("stopped", "vol_stop_base")):
            s = test.groupby(["date", "vol_q"])[col].transform("sum")
            test[dst] = ((s - test[col]) / (cnt - 1)).where(cnt > 1)

        base = test.y.mean()
        r = {"test_start": str(pd.Timestamp(te_d[0]).date()),
             "test_end": str(pd.Timestamp(te_d[-1]).date()),
             "n": len(test), "base": base}
        for eng, _ in ENGINES:
            sel = test[test.groupby("date")[eng].rank(ascending=False, method="first") <= top_n]
            vbase = sel.vol_base.mean()
            r[f"{eng}_prec"] = sel.y.mean()
            r[f"{eng}_lift"] = sel.y.mean() / base if base else np.nan
            r[f"{eng}_vbase"] = vbase
            r[f"{eng}_vlift"] = sel.y.mean() / vbase if vbase else np.nan
            r[f"{eng}_vq"] = sel.vol_q.mean()
            r[f"{eng}_stop"] = sel.stopped.mean()
            r[f"{eng}_stopbase"] = sel.vol_stop_base.mean()
            r[f"{eng}_ret"] = sel.net_ret.mean()
            r[f"{eng}_hold"] = sel.hold_days.mean()
            sel = sel.assign(engine=eng, fold=r["test_start"])
            picks.append(sel)
        rows.append(r)

    return pd.DataFrame(rows), (pd.concat(picks, ignore_index=True) if picks else pd.DataFrame())


def equity(picks, engine, top_n=None):
    """等權持有每日 top-N 的資金曲線（每筆部位獨立，報酬已扣費用）。"""
    p = picks[picks.engine == engine]
    if p.empty:
        return pd.Series(dtype=float)
    daily = p.groupby("date").net_ret.mean()          # 當日開的倉位平均報酬
    return (1 + daily / max(1, (top_n or C.TOP_N))).cumprod()  # 資金分散在 N 檔


def summarize(report, picks, horizon):
    out = [f"\n=== Walk-forward 回測  H={horizon}d  毛門檻 {C.raw_target()*100:.2f}% "
           f"(淨 {C.NET_TARGET*100:.1f}%)  停損 {C.STOP_LOSS*100:.1f}% ===",
           f"{len(report)} 折 | 測試期 {report.test_start.iloc[0]} ~ {report.test_end.iloc[-1]}",
           f"基礎機率 base rate: {report.base.mean()*100:.2f}%  "
           f"（隨機挑一檔，{horizon} 天內先觸停利的機率）\n"]

    out.append(f"{'引擎':<16}{'precision':>10}{'全市場lift':>12}{'同波動lift':>12}"
               f"{'picks波動分層':>14}{'淨報酬':>9}{'最大回撤':>10}")
    for eng, label in ENGINES:
        eq = equity(picks, eng)
        mdd = (eq / eq.cummax() - 1).min() if len(eq) else np.nan
        out.append(f"{label:<16}{report[f'{eng}_prec'].mean()*100:>9.2f}%"
                   f"{report[f'{eng}_lift'].mean():>11.2f}x"
                   f"{report[f'{eng}_vlift'].mean():>11.2f}x"
                   f"{report[f'{eng}_vq'].mean():>13.1f}"
                   f"{report[f'{eng}_ret'].mean()*100:>+8.2f}%"
                   f"{mdd*100:>9.1f}%")
    out.append("  同波動lift = 跟「同一天、同波動率分層」的股票比。這才是真正的選股能力，"
               "\n  全市場lift 會把「專挑高波動股」也算成本事。picks波動分層 0=最低波動 9=最高。")

    out.append("\n逐折 同波動lift（檢查是不是只靠某一段行情撐起來）:")
    out.append(report[["test_start", "base"] + [f"{e}_vlift" for e, _ in ENGINES]]
               .round(3).to_string(index=False))

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
