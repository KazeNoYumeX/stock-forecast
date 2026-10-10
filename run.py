#!/usr/bin/env python3
"""台股淨成長 5% 預測系統 CLI。"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import backtest as B
import chips as CH
import config as C
import data as D
import model as M

pd.set_option("display.width", 200, "display.unicode.east_asian_width", True)


def _features(con, asof=None):
    """讀 DB -> 特徵（未標籤）。回傳 (特徵表, 大盤 close)。asof：只用到該日為止的資料（補跑漏掉的日子）。"""
    df = D.load_prices(con)
    if asof:
        df = df[df.date <= asof]
    bench = df[df.symbol == C.BENCHMARK].set_index("date").close
    if bench.empty:
        sys.exit("缺少大盤資料，先跑 `run.py fetch`")
    con.executescript(CH.SCHEMA)
    ch = CH.load(con)
    if asof:
        ch = ch[ch.date <= asof]
    return M.add_chips(M.build_features(df[df.symbol != C.BENCHMARK], bench), ch), bench


def cmd_fetch(args):
    con = D.db()
    uni = D.select_universe(con)
    print(f"universe {len(uni)} 檔（自選 {len(D.watchlist())}、成交額前 {C.UNIVERSE_N}）")
    rows, syms = D.fetch_prices(con, uni.symbol)
    span = con.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM prices").fetchone()
    print(f"寫入 {rows} 筆 / {syms} 檔；DB 現有 {span[2]} 筆，{span[0]} ~ {span[1]}")


def cmd_chips(args):
    con = D.db()
    con.executescript(CH.SCHEMA)
    dates = [pd.Timestamp(r[0]) for r in
             con.execute("SELECT DISTINCT date FROM prices ORDER BY date")]
    print(f"交易日 {len(dates)} 天：{dates[0].date()} ~ {dates[-1].date()}")

    if args.source == "finmind":
        syms = [r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM prices WHERE symbol NOT LIKE '^%'")]
        n = CH.backfill_finmind(con, syms, str(dates[0].date()),
                                os.getenv("FINMIND_TOKEN"))
        errs = []
    else:
        n, errs = CH.backfill(con, dates, markets=args.market)
    print(f"寫入 {n} 筆籌碼資料"
          f"（{con.execute('SELECT COUNT(DISTINCT date) FROM chips').fetchone()[0]} 個交易日）")
    if errs:
        print(f"{len(errs)} 天失敗：" + "; ".join(errs[:5]))


def cmd_backtest(args):
    con = D.db()
    for h in ([args.horizon] if args.horizon else C.HORIZONS):
        df = M.label(_features(con)[0], h)
        report, picks = B.run(df, h)
        if report.empty:
            sys.exit(f"資料不足以切出任何一折（需要 > {C.TRAIN_DAYS + C.TEST_DAYS + h} 個交易日）")
        print(B.summarize(report, picks, h))


def _one_profile(con, feat, h, name, net, stop, top_n, nm):
    """產生單一版本的候選清單。回傳 (顯示用表格, 寫入 signals 的 rows, 校準文字)。"""
    df = M.label(feat, h, net=net, stop=stop)
    d = M.tradable(df).dropna(subset=M.PRICE_FEATURES).copy()
    today = np.sort(d.date.unique())[-1]

    report, _ = B.run(df, h, top_n=top_n, last_only=True)
    calib = ""
    if not report.empty:
        last = report.iloc[-1]
        cg = B.cagr_of(last.ml_neut_ret, last.ml_neut_hold)
        calib = (f"  最近一折 {last.test_start}~{last.test_end}："
                 f"base {last.base*100:.1f}%　命中 {last.ml_neut_prec*100:.1f}%"
                 f"　同波動 lift {last.ml_neut_vlift:.2f}x　換算年化 {cg*100:+.1f}%")

    train = d[d.date < today - pd.Timedelta(days=2 * h)].dropna(subset=["y"])
    live = d[d.date == today].copy()
    live["ml"] = M.fit_predict(train, live, M.FEATURES)
    live["rule"] = M.rule_score(live)
    live["score"] = M.neutralize(live, "ml") if live.ml.notna().any() else live.rule
    live["vol_q"] = M.vol_bucket(live)
    top = live.sort_values("score", ascending=False).head(top_n).copy()
    top["tp_mult"] = 1 + top.symbol.map(lambda s: C.raw_target(net, etf=C.is_etf(s)))
    top["sl_mult"] = 1 - stop

    out = pd.DataFrame({
        "代號": top.symbol.str.split(".").str[0],
        "名稱": top.symbol.map(nm).fillna(""),
        "基準收盤": top.close.round(2),
        "中性化分": top.score.round(3),
        "波動分層": top.vol_q.astype("Int64"),
        "20日均額(億)": (top.turnover20 / 1e8).round(2),
        "開盤預估低": (top.close * (1 + top.gap_lo)).round(2),
        "開盤預估高": (top.close * (1 + top.gap_hi)).round(2),
        "停利%": ((top.tp_mult - 1) * 100).map("{:+.2f}".format),
        "停損%": (["不停損"] * len(top) if stop >= 0.9
                 else ((top.sl_mult - 1) * 100).map("{:.2f}".format)),
        "一張成本(萬)": (top.close * 1000 / 1e4).round(1),
        "最長持有": f"{h}d",
    }).reset_index(drop=True)
    out.index += 1

    ts = pd.Timestamp(today).date()
    rows = [(str(ts), t.symbol, h, name, float(t.score), i + 1,
             float(t.close), float(t.close * t.tp_mult), float(t.close * t.sl_mult))
            for i, t in enumerate(top.itertuples(index=False))]
    return out, rows, calib, ts, top


def cmd_daily(args):
    con = D.db()
    feat, _ = _features(con, args.asof)
    nm = D.names(con)
    profiles = [p for p in C.PROFILES if not args.profile or p[0] in args.profile]
    if not profiles:
        sys.exit(f"沒有符合的版本，可用：{[p[0] for p in C.PROFILES]}")

    os.makedirs(f"{C.DATA_DIR}/signals", exist_ok=True)
    for name, ph, net, stop, top_n, note in profiles:
        h = args.horizon or ph
        out, rows, calib, ts, top = _one_profile(con, feat, h, name, net, stop, top_n, nm)
        print(f"\n{'=' * 70}")
        print(f"[{name}]  {ts}  H={h}d　停利 淨{net*100:.0f}%"
              f"（毛 {C.raw_target(net)*100:.2f}%）　"
              + ("不停損" if stop >= 0.9 else f"停損 {stop*100:.0f}%")
              + f"　取前 {top_n} 檔")
        print(f"  回測狀態：{note}")
        print('=' * 70)
        print(out.to_string())
        if calib:
            print(calib)
        path = f"{C.DATA_DIR}/signals/{ts}_h{h}_{name}.csv"
        out.to_csv(path, index_label="rank", encoding="utf-8-sig")
        con.executemany("INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        print(f"  -> {path}")

    ex = top.iloc[0]
    exp = float(ex.close * (1 + (ex.gap_lo + ex.gap_hi) / 2))
    print(f"\n進場＝次日開盤市價。「開盤預估低/高」是該檔近 250 日跳空的 80% 範圍，")
    print("只是讓你知道成交價會落在哪 —— 不是進場條件。審核已測過「跳空太多就不買」")
    print("和「掛限價等回檔」，樣本放大後兩者都無效，所以不要自行加條件。")
    print(f"成交後用 `run.py price <代號> <成交價> --profile <版本>` 算停利停損。")
    print("兩個版本都會寫進 signals 表，`run.py verify` 會各自結算後並列比較。")


def cmd_verify(args):
    """用真實價格結算過去發出的訊號，檢驗線上表現是否與回測一致。"""
    con = D.db()
    sig = pd.read_sql("SELECT * FROM signals", con, parse_dates=["date"])
    if sig.empty:
        return print("signals 表是空的，先跑 daily")
    px = D.load_prices(con)
    last = px.date.max()

    out = []
    for s in sig.itertuples(index=False):
        g = px[(px.symbol == s.symbol) & (px.date > s.date)].sort_values("date")
        if g.empty:
            continue
        w = g.iloc[:s.horizon]
        # 用訊號產生當時存下的門檻，不是當下環境變數 —— 兩個版本的停利停損不同
        up_r, dn_r = s.target_price / s.close - 1, 1 - s.stop_price / s.close
        res, net, days = M.settle_one(
            g.open.iloc[0], w.high.values, w.low.values, w.close.values,
            s.horizon, up_r=up_r, dn_r=dn_r, etf=C.is_etf(s.symbol))
        out.append({"date": s.date, "horizon": s.horizon, "model": s.model,
                    "symbol": s.symbol, "rank": s.rank, "entry": g.open.iloc[0],
                    "res": res, "net": net, "days": days})
    v = pd.DataFrame(out)
    if v.empty:
        return print("訊號日之後還沒有新的報價，無法結算")

    names = D.names(con)
    live = {p[0] for p in C.PROFILES}
    print(f"\n訊號結算（價格資料到 {last.date()}）")
    for (dt, h, mdl), g in v.groupby(["date", "horizon", "model"]):
        tag = "" if mdl in live else "　<- 舊實驗，參數已不同，不要跟現行版本混著看"
        done = g[g.res != "open"]
        print(f"\n{dt.date()}  H={h}d  model={mdl}  共 {len(g)} 檔{tag}")
        print(f"  停利 {(g.res=='win').sum()}   停損 {(g.res=='loss').sum()}   "
              f"到期平倉 {(g.res=='expire').sum()}   持有中 {(g.res=='open').sum()}"
              f"（已過 {int(g[g.res=='open'].days.max()) if (g.res=='open').any() else 0} 個交易日）")
        if len(done):
            print(f"  已結算 {len(done)} 檔：命中率 {(done.res=='win').mean()*100:.1f}%  "
                  f"平均淨報酬 {done.net.mean()*100:+.2f}%")

    for mdl, g in v[v.model.isin(live)].groupby("model"):
        t = g.sort_values(["date", "rank"]).assign(
            名稱=g.symbol.map(names).fillna(""), 代號=g.symbol.str.split(".").str[0])
        print(f"\n逐檔明細 [{mdl}]:")
        print(t[["rank", "代號", "名稱", "entry", "res", "net", "days"]]
              .rename(columns={"rank": "#", "entry": "進場價(次日開盤)", "res": "結果",
                               "net": "淨報酬", "days": "已過交易日"})
              .round(4).to_string(index=False))

    lv = v[v.model.isin(live)]
    n_open = int((lv.res == "open").sum())
    print(f"\n現行版本共 {len(lv)} 筆：已結算 {len(lv) - n_open}、持有中 {n_open}"
          f"（另有 {len(v) - len(lv)} 筆舊實驗不計）。")
    print("回測需要約 16,000 筆選股才得出可信的 lift，單日數十檔在統計上等於零資訊。")
    print("這是在驗證結算流程，不是在驗證策略 —— 累積約 1,200 筆前不要據此調參數。")


def cmd_price(args):
    """給實際成交價，算出對應的停利停損。ETF 與個股證交稅不同，手算容易錯。"""
    con = D.db()
    code = args.code.split(".")[0]
    row = con.execute("SELECT symbol, name FROM meta WHERE code = ?", (code,)).fetchone()
    sym, name = row if row else (code, "")
    prof = {p[0]: p for p in C.PROFILES}.get(args.profile or C.PROFILES[0][0])
    if prof is None:
        sys.exit(f"未知版本，可用：{[p[0] for p in C.PROFILES]}")
    _, ph, net, stop, _, _ = prof
    tp, sl = 1 + C.raw_target(net, etf=C.is_etf(sym)), 1 - stop
    h = args.horizon or ph
    print(f"\n[{prof[0]}] 停利 淨{net*100:.0f}% / 停損 {stop*100:.0f}%")
    print(f"\n{code} {name}　成交 {args.fill:.2f}")
    print(f"  停利 {args.fill * tp:.2f} ({(tp - 1) * 100:+.2f}%)   "
          f"停損 {args.fill * sl:.2f} ({(sl - 1) * 100:+.2f}%)   "
          f"最長持有 {h} 個交易日")
    print(f"  {h} 個交易日內兩條線都沒觸到就平倉出場，不要續抱。")


def cmd_audit(args):
    con = D.db()
    feat, bench = _features(con)
    print(B.audit(feat, args.horizon or C.HORIZONS[0],
                  args.targets or [0.05, 0.08, 0.12],
                  args.tops or [1, 2, 3, 5, 10, 20, 40], bench))


def cmd_grid(args):
    con = D.db()
    feat, _ = _features(con)
    hs = args.horizons or [7, 30]
    ts = args.targets or [0.05, 0.12, 0.20]
    ss = args.stops or [0.05, 0.12, 0.20]
    if args.trails:
        print(f"格點 {len(hs) * len(args.trails)} 格"
              f"（移動停損：持有期 {hs} x 回檔 {args.trails}）")
        print(B.grid_report(B.grid(feat, hs, ts, ss, trails=args.trails)))
        return
    print(f"格點 {len(hs) * len(ts) * len(ss)} 格（持有期 {hs} x 停利 {ts} x 停損 {ss}）")
    print(B.grid_report(B.grid(feat, hs, ts, ss)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="更新 universe 與歷史價").set_defaults(fn=cmd_fetch)
    sc = sub.add_parser("chips", help="回補籌碼面（法人買賣超、融資券）")
    sc.add_argument("--source", choices=["finmind", "official"], default="finmind",
                    help="finmind=一檔一請求拿全歷史（回補用）；official=一天一請求拿全市場（每日增量用）")
    sc.add_argument("--market", nargs="*", choices=["TWSE", "TPEX"],
                    help="僅 official 來源適用")
    sc.set_defaults(fn=cmd_chips)
    sub.add_parser("verify", help="用真實價格結算過去的訊號").set_defaults(fn=cmd_verify)
    sp = sub.add_parser("price", help="給成交價，算對應的停利停損")
    sp.add_argument("code")
    sp.add_argument("fill", type=float)
    sp.add_argument("--horizon", type=int)
    sp.add_argument("--profile", help=f"版本，預設 {C.PROFILES[0][0]}")
    sp.set_defaults(fn=cmd_price)
    sg = sub.add_parser("grid", help="三維格點搜尋（持有期 x 停利 x 停損）")
    sg.add_argument("--horizons", type=int, nargs="*")
    sg.add_argument("--targets", type=float, nargs="*")
    sg.add_argument("--stops", type=float, nargs="*")
    sg.add_argument("--trails", type=float, nargs="*", help="移動停損的回檔幅度")
    sg.set_defaults(fn=cmd_grid)
    sa = sub.add_parser("audit", help="穩定性審核（八項查驗）")
    sa.add_argument("--horizon", type=int)
    sa.add_argument("--targets", type=float, nargs="*")
    sa.add_argument("--tops", type=int, nargs="*")
    sa.set_defaults(fn=cmd_audit)
    for name, fn, helptext in [("backtest", cmd_backtest, "walk-forward 回測"),
                               ("daily", cmd_daily, "產生今日候選清單")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--horizon", type=int, help=f"交易日，預設 {C.HORIZONS}")
        if name == "daily":
            s.add_argument("--profile", nargs="*", help="只產指定版本")
            s.add_argument("--asof", help="補跑某個漏掉的交易日（YYYY-MM-DD），只用到該日為止的資料")
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
