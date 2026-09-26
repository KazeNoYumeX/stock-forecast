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


def _panel(con, horizon):
    """讀 DB -> 特徵 -> 標籤。回傳 (帶標籤的長表, 大盤 close)。"""
    df = D.load_prices(con)
    bench = df[df.symbol == C.BENCHMARK].set_index("date").close
    if bench.empty:
        sys.exit("缺少大盤資料，先跑 `run.py fetch`")
    df = df[df.symbol != C.BENCHMARK]
    con.executescript(CH.SCHEMA)
    return M.label(M.add_chips(M.build_features(df, bench), CH.load(con)), horizon)


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
        df = _panel(con, h)
        report, picks = B.run(df, h)
        if report.empty:
            sys.exit(f"資料不足以切出任何一折（需要 > {C.TRAIN_DAYS + C.TEST_DAYS + h} 個交易日）")
        print(B.summarize(report, picks, h))


def cmd_daily(args):
    con = D.db()
    h = args.horizon or C.HORIZONS[-1]
    df = _panel(con, h)
    d = M.tradable(df).dropna(subset=M.PRICE_FEATURES).copy()
    dates = np.sort(d.date.unique())
    today = dates[-1]

    # 用最後一折做校準，讓報表能誠實說出「這份清單歷史命中率是多少」
    calib = ""
    report, _ = B.run(df, h)
    if not report.empty:
        last = report.iloc[-1]
        calib = (f"校準（最近一折 {last.test_start}~{last.test_end}）："
                 f"base {last.base*100:.1f}%  規則 {last.rule_prec*100:.1f}%  "
                 f"ML+籌碼 {last.ml_chip_prec*100:.1f}%  中性化 {last.ml_neut_prec*100:.1f}%"
                 f"（同波動 lift {last.ml_neut_vlift:.2f}x）")

    train = d[d.date < today - pd.Timedelta(days=2 * h)].dropna(subset=["y"])  # embargo
    live = d[d.date == today].copy()
    live["ml"] = M.fit_predict(train, live, M.FEATURES)
    live["rule"] = M.rule_score(live)
    # 中性化後才是誠實的排序：未中性化的分數有 95% 的表面優勢來自「挑波動大的股票」
    live["score"] = M.neutralize(live, "ml") if live.ml.notna().any() else live.rule
    live["vol_q"] = M.vol_bucket(live)
    live = live.sort_values("score", ascending=False)

    nm = D.names(con)
    top = live.head(C.TOP_N).copy()
    tgt = top.symbol.map(lambda s: C.raw_target(etf=C.is_etf(s.split(".")[0])))
    out = pd.DataFrame({
        "代號": top.symbol.str.split(".").str[0],
        "名稱": top.symbol.map(nm).fillna(""),
        "收盤": top.close.round(2),
        "中性化分": top.score.round(3),
        "ML機率": top.ml.round(3),
        "波動分層": top.vol_q.astype("Int64"),
        "20日均額(億)": (top.turnover20 / 1e8).round(2),
        "停利價": (top.close * (1 + tgt)).round(2),
        "停損價": (top.close * (1 - C.STOP_LOSS)).round(2),
        "最長持有": f"{h}d",
    }).reset_index(drop=True)
    out.index += 1

    ts = pd.Timestamp(today).date()
    regime = "多頭" if live.regime.iloc[0] > 0 else "空頭 (TWII < MA60)"
    print(f"\n{ts}  H={h}d  毛門檻 {C.raw_target()*100:.2f}%（淨 {C.NET_TARGET*100:.1f}%，已扣手續費+證交稅+滑價）")
    print(f"market regime: {regime}   候選池 {len(live)} 檔")
    print(out.to_string())
    if calib:
        print("\n" + calib)
    print("這是機率排序，不是保證。停利/停損價請照表執行，超過最長持有天數就出場。")
    print("排序已做波動率中性化：同波動分層內比較，避免只是挑到波動大的股票。")

    os.makedirs(f"{C.DATA_DIR}/signals", exist_ok=True)
    path = f"{C.DATA_DIR}/signals/{ts}_h{h}.csv"
    out.to_csv(path, index_label="rank", encoding="utf-8-sig")
    rows = [(str(ts), r.symbol, h, "ml_neut", float(r.score), i + 1,
             float(r.close), float(r.close * (1 + C.raw_target(etf=C.is_etf(r.symbol.split(".")[0])))),
             float(r.close * (1 - C.STOP_LOSS)))
            for i, r in enumerate(top.itertuples(index=False))]
    con.executemany("INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    print(f"已寫入 {path} 與 signals 表（日後可回頭驗證線上表現）")


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
        res, net, days = M.settle_one(
            g.open.iloc[0], w.high.values, w.low.values, w.close.values,
            s.horizon, etf=C.is_etf(s.symbol))
        out.append({"date": s.date, "horizon": s.horizon, "model": s.model,
                    "symbol": s.symbol, "rank": s.rank, "entry": g.open.iloc[0],
                    "res": res, "net": net, "days": days})
    v = pd.DataFrame(out)
    if v.empty:
        return print("訊號日之後還沒有新的報價，無法結算")

    names = D.names(con)
    print(f"\n訊號結算（價格資料到 {last.date()}）")
    for (dt, h, mdl), g in v.groupby(["date", "horizon", "model"]):
        done = g[g.res != "open"]
        print(f"\n{dt.date()}  H={h}d  model={mdl}  共 {len(g)} 檔")
        print(f"  停利 {(g.res=='win').sum()}   停損 {(g.res=='loss').sum()}   "
              f"到期平倉 {(g.res=='expire').sum()}   持有中 {(g.res=='open').sum()}"
              f"（已過 {int(g[g.res=='open'].days.max()) if (g.res=='open').any() else 0} 個交易日）")
        if len(done):
            print(f"  已結算 {len(done)} 檔：命中率 {(done.res=='win').mean()*100:.1f}%  "
                  f"平均淨報酬 {done.net.mean()*100:+.2f}%")

    top = v[v.model == "ml_neut"].sort_values("rank") if (v.model == "ml_neut").any() else v
    t = top.assign(名稱=top.symbol.map(names).fillna(""),
                   代號=top.symbol.str.split(".").str[0])
    print("\n逐檔明細（中性化引擎）:")
    print(t[["rank", "代號", "名稱", "entry", "res", "net", "days"]]
          .rename(columns={"rank": "#", "entry": "進場價(次日開盤)", "res": "結果",
                           "net": "淨報酬", "days": "已過交易日"})
          .round(4).to_string(index=False))

    n_open = (v.res == "open").sum()
    print(f"\n注意：共 {len(v)} 筆訊號、{n_open} 筆尚未到期。回測用了約 16,000 筆選股"
          f"才得出 lift 1.16x，\n單日 20 檔在統計上沒有意義 —— 這是在驗證結算流程，"
          f"不是在驗證策略。不要據此調參數。")


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
    for name, fn, helptext in [("backtest", cmd_backtest, "walk-forward 回測"),
                               ("daily", cmd_daily, "產生今日候選清單")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--horizon", type=int, help=f"交易日，預設 {C.HORIZONS}")
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
