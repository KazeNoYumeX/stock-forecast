"""籌碼面：三大法人買賣超（股）、融資融券餘額（張）。

兩條取得路徑，分工不同：

* **FinMind**（`backfill_finmind`）— 一檔一次請求就給全歷史，六年份 301 檔只要 602 次。
  用於一次性歷史回補。
* **官方 OpenAPI**（`backfill`）— 一次請求給全市場當日，但只能按日期查。
  用於每日增量，一天 4 次請求，不會踩到證交所的反爬。

兩者已交叉驗證：2330 與 6488 的融資餘額、融券餘額、外資與投信買賣超四個數字完全一致。

**不要用官方 API 做密集歷史回補**：連續數百次請求後證交所會回 HTTP 307 轉址到
「因為安全性考量，您所執行的頁面無法呈現」，而且 requests 預設會跟隨轉址拿到 200，
錯誤會被靜默吞掉。`_get` 現在關掉轉址並把 3xx 當成被擋。

欄位對應是最容易安靜出錯的地方：TWSE 用欄名比對（撐得住 2022 年「外資」改名
「外陸資」那種事），TPEx 欄名全部重複只能用位置，所以附帶欄數斷言，schema 一變就大聲壞掉。
"""
import sqlite3
import time
import threading

import pandas as pd
import requests

import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS chips(
  symbol TEXT, date TEXT,
  foreign_net REAL, trust_net REAL, inst_net REAL,   -- 股
  margin_bal REAL, short_bal REAL,                   -- 張
  PRIMARY KEY(symbol, date));
"""

UA = {"User-Agent": "Mozilla/5.0"}
GAP = 0.3  # 實測 TWSE/TPEx 在此速率無封鎖


def _num(x):
    x = str(x).replace(",", "").strip()
    if x in ("", "--", "---", "-", "N/A"):
        return 0.0
    try:
        return float(x)
    except ValueError:
        return 0.0


def _get(sess, url, **params):
    """官方站台用。關掉轉址：證交所擋爬蟲時回 307 導到安全性警告頁，
    跟隨轉址會拿到 200 + HTML，錯誤就被靜默吞掉了。"""
    for attempt in range(4):
        try:
            r = sess.get(url, params=params, timeout=30, allow_redirects=False)
            if r.status_code in (301, 302, 303, 307, 308):
                raise BlockedError("證交所擋爬蟲（HTTP %d）；歷史回補請改用 FinMind"
                                   % r.status_code)
            if r.ok:
                return r.json()
        except BlockedError:
            raise
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


class BlockedError(RuntimeError):
    pass


# --- FinMind（歷史回補）-------------------------------------------------

FINMIND = "https://api.finmindtrade.com/api/v4/data"
FOREIGN = {"Foreign_Investor", "Foreign_Dealer_Self"}   # 市場慣稱的「外資」= 兩者合計


def _finmind(sess, dataset, code, start, token):
    p = dict(dataset=dataset, data_id=code, start_date=start, end_date="2100-01-01")
    if token:
        p["token"] = token
    for attempt in range(4):
        try:
            r = sess.get(FINMIND, params=p, timeout=60)
            if r.status_code == 402:
                raise BlockedError("FinMind 免費額度用完；等一小時，或設 FINMIND_TOKEN 環境變數")
            if r.ok and r.json().get("msg") == "success":
                return r.json().get("data", [])
        except BlockedError:
            raise
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return []


def pending(con, symbols, cover=0.95):
    """還沒補齊的股票。免費額度一次吃不完六年 x 300 檔，所以必須可以續傳。"""
    total = con.execute("SELECT COUNT(DISTINCT date) FROM prices").fetchone()[0]
    have = dict(con.execute(
        "SELECT symbol, COUNT(DISTINCT date) FROM chips GROUP BY symbol").fetchall())
    return [s for s in sorted(symbols) if have.get(s, 0) < total * cover]


def backfill_finmind(con, symbols, start, token=None):
    """一檔一次請求拿全歷史。兩個市場同一條路徑，每檔寫完就 commit（可續傳）。

    免費額度用完不是錯誤，是正常的中斷點：印出剩餘檔數後乾淨退出，下次再跑會接著補。
    """
    con.executescript(SCHEMA)
    sess = requests.Session()
    sess.headers.update(UA)
    todo = pending(con, symbols)
    print(f"  待補 {len(todo)}/{len(symbols)} 檔", flush=True)
    total = 0
    for i, sym in enumerate(todo, 1):
        code = sym.split(".")[0]
        agg = {}
        try:
            for r in _finmind(sess, "TaiwanStockInstitutionalInvestorsBuySell", code, start, token):
                net = _num(r["buy"]) - _num(r["sell"])
                a = agg.setdefault(r["date"], [0.0, 0.0, 0.0])
                if r["name"] in FOREIGN:
                    a[0] += net
                if r["name"] == "Investment_Trust":
                    a[1] += net
                a[2] += net                               # 三大法人合計
            marg = {r["date"]: (_num(r["MarginPurchaseTodayBalance"]),
                                _num(r["ShortSaleTodayBalance"]))
                    for r in _finmind(sess, "TaiwanStockMarginPurchaseShortSale",
                                      code, start, token)}
        except BlockedError as e:
            print(f"  {e}\n  已補 {i-1} 檔，還剩 {len(todo)-i+1} 檔；"
                  f"重跑同一個指令會從中斷處接續。", flush=True)
            return total

        rows = [(sym, d, *(agg.get(d) or (None, None, None)), *(marg.get(d) or (None, None)))
                for d in set(agg) | set(marg)]
        con.executemany("INSERT OR REPLACE INTO chips VALUES (?,?,?,?,?,?,?)", rows)
        con.commit()
        total += len(rows)
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} 檔，{total:,} 筆（最新 {sym}）", flush=True)
    return total


# --- TWSE ---------------------------------------------------------------

def twse_inst(sess, d):
    """三大法人買賣超。欄名比對：含「買賣超」再依「外 / 投信 / 三大法人」分類。"""
    j = _get(sess, "https://www.twse.com.tw/rwd/zh/fund/T86",
             date=d.strftime("%Y%m%d"), selectType="ALLBUT0999", response="json")
    if not j or j.get("stat") != "OK":
        return {}
    f = j["fields"]
    net = [i for i, x in enumerate(f) if "買賣超" in x]
    fgn = [i for i in net if "外" in f[i]]                      # 外陸資 + 外資自營商
    tru = [i for i in net if "投信" in f[i]]
    tot = [i for i in net if "三大法人" in f[i]]
    if not (fgn and tru and tot):
        raise RuntimeError(f"TWSE T86 欄位對不上 {d}: {f}")
    return {r[0].strip(): (sum(_num(r[i]) for i in fgn), _num(r[tru[0]]), _num(r[tot[0]]))
            for r in j["data"]}


def twse_margin(sess, d):
    """融資融券餘額。欄名全重複，位置 6=融資今日餘額、12=融券今日餘額。"""
    j = _get(sess, "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN",
             date=d.strftime("%Y%m%d"), selectType="ALL", response="json")
    if not j or j.get("stat") != "OK":
        return {}
    for t in j.get("tables", []):
        f = t.get("fields", [])
        if len(f) == 16 and f[6] == "今日餘額" and f[12] == "今日餘額":
            return {r[0].strip(): (_num(r[6]), _num(r[12])) for r in t["data"]}
    raise RuntimeError(f"TWSE MI_MARGN 欄位對不上 {d}")


# --- TPEx ---------------------------------------------------------------

def tpex_inst(sess, d):
    """位置對應（欄名全重複）：10=外資合計、13=投信、23=三大法人合計。"""
    j = _get(sess, "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",
             type="Daily", sect="EW", date=d.strftime("%Y/%m/%d"), response="json")
    if not j or not j.get("tables"):
        return {}
    t = j["tables"][0]
    if len(t["fields"]) != 24:
        raise RuntimeError(f"TPEx insti 欄數變成 {len(t['fields'])} @ {d}")
    return {r[0].strip(): (_num(r[10]), _num(r[13]), _num(r[23])) for r in t["data"]}


def tpex_margin(sess, d):
    """位置對應：6=資餘額(張)、14=券餘額(張)。"""
    j = _get(sess, "https://www.tpex.org.tw/www/zh-tw/margin/balance",
             date=d.strftime("%Y/%m/%d"), response="json")
    if not j or not j.get("tables"):
        return {}
    t = j["tables"][0]
    if len(t["fields"]) != 20:
        raise RuntimeError(f"TPEx margin 欄數變成 {len(t['fields'])} @ {d}")
    return {r[0].strip(): (_num(r[6]), _num(r[14])) for r in t["data"]}


# --- 回補 ----------------------------------------------------------------

WORKERS = 3   # 每個交易所的併發數；實測官方站台在此速率不封鎖
CHUNK = 40    # 每 N 天存檔一次，中途掛掉不會全部重來


def _pull(market, suffix, inst_fn, margin_fn, dates, done, errs):
    sess = requests.Session()
    sess.headers.update(UA)
    con = sqlite3.connect(C.DB_PATH, timeout=120)   # 每條執行緒自己的連線
    con.execute("PRAGMA busy_timeout=120000")
    buf = []
    for n, d in enumerate(dates, 1):
        try:
            inst, marg = inst_fn(sess, d), margin_fn(sess, d)
        except Exception as e:
            errs.append(f"{market} {d.date()}: {e}")
            inst = marg = {}
        ds = d.strftime("%Y-%m-%d")
        for code in set(inst) | set(marg):
            f, t, i = inst.get(code, (None, None, None))
            m, sh = marg.get(code, (None, None))
            buf.append((code + suffix, ds, f, t, i, m, sh))
        if n % CHUNK == 0 or n == len(dates):
            try:
                con.executemany("INSERT OR REPLACE INTO chips VALUES (?,?,?,?,?,?,?)", buf)
                con.commit()
            except Exception as e:          # 不讓寫入失敗靜默殺掉整條執行緒
                errs.append(f"{market} 寫入失敗 @{ds}: {e}")
                continue
            done[0] += len(buf)
            buf = []
            print(f"  [{market}] {done[0]:,} 筆 (最新 {ds}, {n}/{len(dates)})", flush=True)
        time.sleep(GAP)
    con.close()


JOBS = [("TWSE", ".TW", "%.TW", twse_inst, twse_margin),
        ("TPEX", ".TWO", "%.TWO", tpex_inst, tpex_margin)]


def done_dates(con, pat):
    """股票池八成以上有融資餘額的日子，才算補完（理由見 backfill）。"""
    return {r[0] for r in con.execute(
        "SELECT p.date FROM prices p LEFT JOIN chips c ON c.symbol = p.symbol "
        "AND c.date = p.date AND c.margin_bal IS NOT NULL WHERE p.symbol LIKE ? "
        "GROUP BY p.date HAVING COUNT(c.symbol) >= 0.8 * COUNT(*)", (pat,))}


def backfill(con, dates, markets=None):
    """每個交易所 x WORKERS 條執行緒，日期用 stride 切開，各自寫入並分段 commit。

    續傳按「市場」分別判斷：只看 chips 表有沒有那天，會把一邊補完的日子
    誤當成另一邊也補完了。「補完」＝股票池八成以上有融資餘額：
    - 融資券比法人晚公布（約 21:00），傍晚跑只拿到法人，不能算補完，否則永遠是空的
    - FinMind 回補只寫部分股票，有幾筆不代表那天全市場都有了
    門檻用股票池而非全市場，因為 TWSE 的歷史本來就只有 FinMind 補的股票池。
    """
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.commit()

    errs, done, ts = [], [0], []
    for market, suffix, pat, inst_fn, margin_fn in JOBS:
        if markets and market not in markets:
            continue
        have = done_dates(con, pat)
        todo = [d for d in dates if d.strftime("%Y-%m-%d") not in have]
        print(f"  {market}: 待補 {len(todo)}/{len(dates)} 天", flush=True)
        for i in range(WORKERS):
            if len(todo[i::WORKERS]):
                ts.append(threading.Thread(
                    target=_pull,
                    args=(market, suffix, inst_fn, margin_fn, todo[i::WORKERS], done, errs)))
    [t.start() for t in ts]
    [t.join() for t in ts]
    return done[0], errs


def load(con):
    """只載入股票池內的籌碼（chips 表存了全市場，全載會吃掉數百 MB）。"""
    return pd.read_sql(
        "SELECT * FROM chips WHERE symbol IN (SELECT DISTINCT symbol FROM prices)",
        con, parse_dates=["date"])
