"""資料層：universe 選取、歷史價抓取、SQLite 存取。"""
import os
import sqlite3
import time
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import config as C

TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices(
  symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY(symbol, date));
CREATE TABLE IF NOT EXISTS meta(
  symbol TEXT PRIMARY KEY, code TEXT, name TEXT, market TEXT);
CREATE TABLE IF NOT EXISTS signals(
  date TEXT, symbol TEXT, horizon INT, model TEXT, score REAL, rank INT,
  close REAL, target_price REAL, stop_price REAL,
  PRIMARY KEY(date, symbol, horizon, model));
"""


def db():
    os.makedirs(C.DATA_DIR, exist_ok=True)
    con = sqlite3.connect(C.DB_PATH)
    con.executescript(SCHEMA)
    return con


def _tradable(code):
    """只留一般個股（4 碼數字）與 ETF（00 開頭），濾掉權證、特別股、存託憑證。"""
    return (len(code) == 4 and code.isdigit()) or (code.startswith("00") and code[2:].isdigit())


def _get_json(url, tries=3):
    """官方 OpenAPI 偶爾傳到一半斷線（IncompleteRead），10/5、10/7 的 cron 都因此整條中斷。"""
    for i in range(tries):
        try:
            return requests.get(url, timeout=30).json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(10)


def fetch_universe():
    """打官方 OpenAPI 取當日全市場成交金額，回傳 DataFrame[symbol, code, name, market, turnover]."""
    rows = []
    for url, market, suffix, cc, nc, tc in [
        (TWSE_URL, "TWSE", ".TW", "Code", "Name", "TradeValue"),
        (TPEX_URL, "TPEX", ".TWO", "SecuritiesCompanyCode", "CompanyName", "TransactionAmount"),
    ]:
        for r in _get_json(url):
            code = r[cc].strip()
            if not _tradable(code):
                continue
            try:
                turnover = float(str(r[tc]).replace(",", ""))
            except ValueError:
                continue
            rows.append((code + suffix, code, r[nc].strip(), market, turnover))
    return pd.DataFrame(rows, columns=["symbol", "code", "name", "market", "turnover"])


def watchlist():
    """data/watchlist.txt，一行一個代號（2330 或 2330.TW 都吃）。"""
    if not os.path.exists(C.WATCHLIST):
        return []
    out = []
    for line in open(C.WATCHLIST, encoding="utf-8"):
        s = line.split("#")[0].strip()
        if s:
            out.append(s)
    return out


def select_universe(con):
    """自選股 ∪ 成交金額排行前 N。有歷史資料時用 20 日均額排名，否則用當日。"""
    uni = fetch_universe()
    hist = pd.read_sql(
        "SELECT symbol, AVG(close*volume) t FROM prices "
        "WHERE date > date((SELECT MAX(date) FROM prices), '-40 day') "
        "GROUP BY symbol", con)
    if len(hist) > 50:  # 有足夠歷史才改用 20 日均額
        uni = uni.drop(columns="turnover").merge(hist.rename(columns={"t": "turnover"}), on="symbol", how="left")
        uni["turnover"] = uni["turnover"].fillna(0)

    uni.to_sql("meta", con, if_exists="replace", index=False)  # 名稱表順手更新
    con.commit()

    top = uni[uni.turnover >= C.MIN_TURNOVER].nlargest(C.UNIVERSE_N, "turnover")
    wl = watchlist()
    want = set(wl) | {w.split(".")[0] for w in wl}
    picks = uni[uni.code.isin(want) | uni.symbol.isin(want)]
    return pd.concat([top, picks]).drop_duplicates("symbol").reset_index(drop=True)


def fetch_prices(con, symbols):
    """yfinance 批次下載，增量補到最新。"""
    import yfinance as yf

    last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    start = (pd.Timestamp(last) - pd.Timedelta(days=7)).date() if last else \
        dt.date.today() - dt.timedelta(days=365 * C.HISTORY_YEARS)

    # 盤中跑的話今天的 K 棒還沒收完，寫進去 daily 會拿半根 K 棒發訊號（end 不含當天）
    now = dt.datetime.now(ZoneInfo("Asia/Taipei"))
    end = now.date() if now.time() < dt.time(14, 30) else None

    syms = list(dict.fromkeys(list(symbols) + [C.BENCHMARK]))
    raw = yf.download(syms, start=start, end=end, group_by="ticker", threads=True,
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise SystemExit("yfinance 沒回任何資料，檢查網路或代號")

    frames = []
    for s in syms:
        try:
            d = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        d = d.dropna(subset=["Close"])
        if d.empty:
            continue
        frames.append(pd.DataFrame({
            "symbol": s, "date": d.index.strftime("%Y-%m-%d"),
            "open": d.Open.values, "high": d.High.values, "low": d.Low.values,
            "close": d.Close.values, "volume": d.Volume.values,
        }))

    df = pd.concat(frames, ignore_index=True)
    con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)",
                    df.itertuples(index=False, name=None))
    con.commit()
    return len(df), df.symbol.nunique()


def load_prices(con, symbols=None):
    q = "SELECT * FROM prices"
    if symbols is not None:
        q += " WHERE symbol IN (%s)" % ",".join("?" * len(symbols))
    df = pd.read_sql(q, con, params=list(symbols) if symbols is not None else None,
                     parse_dates=["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def names(con):
    return dict(con.execute("SELECT symbol, name FROM meta").fetchall())
