"""台股預測系統參數。全部可用環境變數覆寫。"""
import os

_env = lambda k, d: float(os.getenv(k, d))

# --- 交易成本（單邊）---
FEE_RATE = 0.001425 * _env("BROKER_DISCOUNT", 1.0)  # 券商手續費，買賣各收一次
TAX_STOCK = 0.003                                   # 個股證交稅，賣出時收
TAX_ETF = 0.001                                     # ETF 證交稅
SLIPPAGE = _env("SLIPPAGE", 0.001)                  # 單邊滑價

# --- 策略 ---
NET_TARGET = _env("NET_TARGET", 0.05)   # 想賺到的淨報酬（扣完所有費用）
STOP_LOSS = _env("STOP_LOSS", 0.05)     # 下界停損（毛跌幅）
HORIZONS = [7, 30]                      # 交易日
TOP_N = int(os.getenv("TOP_N", 20))     # 每日輸出候選數

# --- 股票池 ---
UNIVERSE_N = int(os.getenv("UNIVERSE_N", 300))   # 成交金額排行取前 N
MIN_TURNOVER = _env("MIN_TURNOVER", 5e7)         # 日均成交額下限（避免買不到）
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", 6))

# --- 回測 ---
TRAIN_DAYS = 504   # ~2 年
TEST_DAYS = 63     # ~1 季

DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "stocks.db")
WATCHLIST = os.path.join(DATA_DIR, "watchlist.txt")
BENCHMARK = "^TWII"


def is_etf(symbol: str) -> bool:
    """吃 "0050.TW" 或 "0050" 都可以；ETF 證交稅 0.1%，個股 0.3%。"""
    return symbol.split(".")[0].startswith("00")


def raw_target(net=None, etf=False):
    """把「淨賺 net」反解成需要的毛漲幅。

    net = P_sell*(1-fee-tax-slip) / (P_buy*(1+fee+slip)) - 1
    """
    net = NET_TARGET if net is None else net
    tax = TAX_ETF if etf else TAX_STOCK
    return (1 + net) * (1 + FEE_RATE + SLIPPAGE) / (1 - FEE_RATE - tax - SLIPPAGE) - 1


def net_return(raw, etf=False):
    """毛漲幅 -> 實際入袋淨報酬。raw_target 的反函數。"""
    tax = TAX_ETF if etf else TAX_STOCK
    return (1 + raw) * (1 - FEE_RATE - tax - SLIPPAGE) / (1 + FEE_RATE + SLIPPAGE) - 1
