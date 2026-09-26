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
# 可用環境變數覆寫：縮短訓練期能讓測試折涵蓋更早的期間（例如 2022 年空頭），
# 代價是模型看到的歷史較少。
TRAIN_DAYS = int(os.getenv("TRAIN_DAYS", 504))   # ~2 年
TEST_DAYS = int(os.getenv("TEST_DAYS", 63))      # ~1 季

# 並行追蹤的版本。往前走的真實訊號是唯一沒被參數掃描污染的資料，
# 所以兩個版本同時發訊號、分別結算，讓市場決定誰對。
# (名稱, 持有交易日, 淨停利, 停損, 檔數)
#
# 兩組都來自 18 格搜尋（持有期 x 停利 x 停損），錄取條件事前寫死在 backtest.RULES：
# 兩半年化都為正、兩半差距 < 2x、贏隨機 > 10pp、逐折 std < 0.30。
PROFILES = [
    # 使用者版：停利固定 5%。六格全部落選（最好的一格只贏隨機 4.9pp），
    # 仍照使用者要求追蹤 —— 往前走的真實訊號是唯一沒被參數掃描污染的資料。
    ("h30t05s20", 30, 0.05, 0.20, 20),
    # 推薦版：18 格中唯一同時通過四條的最佳格。年化 +45.0%、回撤 -32.1%、
    # 贏隨機 +26pp。但仍小輸 0050 買進持有（+47.8% / -28.5%）。
    ("h30t20s20", 30, 0.20, 0.20, 20),
]

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
