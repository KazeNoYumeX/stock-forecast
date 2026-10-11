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
# 252 是為了讓測試折涵蓋 2022 空頭（504 會讓第一折從 2023-02 才開始，錄取條件必落選），
# 代價是模型看到的歷史較少。
TRAIN_DAYS = int(os.getenv("TRAIN_DAYS", 252))   # ~1 年
TEST_DAYS = int(os.getenv("TEST_DAYS", 63))      # ~1 季

# 並行追蹤的版本。往前走的真實訊號是唯一沒被參數掃描污染的資料，
# 所以兩個版本同時發訊號、分別結算，讓市場決定誰對。
# (名稱, 持有交易日, 淨停利, 停損, 檔數, 驗證狀態)
#
# 三個版本並行追蹤，結構完全相同（H=60、不停損、前 20 檔），只差停利高度。
# 這樣往前走的真實訊號能回答「停利該訂多高」這個單一問題，不會同時動到多個變數。
#
# 回測數字來自含 2022 空頭的 16 折（TRAIN_DAYS=252）。錄取條件見 backtest.RULES。
# 驗證狀態會印在每日輸出上 —— 不通過的版本必須一直標著，不能靠記憶。
PROFILES = [
    ("h60t05none", 60, 0.05, 0.99, 20,
     "未通過：贏隨機僅 +2.3pp（年化 +10.7%，隨機 +8.4%）。停利天花板太低，"
     "模型的優勢完全被壓住。使用者原始構想，保留觀測用"),
    ("h60t15none", 60, 0.15, 0.99, 20,
     "未通過：贏隨機 +7.3pp 未達 10pp（年化 +25.4%，隨機 +18.1%）。"
     "但為全場最穩定的一格（前半 +25.5%／後半 +25.3%，逐折 std 0.08 最低）"),
    ("h60t20none", 60, 0.20, 0.99, 20,
     "通過全部五條：年化 +31.6%、贏隨機 +11.2pp、lift 1.08x、std 0.10。"
     "注意回撤 -16.8% 被低估，equity() 只計已實現部位"),
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
