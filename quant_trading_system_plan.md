# Quant Trading System – Final Consolidated Plan

---

# 🎯 核心定位

本系統目標不是預測市場，而是建立一套：

- 可回測
- 可紀律執行
- 可持續優化
- 具風險控制

的「量化交易決策輔助系統」。

你可承受風險，資金為多餘資金，因此策略允許：

- 中高波動
- ETF + 個股並行
- Alpha 驅動
- 嚴格風控但不過度保守

---

# 🧠 核心策略架構

系統分為兩層：

## 第一層：預測層（Prediction Layer）

目標預測：

```
Alpha = stock_return_60d - index_return_60d
```

模型任務：

- 分類：Alpha > 0 ?
- 或回歸：預測 Alpha 數值

模型僅負責：

> 提供統計優勢

不直接產生買賣點。

---

## 第二層：決策層（Decision Layer）

規則型策略控制：

進場條件：

- 預測 Alpha > 設定門檻（如 2%）
- 預測機率 > 0.65
- 價格結構符合條件（如突破均線）

出場條件：

- +5% 停利
- 或 60 天時間停損
- 或 相對強弱轉負
- 或 結構破壞

---

# 🗄 資料庫設計（Final Version）

## stocks
- id
- symbol
- name
- industry

## daily_prices
- stock_id
- date
- open
- high
- low
- close
- volume

## index_prices
- date
- close

## fundamentals
- stock_id
- quarter
- revenue
- eps

## features_snapshot
- stock_id
- date
- feature_json

## model_predictions
- stock_id
- date
- predicted_alpha
- probability
- model_version

## positions
- id
- stock_id
- entry_price
- entry_date
- quantity
- target_price
- stop_price
- max_holding_days
- status

## signals
- stock_id
- signal_type (buy/sell/watch)
- reason
- confidence
- created_at
- expire_at

## trades
- stock_id
- entry_date
- exit_date
- entry_price
- exit_price
- pnl
- holding_days

---

# 📊 ETF + 個股配置策略

建議初始版本：

- 50% ETF（降低尾部風險）
- 50% 個股 Alpha

後續可根據回測動態調整：

- 市場多頭 → 提高個股比例
- 市場空頭 → 提高 ETF 或降低總倉位

---

# 📈 回測系統規範

必須包含：

- 時間序列切割（避免資料洩漏）
- 扣除交易成本
- 滑價模擬
- 最大回撤計算
- 年化報酬
- Sharpe ratio
- 勝率
- 資金曲線

採用 Walk-forward 機制：

- 滾動訓練
- 固定測試期間
- 不重疊資料

---

# 🧩 執行流程 Pipeline

```
資料抓取
   ↓
特徵工程
   ↓
模型預測 Alpha
   ↓
Ranking 排序
   ↓
規則過濾
   ↓
產生 Buy/Sell Signals
   ↓
倉位管理
   ↓
風險監控
```

---

# ⚖ 風險控制核心原則

1. 不無限攤平
2. 必設時間停損
3. 控制單一持股比例
4. 控制同時持股數
5. 設最大回撤警戒線

---

# 🚀 發展階段規劃

## Phase 1（規則版）
- 不使用 ML
- 建立相對強弱排序
- 固定停利停損
- 建立完整回測

## Phase 2（加入模型）
- 加入分類模型
- 預測 Alpha
- 加入 Ranking 系統

## Phase 3（優化）
- 動態資金配置
- 市場 Regime 判斷
- 模型穩定性監控

---

# 🧭 策略哲學

- 模型提供機率優勢
- 規則負責風險控制
- 不追求神準
- 追求長期統計優勢

---

# 📌 最終目標

建立一套：

- 自動提示買賣條件
- 紀律化執行
- 可回測驗證
- 可持續優化

的量化交易決策系統。

成功標準不是單筆準確，
而是長期資金曲線穩定向上。

