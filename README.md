# Simple Logistic Regression for Bitcoin Direction

這個專案示範如何以極簡的流程下載（或載入預先儲存的）Binance BTC/USDT 歷史資料，
計算常見技術指標作為特徵，並使用 Logistic Regression 來預測下一期價格是否上漲，
最後透過簡易的多空切換策略進行回測與視覺化。

## 專案架構

```text
simple-logistic-regression/
├── data/                       # 預先下載或準備好的 Binance K 線資料
├── fetch_data.py               # 從 Binance API 抓取資料並寫入 CSV
├── notebooks/
│   └── bitcoin_logistic_regression.ipynb  # 主要分析流程
├── requirements.txt            # 需要的 Python 套件
└── README.md
```

### 流程圖

```mermaid
flowchart LR
    A[Binance API<br/>fetch_data.py] -->|K 線資料| B[data/btcusdt_1h.csv]
    B --> C[特徵工程<br/>Momentum / SMA / Volatility / RSI / Volume Z-Score]
    C --> D[Logistic Regression]
    D --> E[機率預測]
    E --> F[回測 & 視覺化]
```

## 使用方式

1. 安裝套件：
   ```bash
   pip install -r requirements.txt
   ```
2. （可選）若環境可以連線 Binance，執行 `fetch_data.py` 取得最新資料：
   ```bash
   python fetch_data.py --symbol BTCUSDT --interval 1h \
     --start 2023-01-01T00:00:00 --end 2023-06-01T00:00:00
   ```
   > 提示：在無法連線的環境下，專案已附上一份離線範例 `data/btcusdt_1h.csv` 供 Notebook 使用。
3. 啟動 Jupyter Lab 或 Notebook，開啟 `notebooks/bitcoin_logistic_regression.ipynb`，依序執行每個區塊。

## Notebook 內容摘要

- **資料載入**：讀取 `data/btcusdt_1h.csv`，若檔案不存在會呼叫 `fetch_data.py` 嘗試從 Binance API 下載。
- **特徵工程**：計算 Momentum、10 期 SMA、10 期波動度、14 期 RSI 與成交量 Z-score。
- **模型訓練**：以時間序列 70% 資料作為訓練集，其餘作為測試集，訓練 scikit-learn 的 Logistic Regression。
- **回測**：當模型預測上漲機率 > 0.5 時做多，否則做空，並與單純買入持有的績效做比較。
- **視覺化**：繪製策略與買入持有的累積報酬曲線。

## 延伸方向

- 加入更多技術指標（MACD、布林通道等）或基本面因子。
- 嘗試不同的時間週期（例如 15 分或 4 小時 K 線）。
- 納入交易成本與風險管理機制，提升回測的真實性。
- 以更進階的模型（如 Gradient Boosting、LSTM）比較 Logistic Regression 的表現。
