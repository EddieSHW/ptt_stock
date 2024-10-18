# PTT 股票版分析與視覺化

## 專案簡介

本專案旨在分析 PTT 股票版的發文內容，並將其與台灣加權指數進行視覺化比較。通過使用 Dash 框架，建立了一個互動式的網頁應用程式，展示了股票討論熱詞、共現網路以及主題模型的結果。

## 功能特點

1. 台灣加權指數 K 線圖
2. 特定日期的熱門詞彙頻率分析
3. 詞彙共現網路視覺化
4. 主題模型分析

## 使用技術

- Dash
- Pandas
- Plotly
- NetworkX
- Gensim (用於主題模型)
- pyLDAvis (用於主題模型視覺化)

## 安裝與執行

1. 克隆此儲存庫：

   ```bash
   git clone https://github.com/EddieSHW/ptt_stock.git
   ```

2. 安裝所需依賴套件：

   ```bash
   pip3 install -r requirements.txt
   ```

3. 執行應用程式：

   ```bash
   python3 app.py
   ```

4. 在瀏覽器中打開 `http://localhost:8050/` 查看應用程式。

## 使用說明

1. 在頁面頂部的下拉式選單中選擇日期。
2. 查看該日期的詞頻分析、共現網路和主題模型結果。
3. 使用台灣加權指數圖表的時間範圍選擇器來調整視圖。
