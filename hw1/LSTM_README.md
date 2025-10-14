# LSTM 語言模型 - 學習曲線與評估功能

## 新增功能

### 1. 學習曲線繪製 (Learning Curves)
- **訓練損失曲線**: 顯示每個 epoch 的訓練損失變化
- **訓練準確率曲線**: 顯示每個 epoch 的訓練準確率變化
- 自動保存圖表為 `lstm_learning_curves.png`

### 2. 測試數據評估 (Test Data Evaluation)
- **測試損失 (Test Loss)**: 模型在測試數據上的交叉熵損失
- **測試準確率 (Test Accuracy)**: 下一個詞預測的準確率
- **測試困惑度 (Test Perplexity)**: exp(test_loss)，越低表示模型越好

### 3. LSTM 特殊功能
- **長短期記憶**: 相比 RNN，LSTM 能更好地處理長期依賴關係
- **門控機制**: 包含遺忘門、輸入門和輸出門，解決梯度消失問題
- **細胞狀態**: 維護長期記憶，提升序列建模能力

## LSTM vs RNN 主要差異

### 1. 架構差異
```python
# RNN 隱藏狀態初始化
def init_hidden(self, batch_size, device):
    return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

# LSTM 隱藏狀態和細胞狀態初始化
def init_hidden(self, batch_size, device):
    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
    return (h0, c0)
```

### 2. 網路層選擇
```python
# RNN 使用標準 RNN 層
self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, ...)

# LSTM 使用 LSTM 層
self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, ...)
```

## 使用方法

### 安裝依賴
```bash
pip install -r requirements.txt
```

### 執行訓練和評估
```bash
python LSTM.py
```

## 輸出結果

### 1. 學習曲線圖表
- 文件名: `lstm_learning_curves.png`
- 標題: "LSTM Training Loss/Accuracy Curve"
- 包含兩個子圖：訓練損失和訓練準確率

### 2. 測試評估結果
```
LSTM 模型測試結果
============================================================
測試損失 (Test Loss): X.XXXX
測試準確率 (Test Accuracy): X.XXXX (XX.XX%)
測試困惑度 (Test Perplexity): X.XXXX
============================================================
```

### 3. 模型保存
- 文件名: `lstm_model.pth`
- 包含: 模型權重、詞彙表、超參數

## 性能預期

### LSTM 相比 RNN 的優勢
1. **更好的長期記憶**: 能記住更久遠的信息
2. **梯度穩定性**: 減少梯度消失問題
3. **更高準確率**: 通常能達到更好的預測準確率
4. **更低困惑度**: 表示模型不確定性更小

### 訓練時間
- LSTM 比 RNN 計算量稍大
- 每個 epoch 時間可能增加 20-30%
- 但通常能以更少的 epoch 達到更好效果

## 超參數配置

```python
HIDDEN_DIM = 128      # 隱藏層維度
NUM_LAYERS = 2        # LSTM 層數
LEARNING_RATE = 0.001 # 學習率
NUM_EPOCHS = 10       # 訓練輪數
BATCH_SIZE = 32       # 批次大小
SEQ_LENGTH = 50       # 序列長度
```

## 文件結構
```
hw1/
├── LSTM.py                    # LSTM 主程序
├── RNN.py                     # RNN 主程序
├── requirements.txt           # 依賴庫
├── train.txt                 # 訓練數據
├── test.txt                  # 測試數據
├── incomplete.txt            # 不完整句子數據
├── lstm_model.pth           # LSTM 模型
├── rnn_model.pth            # RNN 模型
├── lstm_learning_curves.png # LSTM 學習曲線
└── rnn_learning_curves.png  # RNN 學習曲線
```

## 模型比較建議

執行兩個模型後，可以比較：
1. **訓練速度**: RNN vs LSTM
2. **最終準確率**: 測試準確率對比
3. **困惑度**: 越低越好
4. **學習曲線**: 收斂速度和穩定性
5. **文本生成質量**: 生成的文本流暢度

## 注意事項
- LSTM 記憶體使用量比 RNN 稍高
- 建議先執行 RNN 再執行 LSTM 進行比較
- 兩個模型使用相同的超參數以確保公平比較