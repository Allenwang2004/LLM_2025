#!/usr/bin/env python3
"""
RNN Language Model Implementation
使用 PyTorch 實現循環神經網路語言模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import Counter, defaultdict
import logging
import time
from typing import List, Tuple, Dict
import os
import matplotlib.pyplot as plt
import math

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Vocabulary:
    """詞彙表類別，處理詞彙到索引的轉換"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # 特殊標記
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        
        # 初始化特殊標記
        self.add_word(self.pad_token)
        self.add_word(self.unk_token)
        self.add_word(self.start_token)
        self.add_word(self.end_token)
        
    def add_word(self, word: str) -> int:
        """添加詞彙到詞彙表"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1
        return self.word2idx[word]
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """建立詞彙表"""
        logger.info("Building vocabulary...")
        
        # 統計詞頻
        for text in texts:
            words = self.preprocess_text(text)
            for word in words:
                self.word_count[word] += 1
        
        # 添加高頻詞到詞彙表
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                self.add_word(word)
        
        logger.info(f"Vocabulary size: {len(self.word2idx)}")
        logger.info(f"Most common words: {self.word_count.most_common(10)}")
    
    def preprocess_text(self, text: str) -> List[str]:
        """文本預處理"""
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def text_to_indices(self, text: str) -> List[int]:
        """將文本轉換為索引序列"""
        words = self.preprocess_text(text)
        indices = [self.word2idx[self.start_token]]
        
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx[self.unk_token])
        
        indices.append(self.word2idx[self.end_token])
        return indices
    
    def indices_to_text(self, indices: List[int]) -> str:
        """將索引序列轉換為文本"""
        words = []
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word not in [self.pad_token, self.start_token, self.end_token]:
                    words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)

class TextDataset(Dataset):
    """文本數據集類別"""
    
    def __init__(self, texts: List[str], vocab: Vocabulary, seq_length: int = 50):
        self.vocab = vocab
        self.seq_length = seq_length
        self.sequences = []
        
        self.prepare_sequences(texts)
    
    def prepare_sequences(self, texts: List[str]):
        """準備訓練序列"""
        logger.info("Preparing training sequences...")
        
        for text in texts:
            indices = self.vocab.text_to_indices(text)
            
            # 如果序列太短，跳過
            if len(indices) < 2:
                continue

            # 限制總 sequence 數量
            if len(self.sequences) >= 100000:
                break
            
            # 創建滑動窗口序列
            for i in range(len(indices) - 1):
                # 輸入序列和目標序列
                input_seq = indices[max(0, i - self.seq_length + 1):i + 1]
                target = indices[i + 1]
                
                # 填充到固定長度
                if len(input_seq) < self.seq_length:
                    padding = [self.vocab.word2idx[self.vocab.pad_token]] * (self.seq_length - len(input_seq))
                    input_seq = padding + input_seq
                
                self.sequences.append((input_seq, target))
        
        logger.info(f"Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class RNNLanguageModel(nn.Module):
    """RNN 語言模型"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(RNNLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # RNN 層
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # 輸出層
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化權重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型權重"""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, x, hidden=None):
        """前向傳播"""
        batch_size = x.size(0)
        
        # 詞嵌入
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # RNN
        rnn_out, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # 取最後一個時間步的輸出
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Dropout
        output = self.dropout(last_output)
        
        # 線性層
        output = self.linear(output)  # (batch_size, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隱藏狀態"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

class RNNTrainer:
    """RNN 訓練器"""
    
    def __init__(self, model, vocab, device, learning_rate: float = 0.001):
        self.model = model
        self.vocab = vocab
        self.device = device
        
        # 損失函數和優化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.pad_token])
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 追蹤訓練歷史
        self.train_losses = []
        self.train_accuracies = []
        
    def train_epoch(self, dataloader):
  
        self.model.train()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        # print out how many batches
        logger.info(f"Training on {len(dataloader)} batches...")
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            batch_size = data.size(0)
            
            hidden = self.model.init_hidden(batch_size, self.device)
            
            outputs, _ = self.model(data, hidden)
            loss = self.criterion(outputs, targets)
            
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """評估模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                batch_size = data.size(0)
                
                hidden = self.model.init_hidden(batch_size, self.device)
                outputs, _ = self.model(data, hidden)
                loss = self.criterion(outputs, targets)
                
                # 計算準確率
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == targets).sum().item()
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        perplexity = math.exp(avg_loss)
        
        return avg_loss, accuracy, perplexity
    
    def generate_text(self, start_text: str, max_length: int = 50, temperature: float = 1.0):
        """生成文本"""
        self.model.eval()
        
        # 預處理起始文本
        indices = self.vocab.text_to_indices(start_text)
        if len(indices) == 0:
            indices = [self.vocab.word2idx[self.vocab.start_token]]
        
        generated = indices.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 準備輸入序列
                input_seq = generated[-50:]  # 取最後50個詞作為上下文
                if len(input_seq) < 50:
                    padding = [self.vocab.word2idx[self.vocab.pad_token]] * (50 - len(input_seq))
                    input_seq = padding + input_seq
                
                input_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)
                hidden = self.model.init_hidden(1, self.device)
                
                # 預測下一個詞
                outputs, _ = self.model(input_tensor, hidden)
                outputs = outputs / temperature
                probabilities = torch.softmax(outputs, dim=-1)
                
                # 隨機採樣
                next_word_idx = torch.multinomial(probabilities, 1).item()
                
                # 如果生成結束標記，停止生成
                if next_word_idx == self.vocab.word2idx[self.vocab.end_token]:
                    break
                
                generated.append(next_word_idx)
        
        return self.vocab.indices_to_text(generated)
    
    def plot_learning_curves(self, save_path='rnn_learning_curves.png'):
        """繪製學習曲線"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 繪製訓練損失
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.set_title('Training Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 繪製訓練準確率
        ax2.plot(epochs, self.train_accuracies, 'r-', label='Training Accuracy')
        ax2.set_title('Training Accuracy Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Learning curves saved to {save_path}")

def load_data(file_path: str) -> List[str]:
    """載入訓練數據"""
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    except FileNotFoundError:
        logger.error(f"File {file_path} not found!")
        return []
    
    logger.info(f"Loaded {len(texts)} texts from {file_path}")
    return texts

def evaluate_on_test_data(trainer, vocab, test_file: str, seq_length: int = 50, batch_size: int = 32):
    """在測試數據上評估模型"""
    logger.info(f"Evaluating model on {test_file}...")
    
    # 載入測試數據
    test_texts = load_data(test_file)
    if not test_texts:
        return
    
    # 創建測試數據集
    test_dataset = TextDataset(test_texts, vocab, seq_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 評估模型
    test_loss, test_accuracy, test_perplexity = trainer.evaluate(test_dataloader)
    
    print("\n" + "="*60)
    print("RNN 模型測試結果")
    print("="*60)
    print(f"測試損失 (Test Loss): {test_loss:.4f}")
    print(f"測試準確率 (Test Accuracy): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"測試困惑度 (Test Perplexity): {test_perplexity:.4f}")
    print("="*60)
    
    return test_loss, test_accuracy, test_perplexity

def test_incomplete_sentences(trainer, incomplete_file: str):
    """測試不完整句子的補全"""
    logger.info("Testing incomplete sentence completion...")
    
    try:
        with open(incomplete_file, 'r', encoding='utf-8') as f:
            incomplete_texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"File {incomplete_file} not found!")
        return
    
    print("\n" + "="*60)
    print("RNN 模型文本補全結果")
    print("="*60)
    
    for incomplete_text in incomplete_texts:
        completed = trainer.generate_text(incomplete_text, max_length=20, temperature=0.8)
        print(f"輸入: {incomplete_text}")
        print(f"補全: {completed}")
        print("-" * 50)

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Hyperparameters
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    SEQ_LENGTH = 50
    
    # Load data
    train_texts = load_data('data/train.txt')
    if not train_texts:
        return

    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(train_texts, min_freq=3)

    # Create dataset
    train_dataset = TextDataset(train_texts, vocab, SEQ_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = RNNLanguageModel(
        vocab_size=len(vocab),
        embed_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # create trainer
    trainer = RNNTrainer(model, vocab, device, LEARNING_RATE)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss, train_accuracy = trainer.train_epoch(train_dataloader)
        
        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Time: {epoch_time:.2f}s')
    
    # save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'hyperparameters': {
            'vocab_size': len(vocab),
            'embed_dim': HIDDEN_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS
        }
    }, 'rnn_model.pth')
    
    logger.info("Model saved to rnn_model.pth")
    
    # plot learning curves
    trainer.plot_learning_curves('rnn_learning_curves.png')
    
    # evaluate on test data
    evaluate_on_test_data(trainer, vocab, 'test.txt', SEQ_LENGTH, BATCH_SIZE)
    
    # test incomplete sentences
    test_incomplete_sentences(trainer, 'incomplete.txt')

if __name__ == "__main__":
    main()