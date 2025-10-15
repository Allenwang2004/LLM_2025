import re
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import logging

class NGramModel:
    """
    N-gram語言模型實現
    支援 n=2 (bigram) 和 n=3 (trigram)
    """
    
    def __init__(self, n: int = 2):
        """
        初始化N-gram模型
        
        Args:
            n (int): n-gram的階數 (2 for bigram, 3 for trigram)
        """
        self.n = n
        self.ngram_counts = defaultdict(int)  # n-gram計數
        self.context_counts = defaultdict(int)  # (n-1)-gram計數
        self.vocabulary = set()  # 詞彙表
        self.total_words = 0
        
        # 特殊符號
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.unk_token = "<unk>"
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        文本預處理
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 處理後的詞列表
        """
        # 轉小寫，保留字母、數字和基本標點
        text = text.lower()
        # 用正則表達式分詞
        words = re.findall(r'\b\w+\b|[.!?]', text)
        return words
    
    def add_sentence_markers(self, words: List[str]) -> List[str]:
        """
        為句子添加開始和結束標記
        
        Args:
            words (List[str]): 詞列表
            
        Returns:
            List[str]: 添加標記後的詞列表
        """
        # 根據n-gram階數添加適當數量的開始標記
        start_markers = [self.start_token] * (self.n - 1)
        return start_markers + words + [self.end_token]
    
    def get_ngrams(self, words: List[str]) -> List[Tuple[str, ...]]:
        """
        從詞列表生成n-gram
        
        Args:
            words (List[str]): 詞列表
            
        Returns:
            List[Tuple[str, ...]]: n-gram列表
        """
        ngrams = []
        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def get_contexts(self, words: List[str]) -> List[Tuple[str, ...]]:
        """
        從詞列表生成context (n-1)-gram
        
        Args:
            words (List[str]): 詞列表
            
        Returns:
            List[Tuple[str, ...]]: context列表
        """
        contexts = []
        for i in range(len(words) - self.n + 1):
            context = tuple(words[i:i + self.n - 1])
            contexts.append(context)
        return contexts
    
    def train(self, train_file: str):
        """
        訓練N-gram模型
        
        Args:
            train_file (str): 訓練文件路徑
        """
        self.logger.info(f"Start training {self.n}-gram model...")
        
        line_count = 0
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 預處理文本
                words = self.preprocess_text(line)
                if len(words) == 0:
                    continue
                
                # 添加句子標記
                words_with_markers = self.add_sentence_markers(words)
                
                # 更新詞彙表
                self.vocabulary.update(words)
                self.total_words += len(words)
                
                # 生成n-gram和context
                ngrams = self.get_ngrams(words_with_markers)
                contexts = self.get_contexts(words_with_markers)
                
                # 更新計數
                for ngram in ngrams:
                    self.ngram_counts[ngram] += 1
                
                for context in contexts:
                    self.context_counts[context] += 1

        self.logger.info(f"Training completed!")
        self.logger.info(f"Total words: {self.total_words}")
        self.logger.info(f"Vocabulary size: {len(self.vocabulary)}")
        self.logger.info(f"N-gram total: {len(self.ngram_counts)}")
        self.logger.info(f"Context total: {len(self.context_counts)}")

    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        計算n-gram的條件概率
        使用最大似然估計 (MLE)
        
        Args:
            ngram (Tuple[str, ...]): n-gram
            
        Returns:
            float: 條件概率
        """
        if len(ngram) != self.n:
            raise ValueError(f"N-gram長度應為 {self.n}")
        
        # 取得context
        context = ngram[:-1]
        
        # 處理未見過的context
        if self.context_counts[context] == 0:
            return 1e-10  # 平滑處理，避免概率為0
        
        # P(w_n | w_1, ..., w_{n-1}) = Count(w_1, ..., w_n) / Count(w_1, ..., w_{n-1})
        return self.ngram_counts[ngram] / self.context_counts[context]
    
    def get_sentence_probability(self, sentence: str) -> float:
        """
        計算句子的概率
        
        Args:
            sentence (str): 句子
            
        Returns:
            float: 句子概率的對數值
        """
        words = self.preprocess_text(sentence)
        if len(words) == 0:
            return float('-inf')
        
        words_with_markers = self.add_sentence_markers(words)
        ngrams = self.get_ngrams(words_with_markers)
        
        log_prob = 0.0
        for ngram in ngrams:
            prob = self.get_probability(ngram)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += math.log(1e-10)  # 平滑處理
        
        return log_prob
    
    def calculate_perplexity(self, test_file: str) -> float:
        """
        計算測試集的困惑度 (perplexity)
        
        Args:
            test_file (str): 測試文件路徑
            
        Returns:
            float: 困惑度值
        """
        self.logger.info(f"Calculating {self.n}-gram model perplexity...")
        
        total_log_prob = 0.0
        total_words = 0
        line_count = 0
        
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                words = self.preprocess_text(line)
                if len(words) == 0:
                    continue
                
                # 計算句子概率
                log_prob = self.get_sentence_probability(line)
                total_log_prob += log_prob
                total_words += len(words)
                
        
        # 計算困惑度: PP = exp(-1/N * sum(log P(sentence)))
        avg_log_prob = total_log_prob / total_words
        perplexity = math.exp(-avg_log_prob)
        
        self.logger.info(f"測試集總詞數: {total_words}")
        self.logger.info(f"平均對數概率: {avg_log_prob:.6f}")
        self.logger.info(f"{self.n}-gram 困惑度: {perplexity:.2f}")
        
        return perplexity
    
    def calculate_accuracy(self, test_file: str) -> float:
        """
        計算測試集的準確率 (accuracy)
        預測下一個詞的準確率
        
        Args:
            test_file (str): 測試文件路徑
            
        Returns:
            float: 準確率值 (0-1)
        """
        self.logger.info(f"Calculating {self.n}-gram model accuracy...")
        
        correct_predictions = 0
        total_predictions = 0
        
        with open(test_file, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                words = self.preprocess_text(line)
                if len(words) < self.n:
                    continue
                
                # 添加句子標記
                words_with_markers = self.add_sentence_markers(words)

                # 對每個可能的 n-gram 進行預測
                for i in range(self.n - 1, len(words_with_markers)):
                    count += 1
                    if count > 100:  # 每行最多預測100次
                        break
                    # 取得 context (前 n-1 個詞)
                    context = tuple(words_with_markers[i - self.n + 1:i])
                    # 實際的下一個詞
                    actual_word = words_with_markers[i]
                    
                    # 預測下一個詞 (選擇概率最高的詞)
                    predicted_word = self.predict_next_word(context)
                    
                    if predicted_word == actual_word:
                        correct_predictions += 1
                    total_predictions += 1
        
        if total_predictions == 0:
            self.logger.warning("No predictions made!")
            return 0.0
        
        accuracy = correct_predictions / total_predictions
        
        self.logger.info(f"正確預測數: {correct_predictions}")
        self.logger.info(f"總預測數: {total_predictions}")
        self.logger.info(f"{self.n}-gram 準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return accuracy
    
    def predict_next_word(self, context: Tuple[str, ...]) -> str:
        """
        根據給定的 context 預測下一個最可能的詞
        
        Args:
            context (Tuple[str, ...]): 上下文
            
        Returns:
            str: 預測的下一個詞
        """
        if len(context) != self.n - 1:
            raise ValueError(f"Context 長度應為 {self.n - 1}")
        
        # 找出所有以此 context 開始的 n-gram
        candidate_ngrams = []
        for ngram in self.ngram_counts:
            if ngram[:-1] == context:
                probability = self.get_probability(ngram)
                candidate_ngrams.append((ngram[-1], probability))
        
        if not candidate_ngrams:
            # 如果沒有找到匹配的 n-gram，返回未知詞標記
            return self.unk_token
        
        # 選擇概率最高的詞
        best_word = max(candidate_ngrams, key=lambda x: x[1])[0]
        return best_word
    
    def generate_text(self, context: Tuple[str, ...], max_length: int = 20) -> str:
        """
        基於給定context生成文本
        
        Args:
            context (Tuple[str, ...]): 初始context
            max_length (int): 最大生成長度
            
        Returns:
            str: 生成的文本
        """
        if len(context) != self.n - 1:
            raise ValueError(f"Context長度應為 {self.n - 1}")
        
        result = list(context)
        current_context = context
        
        for _ in range(max_length):
            # 找到所有以current_context開頭的n-gram
            candidates = []
            for ngram, count in self.ngram_counts.items():
                if ngram[:-1] == current_context:
                    candidates.extend([ngram[-1]] * count)
            
            if not candidates:
                break
            
            # 隨機選擇下一個詞
            import random
            next_word = random.choice(candidates)
            
            if next_word == self.end_token:
                break
            
            result.append(next_word)
            # 更新context
            current_context = tuple(result[-(self.n-1):])
        
        # 移除開始標記
        filtered_result = [word for word in result if word != self.start_token]
        return ' '.join(filtered_result)
