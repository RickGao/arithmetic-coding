from collections import Counter, defaultdict
from typing import List, Dict, Tuple


class NGramModel:
    """
    N-gram语言模型类

    用于从索引序列中学习n-gram概率分布
    支持开始符（-1）和结束符（-2）
    """

    def __init__(self, n: int = 3, k: float = 0.00001, start_token: int = -1, end_token: int = -2):
        """
        初始化N-gram模型

        参数:
            n: n-gram的n值（例如，n=3表示trigram）
            k: 平滑参数（加法平滑/Laplace平滑）
            start_token: 开始符的索引值（默认-1）
            end_token: 结束符的索引值（默认-2）
        """
        self.n = n
        self.k = k
        self.start_token = start_token
        self.end_token = end_token
        self.vocab = set()  # 词汇表
        self.ngram_counts = Counter()  # n-gram计数
        self.context_counts = Counter()  # 前n-1个字符的计数
        self.prob_distribution = {}  # 条件概率分布

    def fit(self, sequences: List[List[int]]):
        """
        训练模型

        参数:
            sequences: 输入的多行序列，每行是一个list of indices
                      例如: [[1, 2, 3, 4], [2, 3, 4, 5, 6], ...]

        注意:
            - 会自动在每个序列开头添加 n-1 个开始符（默认-1）
            - 会自动在每个序列结尾添加 1 个结束符（默认-2）
        """
        # 预处理序列：添加开始符和结束符
        processed_sequences = []
        for seq in sequences:
            # 在开头添加 n-1 个开始符
            padded_seq = [self.start_token] * (self.n - 1) + seq + [self.end_token]
            processed_sequences.append(padded_seq)

        # 合并所有序列来统计（包括特殊符号）
        all_indices = []
        for seq in processed_sequences:
            all_indices.extend(seq)

        # 构建词汇表（包括开始符和结束符）
        self.vocab = set(all_indices)
        V = len(self.vocab)

        # 统计n-gram
        for seq in processed_sequences:
            for i in range(len(seq) - self.n + 1):
                ngram = tuple(seq[i:i + self.n])
                context = ngram[:-1]  # 前n-1个元素

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        # 计算条件概率分布
        # 格式: {context: {next_char: probability}}
        self.prob_distribution = defaultdict(dict)

        for ngram, count in self.ngram_counts.items():
            context = ngram[:-1]  # 前n-1个字符
            next_char = ngram[-1]  # 最后一个字符

            # 使用加法平滑 (Add-k smoothing / Laplace smoothing)
            numerator = count + self.k
            denominator = self.context_counts[context] + self.k * V

            probability = numerator / denominator
            self.prob_distribution[context][next_char] = probability

        # 为未见过的context添加均匀分布
        for context in self.context_counts.keys():
            if context not in self.prob_distribution:
                self.prob_distribution[context] = {}

            # 确保所有可能的next_char都有概率
            for char in self.vocab:
                if char not in self.prob_distribution[context]:
                    # 对未见过的组合，使用平滑
                    denominator = self.context_counts[context] + self.k * V
                    self.prob_distribution[context][char] = self.k / denominator

    def get_probability_distribution(self) -> Dict[Tuple, Dict[int, float]]:
        """
        获取完整的概率分布字典

        返回:
            字典格式: {(前n-1个字符的tuple): {下一个字符: 概率}}
            例如: {(1, 2): {3: 0.5, 4: 0.3, 5: 0.2}}
        """
        return dict(self.prob_distribution)

    def get_next_char_prob(self, context: Tuple[int, ...]) -> Dict[int, float]:
        """
        给定context，返回下一个字符的概率分布

        参数:
            context: 前n-1个字符组成的tuple

        返回:
            字典 {next_char: probability}
        """
        if context in self.prob_distribution:
            return self.prob_distribution[context]
        else:
            # 如果context未见过，返回均匀分布
            V = len(self.vocab)
            return {char: 1.0 / V for char in self.vocab}

    def predict_next(self, context: Tuple[int, ...]) -> int:
        """
        给定context，预测最可能的下一个字符

        参数:
            context: 前n-1个字符组成的tuple

        返回:
            最可能的下一个字符的index
        """
        prob_dist = self.get_next_char_prob(context)
        return max(prob_dist.items(), key=lambda x: x[1])[0]

    def get_probability(self, ngram: Tuple[int, ...]) -> float:
        """
        计算特定n-gram的概率

        参数:
            ngram: 长度为n的tuple

        返回:
            该n-gram的条件概率
        """
        if len(ngram) != self.n:
            raise ValueError(f"ngram长度必须为{self.n}")

        context = ngram[:-1]
        next_char = ngram[-1]

        if context in self.prob_distribution:
            return self.prob_distribution[context].get(next_char, 0.0)
        else:
            return 1.0 / len(self.vocab)

    def get_start_context(self) -> Tuple[int, ...]:
        """
        获取句子开始时的context（n-1个开始符）

        返回:
            开始context的tuple
        """
        return tuple([self.start_token] * (self.n - 1))

    def is_end_token(self, token: int) -> bool:
        """
        判断是否为结束符

        参数:
            token: 要判断的token

        返回:
            是否为结束符
        """
        return token == self.end_token
