from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional


class NGramModel:
    """
    N-gram语言模型类

    用于从索引序列中学习n-gram概率分布
    支持开始符（-1）和结束符（-2）

    改进：支持预定义词汇表（initial_vocab参数）
    """

    def __init__(self,
                 n: int = 3,
                 k: float = 0.00001,
                 start_token: int = -1,
                 end_token: int = -2,
                 initial_vocab: Optional[Set[int]] = None):
        """
        初始化N-gram模型

        参数:
            n: n-gram的n值（例如，n=3表示trigram）
            k: 平滑参数（加法平滑/Laplace平滑）
            start_token: 开始符的索引值（默认-1）
            end_token: 结束符的索引值（默认-2）
            initial_vocab: 预定义的完整词汇表（可选）
                          如果提供，会确保这些符号都在最终词汇表中
                          即使它们在训练数据中没出现

        示例:
            # 不指定initial_vocab - 向后兼容，行为完全相同
            model = NGramModel(n=3, k=0.00001)

            # 指定initial_vocab - 确保所有符号都可编码
            model = NGramModel(n=3, k=0.00001, initial_vocab=set(range(2048)))
        """
        self.n = n
        self.k = k
        self.start_token = start_token
        self.end_token = end_token
        self.initial_vocab = initial_vocab  # 新增：预定义词汇表
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
            - 如果提供了initial_vocab，会确保所有预定义符号都在词汇表中
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

        # 构建词汇表
        if self.initial_vocab is not None:
            # 如果提供了预定义词汇表，使用它作为基础
            self.vocab = set(self.initial_vocab)
            # 同时包含训练数据中的符号（确保不遗漏）
            self.vocab.update(all_indices)
        else:
            # 否则只使用训练数据中的符号（原始行为，向后兼容）
            self.vocab = set(all_indices)

        V = len(self.vocab)

        # 统计n-gram（只统计实际出现的）
        for seq in processed_sequences:
            for i in range(len(seq) - self.n + 1):
                ngram = tuple(seq[i:i + self.n])
                context = ngram[:-1]  # 前n-1个元素

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        # 计算条件概率分布
        # 格式: {context: {next_char: probability}}
        self.prob_distribution = defaultdict(dict)

        # 为每个见过的上下文计算概率
        for context in self.context_counts.keys():
            context_count = self.context_counts[context]

            # 关键改进：为词汇表中的每个符号计算概率
            for next_char in self.vocab:
                ngram = context + (next_char,)
                ngram_count = self.ngram_counts.get(ngram, 0)  # 未见过的为0

                # 使用加法平滑 (Add-k smoothing / Laplace smoothing)
                numerator = ngram_count + self.k
                denominator = context_count + self.k * V

                probability = numerator / denominator
                self.prob_distribution[context][next_char] = probability

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

        上下文处理:
            - 如果context在训练中见过：返回训练得到的概率分布
            - 如果context未见过：返回均匀分布（所有vocab符号等概率）
        """
        if context in self.prob_distribution:
            # 上下文见过，返回训练的概率分布
            return self.prob_distribution[context]
        else:
            # 上下文未见过，返回均匀分布
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
            # 上下文未见过，返回均匀概率
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


# 使用示例
if __name__ == "__main__":
    print("=" * 70)
    print("NGramModel - 向后兼容 + 新增initial_vocab支持")
    print("=" * 70)

    # 示例数据
    sequences = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 2, 3, 5, 6],
    ]

    # ========================================================================
    print("\n示例 1: 原始用法（向后兼容，行为完全相同）")
    print("=" * 70)

    model_old = NGramModel(n=3, k=0.01, start_token=-1, end_token=-2)
    model_old.fit(sequences)

    print(f"词汇表大小: {len(model_old.vocab)}")
    print(f"词汇表: {sorted(model_old.vocab)}")

    # 测试未见符号
    test_seq = [1, 2, 100]  # 100未见过
    missing = [s for s in test_seq if s not in model_old.vocab]
    if missing:
        print(f"❌ 无法编码符号: {missing}")

    # ========================================================================
    print("\n示例 2: 新用法（使用initial_vocab）")
    print("=" * 70)

    # 预定义完整词汇表
    full_vocab = set(range(200))  # 0-199

    model_new = NGramModel(
        n=3,
        k=0.01,
        start_token=-1,
        end_token=-2,
        initial_vocab=full_vocab  # 🔑 新增参数
    )
    model_new.fit(sequences)  # 训练数据不变

    print(f"词汇表大小: {len(model_new.vocab)}")

    # 测试未见符号
    missing = [s for s in test_seq if s not in model_new.vocab]
    if missing:
        print(f"❌ 无法编码符号: {missing}")
    else:
        print(f"✅ 所有符号都可编码（包括100）")

    # ========================================================================
    print("\n示例 3: 上下文未见过的处理")
    print("=" * 70)

    # 训练中见过的上下文
    seen_context = (1, 2)
    print(f"\n上下文 {seen_context} (训练中见过):")
    probs_seen = model_new.get_next_char_prob(seen_context)
    print(f"  包含 {len(probs_seen)} 个符号的概率")
    # 显示前3个最高概率
    top3 = sorted(probs_seen.items(), key=lambda x: -x[1])[:3]
    for char, prob in top3:
        print(f"    符号 {char}: {prob:.6f}")

    # 训练中未见过的上下文
    unseen_context = (100, 101)
    print(f"\n上下文 {unseen_context} (训练中未见过):")
    probs_unseen = model_new.get_next_char_prob(unseen_context)
    print(f"  返回均匀分布: 所有 {len(probs_unseen)} 个符号等概率")
    print(f"  每个符号概率: {1.0 / len(probs_unseen):.6f}")
    # 验证是否均匀
    unique_probs = set(probs_unseen.values())
    print(f"  是否均匀: {'✓' if len(unique_probs) == 1 else '✗'}")

    # ========================================================================
    print("\n示例 4: 概率分布对比")
    print("=" * 70)

    context = (1, 2)

    # 有initial_vocab时
    probs_with = model_new.prob_distribution.get(context, {})
    print(f"\n使用initial_vocab:")
    print(f"  上下文 {context} 包含 {len(probs_with)} 个符号的概率")

    # 没有initial_vocab时
    probs_without = model_old.prob_distribution.get(context, {})
    print(f"\n不使用initial_vocab:")
    print(f"  上下文 {context} 包含 {len(probs_without)} 个符号的概率")

    print(f"\n差异: {len(probs_with) - len(probs_without)} 个额外符号")

    # ========================================================================
    print("\n示例 5: 实际编码测试")
    print("=" * 70)

    from arithmetic_coding import ArithmeticEncoder

    # 使用新模型
    encoder = ArithmeticEncoder(ngram_model=model_new, bits=32)

    test_sequences = [
        [1, 2, 3],  # 训练中见过
        [1, 2, 100],  # 包含未见符号
        [100, 101, 102],  # 全是未见符号
    ]

    for i, seq in enumerate(test_sequences):
        try:
            encoded = encoder.encode(seq)
            decoded = encoder.decode(encoded)
            correct = "✓" if decoded == seq else "✗"
            print(f"序列 {i} {seq}: {len(encoded)} bits {correct}")
        except Exception as e:
            print(f"序列 {i} {seq}: ✗ {e}")

    # ========================================================================
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
✅ 向后兼容：
   - 不提供initial_vocab时，行为与原版完全相同
   - 类名保持NGramModel，可以直接替换

✅ 新功能：
   - 提供initial_vocab参数，预定义完整词汇表
   - 训练数据无需修改，不添加占位符
   - 自动为所有vocab符号分配概率

✅ 上下文处理：
   - 见过的上下文：使用训练得到的概率分布
   - 未见的上下文：返回均匀分布（所有符号等概率）

🎯 推荐用法：
   model = NGramModel(n=3, k=0.00001, initial_vocab=set(range(2048)))
    """)