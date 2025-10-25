from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
import pickle
import json
from pathlib import Path


class NGramModel:
    """
    N-gram语言模型类

    用于从索引序列中学习n-gram概率分布
    支持开始符（-1）和结束符（-2）

    改进：
    - 支持预定义词汇表（initial_vocab参数）
    - 支持模型保存和加载（save/load方法）
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

    def save(self, filepath: str):
        """
        保存模型到文件（使用pickle）

        参数:
            filepath: 保存路径，建议使用 .pkl 扩展名
                     例如: "model.pkl" 或 "models/ngram_model.pkl"

        示例:
            model.save("ngram_model.pkl")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 保存所有需要的属性
        save_dict = {
            'n': self.n,
            'k': self.k,
            'start_token': self.start_token,
            'end_token': self.end_token,
            'initial_vocab': self.initial_vocab,
            'vocab': self.vocab,
            'ngram_counts': self.ngram_counts,
            'context_counts': self.context_counts,
            'prob_distribution': dict(self.prob_distribution)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Model saved to: {filepath}")
        print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")

    @classmethod
    def load(cls, filepath: str) -> 'NGramModel':
        """
        从文件加载模型（类方法）

        参数:
            filepath: 模型文件路径

        返回:
            加载的NGramModel实例

        示例:
            model = NGramModel.load("ngram_model.pkl")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model does not exist: {filepath}")

        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        # 创建新实例
        model = cls(
            n=save_dict['n'],
            k=save_dict['k'],
            start_token=save_dict['start_token'],
            end_token=save_dict['end_token'],
            initial_vocab=save_dict['initial_vocab']
        )

        # 恢复训练后的数据
        model.vocab = save_dict['vocab']
        model.ngram_counts = save_dict['ngram_counts']
        model.context_counts = save_dict['context_counts']
        model.prob_distribution = defaultdict(dict, save_dict['prob_distribution'])

        print(f"Model Loaded: {filepath}")
        print(f"n={model.n}, k={model.k}, vocab_size={len(model.vocab)}")

        return model

    # def save_json(self, filepath: str):
    #     """
    #     保存模型到JSON文件（可读性好，但文件较大）
    #
    #     参数:
    #         filepath: 保存路径，建议使用 .json 扩展名
    #
    #     注意:
    #         - JSON不支持tuple作为key，会转换为字符串
    #         - JSON不支持set，会转换为list
    #         - 文件比pickle格式大，但人类可读
    #     """
    #     filepath = Path(filepath)
    #     filepath.parent.mkdir(parents=True, exist_ok=True)
    #
    #     # 转换为JSON兼容格式
    #     save_dict = {
    #         'n': self.n,
    #         'k': self.k,
    #         'start_token': self.start_token,
    #         'end_token': self.end_token,
    #         'initial_vocab': list(self.initial_vocab) if self.initial_vocab else None,
    #         'vocab': list(self.vocab),
    #         'ngram_counts': {str(k): v for k, v in self.ngram_counts.items()},
    #         'context_counts': {str(k): v for k, v in self.context_counts.items()},
    #         'prob_distribution': {
    #             str(context): {str(char): prob for char, prob in dist.items()}
    #             for context, dist in self.prob_distribution.items()
    #         }
    #     }
    #
    #     with open(filepath, 'w', encoding='utf-8') as f:
    #         json.dump(save_dict, f, indent=2)
    #
    #     print(f"✅ 模型已保存到JSON: {filepath}")
    #     print(f"   文件大小: {filepath.stat().st_size / 1024:.2f} KB")

    # @classmethod
    # def load_json(cls, filepath: str) -> 'NGramModel':
    #     """
    #     从JSON文件加载模型
    #
    #     参数:
    #         filepath: JSON模型文件路径
    #
    #     返回:
    #         加载的NGramModel实例
    #     """
    #     filepath = Path(filepath)
    #
    #     if not filepath.exists():
    #         raise FileNotFoundError(f"模型文件不存在: {filepath}")
    #
    #     with open(filepath, 'r', encoding='utf-8') as f:
    #         save_dict = json.load(f)
    #
    #     # 创建新实例
    #     model = cls(
    #         n=save_dict['n'],
    #         k=save_dict['k'],
    #         start_token=save_dict['start_token'],
    #         end_token=save_dict['end_token'],
    #         initial_vocab=set(save_dict['initial_vocab']) if save_dict['initial_vocab'] else None
    #     )
    #
    #     # 恢复训练后的数据，将字符串key转回tuple
    #     model.vocab = set(save_dict['vocab'])
    #     model.ngram_counts = Counter({
    #         eval(k): v for k, v in save_dict['ngram_counts'].items()
    #     })
    #     model.context_counts = Counter({
    #         eval(k): v for k, v in save_dict['context_counts'].items()
    #     })
    #     model.prob_distribution = defaultdict(dict, {
    #         eval(context): {int(char): prob for char, prob in dist.items()}
    #         for context, dist in save_dict['prob_distribution'].items()
    #     })
    #
    #     print(f"✅ 模型已从JSON加载: {filepath}")
    #     print(f"   n={model.n}, k={model.k}, vocab_size={len(model.vocab)}")
    #
    #     return model

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

    def get_model_info(self) -> Dict:
        """
        获取模型信息摘要

        返回:
            包含模型统计信息的字典
        """
        return {
            'n': self.n,
            'k': self.k,
            'vocab_size': len(self.vocab),
            'num_unique_ngrams': len(self.ngram_counts),
            'num_unique_contexts': len(self.context_counts),
            'has_initial_vocab': self.initial_vocab is not None,
            'start_token': self.start_token,
            'end_token': self.end_token
        }


# 使用示例
if __name__ == "__main__":
    print("=" * 70)
    print("NGramModel - 保存和加载功能演示")
    print("=" * 70)

    # 示例数据
    sequences = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 2, 3, 5, 6],
        [1, 1, 2, 3, 4],
        [2, 2, 3, 4, 5],
    ]

    # ========================================================================
    print("\n步骤 1: 训练模型")
    print("=" * 70)

    model = NGramModel(
        n=3,
        k=0.01,
        start_token=-1,
        end_token=-2,
        initial_vocab=set(range(100))  # 预定义词汇表
    )
    model.fit(sequences)

    print("模型信息:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")

    # 测试预测
    test_context = (1, 2)
    pred = model.predict_next(test_context)
    print(f"\n预测: context {test_context} -> {pred}")

    # ========================================================================
    print("\n步骤 2: 保存模型（pickle格式）")
    print("=" * 70)

    model.save("ngram_model.pkl")

    # ========================================================================
    print("\n步骤 3: 保存模型（JSON格式）")
    print("=" * 70)

    model.save_json("ngram_model.json")

    # ========================================================================
    print("\n步骤 4: 加载模型（pickle格式）")
    print("=" * 70)

    loaded_model = NGramModel.load("ngram_model.pkl")

    # 验证加载的模型
    pred_loaded = loaded_model.predict_next(test_context)
    print(f"\n加载后预测: context {test_context} -> {pred_loaded}")
    print(f"预测是否一致: {'✓' if pred == pred_loaded else '✗'}")

    # ========================================================================
    print("\n步骤 5: 加载模型（JSON格式）")
    print("=" * 70)

    loaded_model_json = NGramModel.load_json("ngram_model.json")

    pred_json = loaded_model_json.predict_next(test_context)
    print(f"\nJSON加载后预测: context {test_context} -> {pred_json}")
    print(f"预测是否一致: {'✓' if pred == pred_json else '✗'}")

    # ========================================================================
    print("\n步骤 6: 对比概率分布")
    print("=" * 70)

    prob_original = model.get_next_char_prob(test_context)
    prob_loaded = loaded_model.get_next_char_prob(test_context)
    prob_json = loaded_model_json.get_next_char_prob(test_context)

    print(f"\n上下文 {test_context} 的概率分布（显示前5个）:")
    top5_original = sorted(prob_original.items(), key=lambda x: -x[1])[:5]

    print("\n原始模型:")
    for char, prob in top5_original:
        print(f"  {char}: {prob:.6f}")

    print("\nPickle加载:")
    for char, prob in sorted(prob_loaded.items(), key=lambda x: -x[1])[:5]:
        print(f"  {char}: {prob:.6f}")

    print("\nJSON加载:")
    for char, prob in sorted(prob_json.items(), key=lambda x: -x[1])[:5]:
        print(f"  {char}: {prob:.6f}")

    # 检查概率是否完全一致
    prob_match = all(
        abs(prob_original.get(char, 0) - prob_loaded.get(char, 0)) < 1e-10
        for char in set(prob_original.keys()) | set(prob_loaded.keys())
    )
    print(f"\n概率完全一致: {'✓' if prob_match else '✗'}")

    # ========================================================================
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
✅ 两种保存格式：
   1. Pickle格式（推荐）：
      - 文件小，速度快
      - 完全保留所有Python对象
      - 使用: model.save("model.pkl")
      - 加载: model = NGramModel.load("model.pkl")

   2. JSON格式：
      - 人类可读
      - 跨语言兼容
      - 文件较大
      - 使用: model.save_json("model.json")
      - 加载: model = NGramModel.load_json("model.json")

✅ 使用场景：
   - 生产环境：使用pickle，效率高
   - 调试/检查：使用JSON，可以手动查看
   - 跨平台：使用JSON，更通用

🎯 推荐用法：
   # 训练
   model = NGramModel(n=3, k=0.00001, initial_vocab=set(range(2048)))
   model.fit(sequences)
   model.save("models/ngram_model.pkl")

   # 加载
   model = NGramModel.load("models/ngram_model.pkl")
    """)