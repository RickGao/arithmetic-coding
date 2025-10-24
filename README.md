# N-gram 静态算术编码器

这个项目实现了基于N-gram概率分布的静态算术编码器。

## 文件说明

### 核心文件

1. **ngram.py** - N-gram语言模型
   - `NGramModel` 类：从序列数据学习n-gram概率分布
   - 支持开始符（-1）和结束符（-2）
   - 使用加法平滑（Laplace平滑）处理未见过的组合

2. **arithmetic_coding.py** - 静态算术编码器
   - `ArithmeticEncoder` 类：使用N-gram概率分布进行编码/解码
   - 上下文相关编码：根据当前上下文选择概率分布
   - 支持可配置的精度（bits参数）

3. **ngram_encode.py** - 主程序
   - 读取序列文件
   - 训练N-gram模型
   - 对每个序列进行编码和解码验证
   - 显示详细的统计信息

4. **test_sequences.txt** - 测试数据文件

## 使用方法

### 基本使用

运行简单示例测试：
```bash
python ngram_encode.py
```

### 从文件读取并编码

```bash
python ngram_encode.py <filename> [num_lines] [n] [bits]
```

参数说明：
- `filename`: 输入文件路径（每行一个序列，空格分隔的整数）
- `num_lines`: 读取的行数（可选，默认读取所有行）
- `n`: N-gram大小（可选，默认3表示trigram）
- `bits`: 算术编码精度位数（可选，默认32）

### 示例

```bash
# 处理测试文件的前100行，使用trigram和32位精度
python ngram_encode.py test_sequences.txt 100 3 32

# 处理整个文件，使用4-gram和64位精度
python ngram_encode.py test_sequences.txt 0 4 64
```

## 工作原理

### 1. N-gram模型训练
```python
from ngram import NGramModel

# 训练数据
sequences = [
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    # ...
]

# 创建和训练模型
model = NGramModel(n=3, k=0.00001, start_token=-1, end_token=-2)
model.fit(sequences)

# 获取概率分布
prob_dist = model.get_probability_distribution()
# 格式: {(context_tuple): {next_char: probability}}
# 例如: {(-1, -1): {1: 0.5, 2: 0.3, ...}, (1, 2): {3: 0.6, 4: 0.3, ...}}
```

### 2. 算术编码
```python
from arithmetic_coding import ArithmeticEncoder

# 使用N-gram模型创建编码器
encoder = ArithmeticEncoder(ngram_model=model, bits=32)

# 编码序列
sequence = [1, 2, 3, 4, 5]
encoded_bits = encoder.encode(sequence)

# 解码
decoded_sequence = encoder.decode(encoded_bits)
assert decoded_sequence == sequence
```

## 特点

1. **上下文相关编码**
   - 根据前n-1个符号的上下文动态选择概率分布
   - 利用序列的统计规律进行高效压缩

2. **统一的结束符**
   - 使用 -2 作为结束符（EOM标记）
   - 自动处理序列边界

3. **静态概率分布**
   - 先训练N-gram模型，得到固定的概率分布
   - 编码时不更新概率（非自适应）

4. **高压缩率**
   - 测试数据显示压缩率约10%（从11 bits/符号压缩到约1 bit/符号）
   - 压缩率取决于数据的可预测性

## 输入文件格式

输入文件应该是文本文件，每行一个序列，符号之间用空格分隔：

```
1 2 3 4 5 1 8 9
2 3 4 5 6 2 3 4
3 4 5 6 7 3 4 5
...
```

支持的符号：任何整数（通常是非负整数，但也支持负数）

## 输出说明

程序会输出：
1. 训练统计：序列数量、总符号数、词汇表大小、上下文数量
2. 编码结果：每个序列的原始大小、编码后大小、压缩率、正确性验证
3. 总体统计：总压缩率、节省的比特数
4. 概率分布样本：显示一些重要上下文的概率分布

示例输出：
```
======================================================================
Training 3-gram model
======================================================================
Number of sequences: 8
Total symbols: 64
Vocabulary size: 11
Number of unique contexts: 25

======================================================================
Encoding sequences
======================================================================
Sequence    0:    8 symbols ->      7 bits (ratio  7.95%) ✓
Sequence    1:    8 symbols ->     10 bits (ratio 11.36%) ✓
...

======================================================================
Summary
======================================================================
Total original: 704 bits
Total encoded:  70 bits
Average compression ratio: 9.94%
Bits saved: 634 bits
```

## 技术细节

### N-gram平滑
使用加法平滑（Add-k smoothing）处理未见过的n-gram组合：
```
P(w_n | w_1...w_{n-1}) = (count(w_1...w_n) + k) / (count(w_1...w_{n-1}) + k*V)
```
其中 V 是词汇表大小，k 是平滑参数。

### 算术编码精度
- `bits` 参数控制内部寄存器的位宽
- 更高的位数 = 更高的精度，但也需要更多内存
- 推荐值：28-64位

### 特殊符号
- 开始符（start_token）: -1（默认）
- 结束符（end_token）: -2（默认）
- 这些符号会自动添加，不需要在输入数据中包含

## 注意事项

1. 所有序列必须使用相同的N-gram模型进行编码
2. 解码时需要使用与编码时相同的概率分布
3. 较大的n值可能提高压缩率，但也需要更多训练数据
4. 平滑参数k应该很小（如0.00001）以避免过度平滑

## 扩展建议

1. **保存/加载模型**：添加序列化功能保存训练好的模型
2. **自适应编码**：实现动态更新概率分布的版本
3. **分层模型**：使用多个N-gram大小的组合
4. **并行处理**：支持多线程编码多个序列
