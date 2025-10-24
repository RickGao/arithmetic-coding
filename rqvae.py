from ngram import NGramModel
from arithmetic_coding import ArithmeticEncoder


def readcode(filename, n=None):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:  # 如果指定了行数且已读够，则停止
                break
            line = line.strip()
            if line:  # 跳过空行
                numbers = [int(x) for x in line.split()]
                result.append(numbers)
    return result



training_sequences = readcode('codes23x40x4.txt', 900)

# model = NGramModel(n=1, k=0.1, start_token=-1, end_token=-2, initial_vocab=set(range(2048)))
model = NGramModel(n=2, k=0.1, start_token=-1, end_token=-2, initial_vocab=set(range(2048)))


model.fit(training_sequences)


prob_dist = model.get_probability_distribution()
with open('prob_dist.txt', 'w') as f:
    f.write(str(prob_dist))
print("Context:", len(prob_dist))

# inner_dict = prob_dist[(1498,)]
# total = sum(inner_dict.values())
# print(total)


encoder = ArithmeticEncoder(ngram_model=model, bits=32)

test_sequence = readcode('codes23x40x4.txt', 1000)[990][:3000]

encoded_bits = encoder.encode(test_sequence)
# print(encoded_bits)
# print(f"\n原始序列: {test_sequence}")
print(f"编码后: {len(encoded_bits)} bits")
print(f"压缩率: {len(encoded_bits) / (len(test_sequence) * 11):.2%}")

# 步骤4: 解码验证
decoded_sequence = encoder.decode(encoded_bits)
# print(f"解码结果: {decoded_sequence}")
print(f"正确性: {'正确' if decoded_sequence == test_sequence else '错误'}")




