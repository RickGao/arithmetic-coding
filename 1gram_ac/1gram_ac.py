#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Arithmetic Coding with Frequency Statistics
"""

from collections import Counter
from arithmetic_coding import ArithmeticEncoder


def encode_list(data, bits=16):
    """
    Encode a list using frequency statistics from the list itself.

    Parameters:
        data: list to encode
        bits: number of bits for arithmetic encoder

    Returns:
        list of encoded bits
    """
    # Count frequencies
    data_str = [str(s) for s in data]
    frequencies = dict(Counter(data_str))
    frequencies['<EOM>'] = 1

    print(f"Number of unique symbols: {len(frequencies) - 1}")
    print(f"Frequency counts: {frequencies}")

    # Encode
    encoder = ArithmeticEncoder(frequencies=frequencies, bits=bits, EOM='<EOM>')
    message = data_str + ['<EOM>']
    all_bits = list(encoder.encode(message))

    print(f"Original: {len(data)} symbols")
    print(f"Encoded: {len(all_bits)} bits")
    print(f"Compression ratio: {len(all_bits) / (len(data) * 8):.2%}")

    return all_bits


def encode_file(filename, num_lines=None, bits=16):
    """
    Read from file and encode each line.
    Uses global frequency statistics from all lines, but encodes each line independently.

    Parameters:
        filename: path to input file
        num_lines: number of lines to read (None = all lines)
        bits: number of bits for arithmetic encoder

    Returns:
        list of tuples (original_data, encoded_bits) for each line
    """
    # Read file
    sequences = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            if num_lines and line_num >= num_lines:
                break

            tokens = line.strip().split()
            if tokens:
                try:
                    data = [int(token) for token in tokens]
                except ValueError:
                    data = tokens
                sequences.append(data)

    print(f"Read {len(sequences)} lines")
    total_symbols = sum(len(seq) for seq in sequences)
    print(f"Total symbols: {total_symbols}")

    # Compute global frequency statistics (across all lines, but lines are not concatenated)
    all_symbols = []
    for seq in sequences:
        # print(seq)
        all_symbols.extend(str(s) for s in seq)

    frequencies = dict(Counter(all_symbols))
    frequencies['<EOM>'] = 1

    print(f"Number of unique symbols: {len(frequencies) - 1}")
    print(f"Frequency counts: {frequencies}")
    print()

    # Encode each line independently using global frequencies
    results = []
    total_bits = 0

    for line_num, data in enumerate(sequences):
        data_str = [str(s) for s in data]

        # Use global frequency table
        encoder = ArithmeticEncoder(frequencies=frequencies, bits=bits, EOM='<EOM>')
        message = data_str + ['<EOM>']
        encoded_bits = list(encoder.encode(message))
        decoded_bits = list(encoder.decode(encoded_bits))
        # print(decoded_bits)

        results.append((data, encoded_bits))
        total_bits += len(encoded_bits)

        ratio = len(encoded_bits) / (len(data) * 11) if len(data) > 0 else 0
        print(f"Line {line_num}: {len(data)} symbols -> {len(encoded_bits)} bits (ratio {ratio:.2%})")

    print(f"\nTotal: {total_symbols} symbols -> {total_bits} bits")
    print(f"Average compression ratio: {total_bits / (total_symbols * 11):.2%}")

    return results


if __name__ == "__main__":
    # print("=" * 60)
    # print("Encode a list")
    # print("=" * 60)
    # data = [1, 2, 2, 3, 1, 2, 3, 1, 2, 2]
    # bits = encode_list(data)
    # print(bits)
    # print()

    print("=" * 60)
    print("Encode from file")
    print("=" * 60)

    # Create test file
    # with open('test_codes.txt', 'w') as f:
    #     f.write("1 2 3 4 5 1 8 9\n")
    #     f.write("2 3 4 5 6 2 3 4\n")
    #     f.write("3 4 5 6 7 3 4 5\n")
    #     f.write("1 1 2 2 3 3 4 4\n")

    # results_test = encode_file('test_codes.txt', num_lines=1)

    results_23x40x4 = encode_file('codes23x40x4.txt', num_lines=10, bits=32)

    # results_23x40x8 = encode_file('codes23x40x8.txt', num_lines=1000, bits=32)

    # results_23x40x16 = encode_file('codes23x40x16.txt', num_lines=1000, bits=32)


