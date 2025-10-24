#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-gram based Arithmetic Coding for Sequences
"""

from ngram import NGramModel
from arithmetic_coding import ArithmeticEncoder


def encode_sequences_with_ngram(sequences, n=3, k=0.00001, bits=32, 
                                 start_token=-1, end_token=-2, verbose=True):
    """
    Train N-gram model on sequences and encode each sequence.
    
    Parameters:
        sequences: List of sequences (each sequence is a list of integers)
        n: N-gram size (e.g., 3 for trigram)
        k: Smoothing parameter for N-gram model
        bits: Precision bits for arithmetic coding
        start_token: Start token index
        end_token: End token index (also used as EOM)
        verbose: Whether to print detailed statistics
        
    Returns:
        tuple: (ngram_model, results)
            - ngram_model: Trained NGramModel object
            - results: List of tuples (original_sequence, encoded_bits, decoded_sequence)
    """
    if verbose:
        print("=" * 70)
        print(f"Training {n}-gram model")
        print("=" * 70)
        print(f"Number of sequences: {len(sequences)}")
        print(f"Total symbols: {sum(len(seq) for seq in sequences)}")
    
    # Train N-gram model
    model = NGramModel(n=n, k=k, start_token=start_token, end_token=end_token)
    model.fit(sequences)
    
    if verbose:
        print(f"Vocabulary size: {len(model.vocab)}")
        print(f"Number of unique contexts: {len(model.prob_distribution)}")
        print()
    
    # Create arithmetic encoder with N-gram model
    encoder = ArithmeticEncoder(
        ngram_model=model,
        bits=bits,
        start_token=start_token,
        end_token=end_token
    )
    
    # Encode and decode each sequence
    results = []
    total_original_bits = 0
    total_encoded_bits = 0
    
    if verbose:
        print("=" * 70)
        print("Encoding sequences")
        print("=" * 70)
    
    for i, sequence in enumerate(sequences):
        # Encode
        encoded_bits = encoder.encode(sequence)
        
        # Decode to verify
        decoded_sequence = encoder.decode(encoded_bits)
        
        # Statistics
        original_bits = len(sequence) * 11  # Assume 11 bits per symbol (covers -2 to 2047)
        encoded_size = len(encoded_bits)
        compression_ratio = encoded_size / original_bits if original_bits > 0 else 0
        
        # Check correctness
        is_correct = (decoded_sequence == sequence)
        
        results.append((sequence, encoded_bits, decoded_sequence))
        total_original_bits += original_bits
        total_encoded_bits += encoded_size
        
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"Sequence {i:4d}: {len(sequence):4d} symbols -> "
                  f"{encoded_size:6d} bits (ratio {compression_ratio:6.2%}) {status}")
            if not is_correct:
                print(f"  WARNING: Decode mismatch!")
                print(f"  Original:  {sequence[:20]}...")
                print(f"  Decoded:   {decoded_sequence[:20]}...")
    
    if verbose:
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        avg_ratio = total_encoded_bits / total_original_bits if total_original_bits > 0 else 0
        print(f"Total original: {total_original_bits:,} bits")
        print(f"Total encoded:  {total_encoded_bits:,} bits")
        print(f"Average compression ratio: {avg_ratio:.2%}")
        print(f"Bits saved: {total_original_bits - total_encoded_bits:,} bits")
    
    return model, results


def encode_from_file(filename, num_lines=None, n=3, k=0.00001, bits=32,
                     start_token=-1, end_token=-2):
    """
    Read sequences from file and encode using N-gram model.
    
    Parameters:
        filename: Path to input file (each line is a sequence of space-separated integers)
        num_lines: Number of lines to read (None = all lines)
        n: N-gram size
        k: Smoothing parameter
        bits: Precision bits for arithmetic coding
        start_token: Start token index
        end_token: End token index
        
    Returns:
        tuple: (ngram_model, results)
    """
    print("=" * 70)
    print(f"Reading sequences from: {filename}")
    print("=" * 70)
    
    # Read sequences from file
    sequences = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            if num_lines and line_num >= num_lines:
                break
            
            tokens = line.strip().split()
            if tokens:
                try:
                    sequence = [int(token) for token in tokens]
                    sequences.append(sequence)
                except ValueError as e:
                    print(f"Warning: Skipping line {line_num + 1} due to error: {e}")
    
    if not sequences:
        print("No valid sequences found in file!")
        return None, []
    
    print(f"Successfully read {len(sequences)} sequences")
    print()
    
    # Encode with N-gram model
    return encode_sequences_with_ngram(
        sequences=sequences,
        n=n,
        k=k,
        bits=bits,
        start_token=start_token,
        end_token=end_token,
        verbose=True
    )


def test_simple_example():
    """Test with simple example data."""
    print("=" * 70)
    print("SIMPLE EXAMPLE TEST")
    print("=" * 70)
    print()
    
    # Simple test sequences
    sequences = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 2, 3, 5, 6],
        [1, 2, 4, 5, 6],
        [2, 3, 4, 4, 5],
        [3, 3, 3, 3, 3]
    ]
    
    model, results = encode_sequences_with_ngram(
        sequences=sequences,
        n=3,
        k=0.01,
        bits=32,
        verbose=True
    )
    
    # Show some probability distributions
    print()
    print("=" * 70)
    print("Sample probability distributions")
    print("=" * 70)
    
    prob_dist = model.get_probability_distribution()
    
    # Show start context
    start_context = model.get_start_context()
    if start_context in prob_dist:
        print(f"\nContext {start_context} (sentence start):")
        sorted_probs = sorted(prob_dist[start_context].items(), 
                            key=lambda x: -x[1])[:5]
        for char, prob in sorted_probs:
            print(f"  {char}: {prob:.4f}")
    
    # Show some other contexts
    sample_contexts = [(1, 2), (2, 3), (3, 3)]
    for context in sample_contexts:
        if context in prob_dist:
            print(f"\nContext {context}:")
            sorted_probs = sorted(prob_dist[context].items(), 
                                key=lambda x: -x[1])[:5]
            for char, prob in sorted_probs:
                marker = " (END)" if char == model.end_token else ""
                print(f"  {char}{marker}: {prob:.4f}")


if __name__ == "__main__":
    import sys
    
    # Test with simple example
    test_simple_example()
    
    print("\n" * 2)
    
    # If file arguments provided, process them
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        num_lines = int(sys.argv[2]) if len(sys.argv) > 2 else None
        n_value = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        bits_value = int(sys.argv[4]) if len(sys.argv) > 4 else 32
        
        print("=" * 70)
        print(f"FILE PROCESSING: {filename}")
        print("=" * 70)
        print()
        
        model, results = encode_from_file(
            filename=filename,
            num_lines=num_lines,
            n=n_value,
            k=0.00001,
            bits=bits_value
        )
    else:
        print("=" * 70)
        print("Usage for file processing:")
        print("=" * 70)
        print(f"python {sys.argv[0]} <filename> [num_lines] [n] [bits]")
        print()
        print("Example:")
        print(f"python {sys.argv[0]} codes23x40x8.txt 100 3 32")
        print()
        print("Parameters:")
        print("  filename: Input file with sequences (space-separated integers per line)")
        print("  num_lines: Number of lines to process (optional, default=all)")
        print("  n: N-gram size (optional, default=3)")
        print("  bits: Arithmetic coding precision bits (optional, default=32)")
