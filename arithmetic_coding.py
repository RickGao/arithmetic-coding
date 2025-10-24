#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static Arithmetic Coding with N-gram Probability Distribution
"""

from typing import Dict, Tuple, List
from collections import defaultdict


class ArithmeticEncoder:
    """
    Static Arithmetic Encoder using N-gram probability distributions.
    
    This encoder uses context-dependent probability distributions from an N-gram model.
    For each position in the sequence, it selects the appropriate probability distribution
    based on the current context (previous n-1 symbols).
    """

    def __init__(self, ngram_model=None, prob_distribution=None, bits=32, 
                 start_token=-1, end_token=-2):
        """
        Initialize the arithmetic encoder.

        Parameters:
            ngram_model: NGramModel object (optional if prob_distribution is provided)
            prob_distribution: Pre-computed probability distribution dict
                              Format: {(context_tuple): {next_char: probability}}
            bits: Precision in bits for arithmetic coding
            start_token: Start token index (default -1)
            end_token: End token index (default -2, used as EOM)
        """
        self.bits = bits
        self.start_token = start_token
        self.end_token = end_token
        self.EOM = end_token  # End of message marker
        
        # Get probability distribution
        if ngram_model is not None:
            self.prob_distribution = ngram_model.get_probability_distribution()
            self.n = ngram_model.n
            self.vocab = ngram_model.vocab
        elif prob_distribution is not None:
            self.prob_distribution = prob_distribution
            # Infer n from context length
            if self.prob_distribution:
                sample_context = next(iter(self.prob_distribution.keys()))
                self.n = len(sample_context) + 1
            else:
                self.n = 2  # default
            # Build vocabulary from all possible next chars
            self.vocab = set()
            for next_chars in self.prob_distribution.values():
                self.vocab.update(next_chars.keys())
        else:
            raise ValueError("Must provide either ngram_model or prob_distribution")
        
        # Arithmetic coding parameters
        self.full = 1 << self.bits  # 2^bits
        self.half = self.full >> 1  # 2^(bits-1)
        self.quarter = self.half >> 1  # 2^(bits-2)
        
    def get_ranges(self, context: Tuple[int, ...]) -> Dict[int, Tuple[int, int]]:
        """
        Get cumulative probability ranges for a given context.
        
        Parameters:
            context: Tuple of previous (n-1) symbols
            
        Returns:
            Dictionary mapping symbols to (low, high) ranges in [0, full)
        """
        if context not in self.prob_distribution:
            # If context not seen, use uniform distribution
            prob_dict = {char: 1.0 / len(self.vocab) for char in self.vocab}
            # print("1")
        else:
            prob_dict = self.prob_distribution[context]

        # Convert probabilities to cumulative ranges
        ranges = {}
        cumsum = 0
        
        # Sort by symbol for deterministic ordering
        for symbol in sorted(prob_dict.keys()):
            prob = prob_dict[symbol]
            low = int(cumsum * self.full)
            high = int((cumsum + prob) * self.full)
            
            # Ensure high > low (avoid zero-width intervals)
            if high <= low:
                high = low + 1
            
            ranges[symbol] = (low, high)
            cumsum += prob
        
        return ranges

    def encode(self, message: List[int]) -> List[int]:
        """
        Encode a sequence of integers using arithmetic coding.
        
        Parameters:
            message: List of integers to encode (without EOM, will be added)
            
        Returns:
            List of encoded bits
        """
        # Add end token
        full_message = message + [self.end_token]
        
        # Initialize
        low = 0
        high = self.full - 1
        pending_bits = 0
        output_bits = []
        
        # Initialize context with start tokens
        context = tuple([self.start_token] * (self.n - 1))

        
        for symbol in full_message:
            # Get ranges for current context
            ranges = self.get_ranges(context)
            
            if symbol not in ranges:
                raise ValueError(f"Symbol {symbol} not in vocabulary for context {context}")
            
            symbol_low, symbol_high = ranges[symbol]
            
            # Update range
            range_width = high - low + 1
            high = low + (range_width * symbol_high // self.full) - 1
            low = low + (range_width * symbol_low // self.full)
            
            # Output bits and rescale
            while True:
                if high < self.half:
                    # Output 0
                    output_bits.append(0)
                    output_bits.extend([1] * pending_bits)
                    pending_bits = 0
                elif low >= self.half:
                    # Output 1
                    output_bits.append(1)
                    output_bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    # In middle range
                    pending_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break
                
                # Scale up
                low = 2 * low
                high = 2 * high + 1
            
            # Update context (slide window)
            if self.n == 1:
                # 1-gram: context永远是空tuple，不更新
                context = ()
            else:
                # N-gram (n>1): 滑动窗口
                context = context[1:] + (symbol,)
        
        # Output final bits
        pending_bits += 1
        if low < self.quarter:
            output_bits.append(0)
            output_bits.extend([1] * pending_bits)
        else:
            output_bits.append(1)
            output_bits.extend([0] * pending_bits)
        
        return output_bits

    def decode(self, encoded_bits: List[int]) -> List[int]:
        """
        Decode a sequence from arithmetic coded bits.
        
        Parameters:
            encoded_bits: List of bits to decode
            
        Returns:
            Decoded list of integers (without EOM token)
        """
        if not encoded_bits:
            return []
        
        # Initialize
        low = 0
        high = self.full - 1
        value = 0
        
        # Read first 'bits' bits into value
        for i in range(min(self.bits, len(encoded_bits))):
            value = (value << 1) | encoded_bits[i]
        
        # Shift value if we read fewer bits than self.bits
        bits_read = min(self.bits, len(encoded_bits))
        if bits_read < self.bits:
            value <<= (self.bits - bits_read)
        
        bit_index = self.bits
        decoded_message = []
        
        # Initialize context
        context = tuple([self.start_token] * (self.n - 1))
        
        max_iterations = 10000  # Safety limit
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Get ranges for current context
            ranges = self.get_ranges(context)
            
            # Find which symbol the value falls into
            range_width = high - low + 1
            
            # Find symbol by checking each range
            symbol = None
            for sym in sorted(ranges.keys()):
                sym_low, sym_high = ranges[sym]
                
                # Calculate the actual low and high for this symbol
                actual_low = low + (range_width * sym_low // self.full)
                actual_high = low + (range_width * sym_high // self.full) - 1
                
                if actual_low <= value <= actual_high:
                    symbol = sym
                    break
            
            if symbol is None:
                # Could not find symbol - possibly end of stream
                break
            
            # Check for end of message
            if symbol == self.end_token:
                break
            
            decoded_message.append(symbol)
            
            # Update range using the symbol
            symbol_low, symbol_high = ranges[symbol]
            high = low + (range_width * symbol_high // self.full) - 1
            low = low + (range_width * symbol_low // self.full)
            
            # Rescale
            while True:
                if high < self.half:
                    # Do nothing
                    pass
                elif low >= self.half:
                    low -= self.half
                    high -= self.half
                    value -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    low -= self.quarter
                    high -= self.quarter
                    value -= self.quarter
                else:
                    break
                
                # Scale up
                low = 2 * low
                high = 2 * high + 1
                value = 2 * value
                
                # Read next bit
                if bit_index < len(encoded_bits):
                    value |= encoded_bits[bit_index]
                    bit_index += 1
            
            # Update context
            if self.n == 1:
                context = ()
            else:
                context = context[1:] + (symbol,)
        
        return decoded_message


class SimpleArithmeticEncoder:
    """
    Simple wrapper for backward compatibility with frequency-based encoding.
    This is kept for reference but ngram-based encoding is recommended.
    """
    
    def __init__(self, frequencies: Dict[str, int], bits: int = 32, EOM: str = '<EOM>'):
        """
        Initialize simple frequency-based encoder.
        
        Parameters:
            frequencies: Dictionary mapping symbols to frequencies
            bits: Precision bits
            EOM: End of message marker
        """
        self.frequencies = frequencies
        self.bits = bits
        self.EOM = EOM
        self.full = 1 << bits
        
        # Calculate cumulative ranges
        total = sum(frequencies.values())
        self.ranges = {}
        cumsum = 0
        
        for symbol in sorted(frequencies.keys()):
            freq = frequencies[symbol]
            low = int(cumsum * self.full / total)
            high = int((cumsum + freq) * self.full / total)
            if high <= low:
                high = low + 1
            self.ranges[symbol] = (low, high)
            cumsum += freq
    
    def encode(self, message: List[str]) -> List[int]:
        """Encode message using static frequencies."""
        low = 0
        high = self.full - 1
        pending_bits = 0
        output_bits = []
        half = self.full >> 1
        quarter = half >> 1
        
        for symbol in message:
            if symbol not in self.ranges:
                raise ValueError(f"Symbol {symbol} not in frequency table")
            
            symbol_low, symbol_high = self.ranges[symbol]
            range_width = high - low + 1
            high = low + (range_width * symbol_high // self.full) - 1
            low = low + (range_width * symbol_low // self.full)
            
            while True:
                if high < half:
                    output_bits.append(0)
                    output_bits.extend([1] * pending_bits)
                    pending_bits = 0
                elif low >= half:
                    output_bits.append(1)
                    output_bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low -= half
                    high -= half
                elif low >= quarter and high < 3 * quarter:
                    pending_bits += 1
                    low -= quarter
                    high -= quarter
                else:
                    break
                low = 2 * low
                high = 2 * high + 1
        
        pending_bits += 1
        if low < quarter:
            output_bits.append(0)
            output_bits.extend([1] * pending_bits)
        else:
            output_bits.append(1)
            output_bits.extend([0] * pending_bits)
        
        return output_bits
    
    def decode(self, encoded_bits: List[int]) -> List[str]:
        """Decode message using static frequencies."""
        if not encoded_bits:
            return []
        
        low = 0
        high = self.full - 1
        value = 0
        half = self.full >> 1
        quarter = half >> 1
        
        for i in range(min(self.bits, len(encoded_bits))):
            value = (value << 1) | encoded_bits[i]
        
        bit_index = self.bits
        decoded_message = []
        
        while True:
            range_width = high - low + 1
            scaled_value = ((value - low + 1) * self.full - 1) // range_width
            
            symbol = None
            for sym in sorted(self.ranges.keys()):
                sym_low, sym_high = self.ranges[sym]
                if sym_low <= scaled_value < sym_high:
                    symbol = sym
                    break
            
            if symbol is None or symbol == self.EOM:
                break
            
            decoded_message.append(symbol)
            
            symbol_low, symbol_high = self.ranges[symbol]
            high = low + (range_width * symbol_high // self.full) - 1
            low = low + (range_width * symbol_low // self.full)
            
            while True:
                if high < half:
                    pass
                elif low >= half:
                    low -= half
                    high -= half
                    value -= half
                elif low >= quarter and high < 3 * quarter:
                    low -= quarter
                    high -= quarter
                    value -= quarter
                else:
                    break
                
                low = 2 * low
                high = 2 * high + 1
                value = 2 * value
                
                if bit_index < len(encoded_bits):
                    value |= encoded_bits[bit_index]
                    bit_index += 1
        
        return decoded_message
