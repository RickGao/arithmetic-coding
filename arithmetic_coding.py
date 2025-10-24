"""
Static Arithmetic Coding with N-gram Probability Distribution (Memory Optimized)
"""

from typing import Dict, Tuple, List, Optional, Set
from collections import defaultdict


class ArithmeticEncoder:
    """
    Memory-optimized Static Arithmetic Encoder using N-gram probability distributions.

    Optimizations:
    - __slots__ to reduce per-instance memory overhead
    - Cached get_ranges results to avoid recomputation
    - bytearray for bit storage instead of list
    - Lazy vocabulary building
    - Avoided unnecessary list copies
    """

    __slots__ = ['bits', 'start_token', 'end_token', 'EOM', 'prob_distribution',
                 'n', '_vocab', 'full', 'half', 'quarter', '_ranges_cache',
                 '_sorted_symbols_cache']

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
            self._vocab = ngram_model.vocab  # Use existing vocab if available
        elif prob_distribution is not None:
            self.prob_distribution = prob_distribution
            # Infer n from context length
            if self.prob_distribution:
                sample_context = next(iter(self.prob_distribution.keys()))
                self.n = len(sample_context) + 1
            else:
                self.n = 2  # default
            # Lazy vocabulary building - will be built on first access
            self._vocab = None
        else:
            raise ValueError("Must provide either ngram_model or prob_distribution")

        # Arithmetic coding parameters
        self.full = 1 << self.bits  # 2^bits
        self.half = self.full >> 1  # 2^(bits-1)
        self.quarter = self.half >> 1  # 2^(bits-2)

        # Caches for performance
        self._ranges_cache = {}
        self._sorted_symbols_cache = {}

    @property
    def vocab(self) -> Set[int]:
        """Lazy vocabulary building."""
        if self._vocab is None:
            self._vocab = set()
            for next_chars in self.prob_distribution.values():
                self._vocab.update(next_chars.keys())
        return self._vocab

    def get_ranges(self, context: Tuple[int, ...]) -> Dict[int, Tuple[int, int]]:
        """
        Get cumulative probability ranges for a given context (with caching).

        Parameters:
            context: Tuple of previous (n-1) symbols

        Returns:
            Dictionary mapping symbols to (low, high) ranges in [0, full)
        """
        # Check cache first
        if context in self._ranges_cache:
            return self._ranges_cache[context]

        if context not in self.prob_distribution:
            # If context not seen, use uniform distribution
            vocab = self.vocab
            prob_dict = {char: 1.0 / len(vocab) for char in vocab}
        else:
            prob_dict = self.prob_distribution[context]

        # Get or create sorted symbols list for this context
        if context not in self._sorted_symbols_cache:
            self._sorted_symbols_cache[context] = sorted(prob_dict.keys())
        sorted_symbols = self._sorted_symbols_cache[context]

        # Convert probabilities to cumulative ranges
        ranges = {}
        cumsum = 0

        for symbol in sorted_symbols:
            prob = prob_dict[symbol]
            low = int(cumsum * self.full)
            high = int((cumsum + prob) * self.full)

            # Ensure high > low (avoid zero-width intervals)
            if high <= low:
                high = low + 1

            ranges[symbol] = (low, high)
            cumsum += prob

        # Cache the result
        self._ranges_cache[context] = ranges
        return ranges

    def encode(self, message: List[int]) -> bytearray:
        """
        Encode a sequence of integers using arithmetic coding.

        Parameters:
            message: List of integers to encode (without EOM, will be added)

        Returns:
            bytearray of encoded bits (0s and 1s stored as bytes)
        """
        # Initialize
        low = 0
        high = self.full - 1
        pending_bits = 0
        output_bits = bytearray()  # More memory efficient than list

        # Initialize context with start tokens
        context = tuple([self.start_token] * (self.n - 1))

        # Process message + end token (avoid list copy)
        message_length = len(message)

        for i in range(message_length + 1):
            # Get current symbol (message or end token)
            symbol = message[i] if i < message_length else self.end_token

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

    def decode(self, encoded_bits: bytearray) -> List[int]:
        """
        Decode a sequence from arithmetic coded bits.

        Parameters:
            encoded_bits: bytearray or list of bits to decode

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
        bits_to_read = min(self.bits, len(encoded_bits))
        for i in range(bits_to_read):
            value = (value << 1) | encoded_bits[i]

        # Shift value if we read fewer bits than self.bits
        if bits_to_read < self.bits:
            value <<= (self.bits - bits_to_read)

        bit_index = self.bits
        decoded_message = []

        # Initialize context
        context = tuple([self.start_token] * (self.n - 1))

        max_iterations = 10000  # Safety limit
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # Get ranges for current context (cached)
            ranges = self.get_ranges(context)

            # Find which symbol the value falls into
            range_width = high - low + 1

            # Get sorted symbols for deterministic search
            if context not in self._sorted_symbols_cache:
                self._sorted_symbols_cache[context] = sorted(ranges.keys())
            sorted_symbols = self._sorted_symbols_cache[context]

            # Find symbol by checking each range
            symbol = None
            for sym in sorted_symbols:
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

    def clear_cache(self):
        """Clear all caches to free memory if needed."""
        self._ranges_cache.clear()
        self._sorted_symbols_cache.clear()


class SimpleArithmeticEncoder:
    """
    Memory-optimized simple wrapper for backward compatibility with frequency-based encoding.
    """

    __slots__ = ['frequencies', 'bits', 'EOM', 'full', 'ranges', 'half', 'quarter']

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
        self.half = self.full >> 1
        self.quarter = self.half >> 1

        # Calculate cumulative ranges (pre-computed, not cached per call)
        total = sum(frequencies.values())
        self.ranges = {}
        cumsum = 0

        sorted_symbols = sorted(frequencies.keys())
        for symbol in sorted_symbols:
            freq = frequencies[symbol]
            low = int(cumsum * self.full / total)
            high = int((cumsum + freq) * self.full / total)
            if high <= low:
                high = low + 1
            self.ranges[symbol] = (low, high)
            cumsum += freq

    def encode(self, message: List[str]) -> bytearray:
        """Encode message using static frequencies."""
        low = 0
        high = self.full - 1
        pending_bits = 0
        output_bits = bytearray()  # More memory efficient

        for symbol in message:
            if symbol not in self.ranges:
                raise ValueError(f"Symbol {symbol} not in frequency table")

            symbol_low, symbol_high = self.ranges[symbol]
            range_width = high - low + 1
            high = low + (range_width * symbol_high // self.full) - 1
            low = low + (range_width * symbol_low // self.full)

            while True:
                if high < self.half:
                    output_bits.append(0)
                    output_bits.extend([1] * pending_bits)
                    pending_bits = 0
                elif low >= self.half:
                    output_bits.append(1)
                    output_bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < 3 * self.quarter:
                    pending_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break
                low = 2 * low
                high = 2 * high + 1

        pending_bits += 1
        if low < self.quarter:
            output_bits.append(0)
            output_bits.extend([1] * pending_bits)
        else:
            output_bits.append(1)
            output_bits.extend([0] * pending_bits)

        return output_bits

    def decode(self, encoded_bits: bytearray) -> List[str]:
        """Decode message using static frequencies."""
        if not encoded_bits:
            return []

        low = 0
        high = self.full - 1
        value = 0

        bits_to_read = min(self.bits, len(encoded_bits))
        for i in range(bits_to_read):
            value = (value << 1) | encoded_bits[i]

        bit_index = self.bits
        decoded_message = []

        # Pre-sort symbols for deterministic iteration
        sorted_symbols = sorted(self.ranges.keys())

        while True:
            range_width = high - low + 1
            scaled_value = ((value - low + 1) * self.full - 1) // range_width

            symbol = None
            for sym in sorted_symbols:
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
                if high < self.half:
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

                low = 2 * low
                high = 2 * high + 1
                value = 2 * value

                if bit_index < len(encoded_bits):
                    value |= encoded_bits[bit_index]
                    bit_index += 1

        return decoded_message