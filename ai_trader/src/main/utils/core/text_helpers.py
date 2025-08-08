"""
Text processing and comparison utilities.

This module provides text manipulation, comparison, and similarity calculation
functions used throughout the AI Trader system, particularly for deduplication
and text analysis tasks.
"""

import re
import string
from typing import Set, Tuple, Optional, List, Dict
from difflib import SequenceMatcher


def normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for comparison by removing special characters,
    converting to lowercase, and standardizing whitespace.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def tokenize_text(text: str, normalize: bool = True) -> Set[str]:
    """
    Tokenize text into a set of words for comparison.
    
    Args:
        text: Input text to tokenize
        normalize: Whether to normalize text before tokenizing
        
    Returns:
        Set of word tokens
    """
    if normalize:
        text = normalize_text_for_comparison(text)
    
    # Split into words and filter out empty strings
    words = [word for word in text.split() if word]
    
    # Return as set for efficient comparison
    return set(words)


def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    The Jaccard similarity is the size of the intersection divided by
    the size of the union of the sets.
    
    Args:
        set1: First set of items
        set2: Second set of items
        
    Returns:
        Jaccard similarity score between 0 and 1
    """
    if not set1 and not set2:
        return 1.0  # Both empty sets are considered identical
    
    if not set1 or not set2:
        return 0.0  # One empty set means no similarity
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_text_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
    """
    Calculate similarity between two text strings.
    
    This is the main text similarity function that can use different
    similarity calculation methods.
    
    Args:
        text1: First text string
        text2: Second text string
        method: Similarity method to use ('jaccard', 'sequence', 'token_sort')
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Quick exact match check
    if text1 == text2:
        return 1.0
    
    if method == 'jaccard':
        # Token-based Jaccard similarity
        tokens1 = tokenize_text(text1)
        tokens2 = tokenize_text(text2)
        return calculate_jaccard_similarity(tokens1, tokens2)
    
    elif method == 'sequence':
        # Character sequence similarity using difflib
        normalized1 = normalize_text_for_comparison(text1)
        normalized2 = normalize_text_for_comparison(text2)
        return SequenceMatcher(None, normalized1, normalized2).ratio()
    
    elif method == 'token_sort':
        # Sort tokens before comparison (order-independent)
        tokens1 = sorted(tokenize_text(text1))
        tokens2 = sorted(tokenize_text(text2))
        
        # Compare sorted token strings
        sorted_text1 = ' '.join(tokens1)
        sorted_text2 = ' '.join(tokens2)
        return SequenceMatcher(None, sorted_text1, sorted_text2).ratio()
    
    else:
        # Default to Jaccard
        return calculate_text_similarity(text1, text2, method='jaccard')


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change
    one string into the other.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance (integer)
    """
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    # Create distance matrix
    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Initialize first column and row
    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j
    
    # Calculate distances
    for i in range(1, rows):
        for j in range(1, cols):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1
            
            dist[i][j] = min(
                dist[i-1][j] + 1,      # deletion
                dist[i][j-1] + 1,      # insertion
                dist[i-1][j-1] + cost  # substitution
            )
    
    return dist[-1][-1]


def normalize_levenshtein_distance(s1: str, s2: str) -> float:
    """
    Calculate normalized Levenshtein distance between two strings.
    
    The normalized distance is the Levenshtein distance divided by
    the length of the longer string, giving a value between 0 and 1.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Normalized Levenshtein distance (0 to 1, where 0 is identical)
    """
    if not s1 and not s2:
        return 0.0
    
    distance = calculate_levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    if max_len == 0:
        return 0.0
    
    return distance / max_len


def find_common_phrases(texts: List[str], min_length: int = 3, min_occurrences: int = 2) -> Dict[str, int]:
    """
    Find common phrases across multiple texts.
    
    Useful for identifying repeated content or patterns in text data.
    
    Args:
        texts: List of text strings to analyze
        min_length: Minimum phrase length in words
        min_occurrences: Minimum times a phrase must appear
        
    Returns:
        Dictionary mapping phrases to their occurrence counts
    """
    from collections import defaultdict
    
    phrase_counts = defaultdict(int)
    
    for text in texts:
        # Normalize and tokenize
        words = tokenize_text(text, normalize=True)
        words_list = list(words)
        
        # Extract phrases of different lengths
        for phrase_len in range(min_length, len(words_list) + 1):
            for i in range(len(words_list) - phrase_len + 1):
                phrase = ' '.join(words_list[i:i + phrase_len])
                phrase_counts[phrase] += 1
    
    # Filter by minimum occurrences
    return {
        phrase: count
        for phrase, count in phrase_counts.items()
        if count >= min_occurrences
    }


def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
    """
    Extract key phrases from text using simple heuristics.
    
    This is a simplified key phrase extraction that looks for
    capitalized sequences and common patterns.
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to return
        
    Returns:
        List of key phrases
    """
    # Find capitalized sequences (potential proper nouns/key terms)
    capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    capitalized_phrases = re.findall(capitalized_pattern, text)
    
    # Find quoted phrases
    quoted_pattern = r'"([^"]+)"|\'([^\']+)\''
    quoted_matches = re.findall(quoted_pattern, text)
    quoted_phrases = [match[0] or match[1] for match in quoted_matches]
    
    # Combine and deduplicate
    all_phrases = list(set(capitalized_phrases + quoted_phrases))
    
    # Sort by length (longer phrases often more specific)
    all_phrases.sort(key=len, reverse=True)
    
    return all_phrases[:max_phrases]


# Re-export main functions for backward compatibility
__all__ = [
    'calculate_text_similarity',
    'normalize_text_for_comparison',
    'tokenize_text',
    'calculate_jaccard_similarity',
    'calculate_levenshtein_distance',
    'normalize_levenshtein_distance',
    'find_common_phrases',
    'extract_key_phrases'
]