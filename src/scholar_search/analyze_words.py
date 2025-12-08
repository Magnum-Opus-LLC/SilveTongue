# analyze_words.py
import re
from collections import Counter
from typing import List, Dict


def analyze_words(titles: List[str]) -> Dict[str, float]:
    common_words = {
        "the", "and", "is", "at", "which", "on", "for", "this", "that", "to",
        "in", "a", "with", "as", "it", "be", "by", "was", "of", "or", "are",
        "from", "an", "but", "not", "you", "their", "has", "can", "will",
        "some", "more", "most", "other", "all", "any", "no", "new", "one",
        "two", "three", "four", "five", "six"
    }
    punctuation = {',', '.', ':', ';', '!', '?', '"', '\''}
    word_counts = Counter()
    for title in titles:
        for word_raw in title.split():
            word = ''.join(
                ch for ch in word_raw if ch not in punctuation).lower()
            if word and word not in common_words:
                word_counts[word] += 1
    total = sum(word_counts.values())
    if total == 0:
        return {}
    # Return percentages (as percentage of total counted words)
    return {word: (count / total) * 100 for word, count in word_counts.items()}
