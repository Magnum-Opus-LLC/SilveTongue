# generate_histogram.py
import math
from typing import List, Dict


def generate_histogram(citations: List[int]) -> Dict[str, float]:
    if not citations:
        return {}
    min_val = min(citations)
    max_val = max(citations)
    n = len(citations)
    # Use square-root rule but ensure at least 10 bins
    num_bins = max(round(math.sqrt(n)), 10)
    bin_size = 1 if max_val == min_val else math.ceil(
        (max_val - min_val) / num_bins)
    bins = {}
    for citation in citations:
        bin_index = (citation - min_val) // bin_size
        bin_start = bin_index * bin_size + min_val
        bin_end = (bin_index + 1) * bin_size + min_val - 1
        bin_range = f"{bin_start}-{bin_end}"
        bins[bin_range] = bins.get(bin_range, 0) + 1
    total = len(citations)
    return {bin_range: (count / total) * 100 for bin_range, count in bins.items()}
