"""
Test automatic token limit detection and adaptive processing.
"""

from src.utils.chat_utils import get_model_context_window, estimate_tokens, get_safe_chunk_size

print("="*70)
print("MAX TOKEN DETECTION & ADAPTIVE PROCESSING TEST")
print("="*70)

# 1. Current model context window
print("\n1. Current Model Configuration")
print("-"*70)
context_window = get_model_context_window()
print(f"   Context Window: {context_window:,} tokens")
print(f"   In characters: ~{context_window * 4:,} chars (rough estimate)")
print(f"   In words: ~{context_window * 0.75:,.0f} words")

# 2. Test different scenarios
print("\n2. Scenario Testing")
print("-"*70)

scenarios = [
    {"name": "Small paper (5K words)", "words": 5000},
    {"name": "Medium paper (10K words)", "words": 10000},
    {"name": "Large paper (15K words)", "words": 15000},
    {"name": "Very large paper (25K words)", "words": 25000},
]

for scenario in scenarios:
    words = scenario["words"]
    chars = words * 5  # ~5 chars per word
    tokens = estimate_tokens("x" * chars)
    
    print(f"\n   {scenario['name']}:")
    print(f"   - Content: {tokens:,} tokens")
    
    # Calculate how many chunks needed
    overhead = 2000  # prompt + response
    available = context_window - overhead
    chunks_needed = (tokens // available) + 1
    
    print(f"   - Chunks needed: {chunks_needed}")
    print(f"   - Tokens per chunk: {available:,}")
    
    # Calculate synthesis limit
    synthesis_budget = context_window - 4000  # Reserve 4K for prompt/response
    bits_per_token = 150
    max_bits = synthesis_budget // bits_per_token
    
    print(f"   - Max knowledge bits in synthesis: {max_bits}")

# 3. Safe chunk size calculation
print("\n3. Safe Chunk Size Calculation")
print("-"*70)

contexts = [
    {"desc": "No context (just chunk)", "size": 0},
    {"desc": "Small context (paper overview)", "size": 500},
    {"desc": "Medium context (section summaries)", "size": 1000},
    {"desc": "Large context (detailed background)", "size": 2000},
]

for ctx in contexts:
    safe_tokens = get_safe_chunk_size(context_size=ctx["size"], overhead=2000)
    safe_chars = safe_tokens * 4
    safe_words = safe_tokens * 0.75
    
    print(f"\n   {ctx['desc']} ({ctx['size']} tokens):")
    print(f"   - Safe chunk: {safe_tokens:,} tokens")
    print(f"   - In characters: ~{safe_chars:,.0f}")
    print(f"   - In words: ~{safe_words:,.0f}")

# 4. Synthesis capacity
print("\n4. Synthesis Capacity Analysis")
print("-"*70)

synthesis_budget = context_window - 4000
print(f"   Budget for synthesis: {synthesis_budget:,} tokens")
print(f"   (Context window {context_window:,} - 4000 overhead)")
print()

# Estimate knowledge bits capacity
print("   Knowledge bits capacity (by verbosity):")
token_estimates = [
    ("Minimal (50 tokens/bit)", 50),
    ("Concise (100 tokens/bit)", 100),
    ("Normal (150 tokens/bit)", 150),
    ("Detailed (200 tokens/bit)", 200),
    ("Verbose (300 tokens/bit)", 300),
]

for desc, tokens_per_bit in token_estimates:
    max_bits = synthesis_budget // tokens_per_bit
    print(f"   - {desc}: ~{max_bits} bits")

# 5. Real-world example
print("\n5. Real-World Example: LoRA Paper")
print("-"*70)
lora_stats = {
    "words": 13579,
    "chunks": 27,
    "knowledge_bits": 138
}

lora_tokens = estimate_tokens("x" * (lora_stats["words"] * 5))
print(f"   Paper stats:")
print(f"   - Words: {lora_stats['words']:,}")
print(f"   - Estimated tokens: {lora_tokens:,}")
print(f"   - Chunks extracted: {lora_stats['chunks']}")
print(f"   - Knowledge bits: {lora_stats['knowledge_bits']}")
print()

# How many chunks fit?
chunk_budget = get_safe_chunk_size(context_size=500, overhead=2000)
chunks_needed = (lora_tokens // chunk_budget) + 1
print(f"   Processing analysis:")
print(f"   - Tokens per chunk (with context): {chunk_budget:,}")
print(f"   - Chunks needed: {chunks_needed}")
print(f"   - Actual chunks used: {lora_stats['chunks']}")
print()

# Synthesis capacity
bits_budget = (context_window - 4000) // 150
print(f"   Synthesis analysis:")
print(f"   - Max bits for synthesis: {bits_budget}")
print(f"   - Actual bits extracted: {lora_stats['knowledge_bits']}")
print(f"   - Fits in synthesis? {'✅ YES' if lora_stats['knowledge_bits'] <= bits_budget else '❌ NO'}")
print(f"   - Would use: {min(bits_budget, lora_stats['knowledge_bits'])} bits")

# 6. Recommendations
print("\n6. System Recommendations")
print("-"*70)

if context_window >= 32000:
    print("   🚀 Large context window (32K+)")
    print("   - Can process very large papers in fewer chunks")
    print("   - Can synthesize 100+ knowledge bits at once")
    print("   - Recommended: Enable detailed extraction")
elif context_window >= 16000:
    print("   ✅ Medium-large context window (16K+)")
    print("   - Good for most papers")
    print("   - Can synthesize 50-80 knowledge bits")
    print("   - Recommended: Standard extraction mode")
elif context_window >= 8000:
    print("   ⚠️ Standard context window (8K)")
    print("   - Adequate for most papers but requires more chunks")
    print("   - Can synthesize 20-30 knowledge bits")
    print("   - Recommended: Use selective synthesis")
    print("   - Current config: Automatically adapts chunk sizes")
else:
    print("   🔴 Small context window (<8K)")
    print("   - Limited capacity, may struggle with large papers")
    print("   - Can synthesize 10-15 knowledge bits")
    print("   - Recommended: Consider upgrading model")

print("\n" + "="*70)
print("Test complete!")
print("="*70)
print()
print("💡 Tips:")
print("   - System automatically adapts to your context window")
print("   - Larger windows = fewer chunks = faster processing")
print("   - Synthesis automatically limits bits to fit your model")
print(f"   - Your current limit: {context_window:,} tokens")
print()
