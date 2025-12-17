"""
Quick test of Smart Paper Fetcher integration with learning cycle.
"""

from src.actions.study_processor import StudyProcessor, HAS_SMART_FETCHER, HAS_PDF_FETCHER

print("="*70)
print("SMART PAPER FETCHER INTEGRATION TEST")
print("="*70)

print(f"\n✓ PDF Fetcher Available: {HAS_PDF_FETCHER}")
print(f"✓ Smart Fetcher Available: {HAS_SMART_FETCHER}")

if HAS_SMART_FETCHER:
    print("\n🎯 Smart Paper Fetcher is ACTIVE in learning cycle")
    print("   - LLM will analyze search results to find PDF URLs")
    print("   - Multiple extraction strategies will be attempted")
    print("   - PDFs stored in array/data/pdf_cache/ for reuse")
else:
    print("\n⚠️ Smart Fetcher not available")
    print("   Falling back to basic PDF fetcher")

# Test with sample paper
print("\n" + "="*70)
print("TEST: Fetching a known paper")
print("="*70)

processor = StudyProcessor('test_actor')

paper_info = {
    'title': 'LoRA: Low-Rank Adaptation of Large Language Models',
    'url': 'https://arxiv.org/abs/2106.09685',
    'abstract': 'We propose Low-Rank Adaptation...',
    'publication_info': 'Hu et al. - ICLR 2022'
}

study_config = {
    'depth': 'standard',
    'focus_areas': ['LoRA', 'parameter-efficient fine-tuning'],
    'extraction_targets': [
        {'type': 'concept', 'description': 'Key concepts'},
        {'type': 'method', 'description': 'Methods and techniques'}
    ],
    'generate_summary': True,
    'analyze_enlightenment': False  # Skip for quick test
}

print(f"\nPaper: {paper_info['title']}")
print(f"URL: {paper_info['url']}")
print("\nAttempting to fetch and study...")
print("-"*70)

try:
    result = processor.study_from_url(paper_info, study_config)
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    
    content_source = result.get('paper', {}).get('content_source', 'unknown')
    print(f"\nContent Source: {content_source}")
    
    if content_source == 'full_pdf':
        print("✅ SUCCESS - Full PDF was obtained and studied")
        pdf_path = result.get('paper', {}).get('pdf_path')
        print(f"   PDF Location: {pdf_path}")
        
        extraction = result.get('extraction', {})
        print(f"   Chunks Processed: {extraction.get('total_chunks', 0)}")
        print(f"   Knowledge Bits Extracted: {len(extraction.get('knowledge_bits', []))}")
    elif content_source == 'abstract':
        print("⚠️ Only abstract was available")
    else:
        print(f"❌ No content available (source: {content_source})")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test complete!")
print("="*70)
