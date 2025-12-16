#!/usr/bin/env python
"""
Test PDF Fetching

Quick test to verify Puppeteer-based PDF fetching works.
"""

import sys
sys.path.insert(0, '/Volumes/ext_disk/LLM_Projects/silvertongue')

from src.utils.pdf_fetcher import PDFFetcher, PaperContentExtractor

def test_arxiv_pdf():
    """Test fetching a PDF from arXiv."""
    print("=" * 60)
    print("Testing PDF Fetcher")
    print("=" * 60)
    
    fetcher = PDFFetcher()
    
    # Check puppeteer is available
    if not fetcher._check_puppeteer_installed():
        print("❌ Puppeteer not installed!")
        print("   Run: npm install puppeteer")
        return
    
    print("✅ Puppeteer available")
    
    # Test with an arXiv paper (using abs URL to test conversion)
    test_url = "https://arxiv.org/abs/2312.00752"  # Mamba paper
    print(f"\nFetching PDF from: {test_url}")
    
    result = fetcher.fetch_pdf(test_url, "Test Paper")
    
    if result['success']:
        print(f"✅ PDF fetched successfully!")
        print(f"   Path: {result['pdf_path']}")
        print(f"   Size: {result['metadata'].get('file_size', 0) / 1024:.1f} KB")
        print(f"   Source: {result['source_type']}")
        print(f"   Cached: {result['cached']}")
        
        # Try extracting content
        print("\n📖 Extracting content...")
        extractor = PaperContentExtractor(fetcher)
        
        paper_info = {
            'url': test_url,
            'title': 'Test Paper',
            'abstract': 'Test abstract'
        }
        
        content = extractor.extract_from_paper(paper_info)
        
        if content['success']:
            print(f"✅ Content extracted!")
            print(f"   Words: {content['metadata'].get('word_count', 0)}")
            print(f"   Chunks: {len(content['chunks'])}")
            sections = content.get('sections', [])
            unique_sections = list(dict.fromkeys([s['name'] for s in sections]))
            print(f"   Sections: {unique_sections}")
            print(f"\n   First 500 chars:")
            print(f"   {content['content'][:500]}...")
        else:
            print(f"❌ Content extraction failed: {content.get('error')}")
    else:
        print(f"❌ PDF fetch failed: {result.get('error')}")
        print(f"   Metadata: {result.get('metadata', {})}")


def test_from_search_result():
    """Test fetching from a paper search result dict."""
    print("\n" + "=" * 60)
    print("Testing PDF Fetch from Search Result")
    print("=" * 60)
    
    # Simulate a search result
    paper = {
        'title': 'Attention Is All You Need',
        'url': 'https://arxiv.org/abs/1706.03762',
        'abstract': 'The dominant sequence transduction models...',
        'authors': ['Vaswani et al.'],
        'year': 2017,
        'citations': 50000
    }
    
    print(f"Paper: {paper['title']}")
    
    extractor = PaperContentExtractor()
    result = extractor.extract_from_paper(paper)
    
    if result['success']:
        source = result.get('metadata', {}).get('content_source', 'unknown')
        print(f"✅ Content extracted! (Source: {source})")
        print(f"   Words: {result['metadata'].get('word_count', 0)}")
        print(f"   Chunks: {len(result['chunks'])}")
    else:
        print(f"⚠️ Result: {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    test_arxiv_pdf()
    # test_from_search_result()  # Uncomment to also test this
