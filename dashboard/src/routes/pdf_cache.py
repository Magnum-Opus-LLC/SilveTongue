"""
PDF cache routes for Silvertongue Dashboard.
"""

import json
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file

from ..utils import get_pdf_cache_dir, get_array_dir
from ..pdf_stock import (
    get_pdf_stock, rebuild_pdf_stock, find_pdf_in_stock,
    clear_pdf_stock_cache, normalize_title
)

pdf_cache_bp = Blueprint('pdf_cache', __name__)


def get_all_studied_papers():
    """
    Scan all actors' cognitive_loops to build a map of studied papers.
    Returns dict mapping normalized_title -> {actors: [...], studied_at: ..., loop_topic: ...}
    """
    array_dir = get_array_dir()
    actors_dir = array_dir / 'actors'
    studied_map = {}  # normalized_title -> study info
    
    if not actors_dir.exists():
        return studied_map
    
    for actor_dir in actors_dir.iterdir():
        if not actor_dir.is_dir() or actor_dir.name.startswith('.'):
            continue
        
        actor_name = actor_dir.name
        loops_dir = array_dir / 'data' / actor_name / 'cognitive_loops'
        
        if not loops_dir.exists():
            continue
        
        for loop_file in loops_dir.glob('loop_*.json'):
            try:
                with open(loop_file, 'r') as f:
                    data = json.load(f)
                
                state = data.get('state', {})
                loop_topic = state.get('topic', 'Unknown')
                
                for paper_entry in state.get('studied_papers', []):
                    paper = paper_entry.get('paper', {}) or {}
                    title = paper.get('title', '')
                    
                    if not title:
                        continue
                    
                    normalized = normalize_title(title)
                    if not normalized:
                        continue
                    
                    studied_at = paper_entry.get('studied_at', '')
                    
                    if normalized not in studied_map:
                        studied_map[normalized] = {
                            'actors': [],
                            'study_count': 0,
                            'last_studied': '',
                            'topics': [],
                            'original_title': title
                        }
                    
                    info = studied_map[normalized]
                    if actor_name not in info['actors']:
                        info['actors'].append(actor_name)
                    info['study_count'] += 1
                    if studied_at > info['last_studied']:
                        info['last_studied'] = studied_at
                    if loop_topic and loop_topic not in info['topics']:
                        info['topics'].append(loop_topic)
                        
            except Exception as e:
                continue
    
    return studied_map


@pdf_cache_bp.route('/api/pdf-cache/stats')
def api_pdf_cache_stats():
    """Get statistics about the PDF cache from the stock index."""
    try:
        stock = get_pdf_stock()
        studied_map = get_all_studied_papers()
        
        # Count how many cached PDFs have been studied
        studied_count = 0
        for filename, info in stock.get('pdfs', {}).items():
            normalized = info.get('normalized_title', '')
            if normalized in studied_map:
                studied_count += 1

        return jsonify({
            'exists': True,
            'path': str(get_pdf_cache_dir()),
            'total_pdfs': stock.get('total_count', 0),
            'total_size_mb': stock.get('total_size_mb', 0),
            'indexed_titles': len(stock.get('title_index', {})),
            'studied_count': studied_count,
            'not_studied_count': stock.get('total_count', 0) - studied_count,
            'updated_at': stock.get('updated_at')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-cache/stock')
def api_pdf_stock():
    """Get the full PDF stock index."""
    try:
        stock = get_pdf_stock()
        return jsonify(stock)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-cache/rebuild', methods=['POST'])
def api_pdf_cache_rebuild():
    """Rebuild the PDF stock index by scanning the cache directory."""
    try:
        stock = rebuild_pdf_stock()
        # Clear the in-memory cache to force reload
        clear_pdf_stock_cache()

        return jsonify({
            'success': True,
            'total_pdfs': stock.get('total_count', 0),
            'total_size_mb': stock.get('total_size_mb', 0),
            'indexed_titles': len(stock.get('title_index', {}))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-cache/list')
def api_pdf_cache_list():
    """List all PDFs in the cache from the stock index with study status."""
    try:
        stock = get_pdf_stock()
        
        # Get studied papers map
        studied_map = get_all_studied_papers()

        pdfs = []
        for filename, info in stock.get('pdfs', {}).items():
            normalized = info.get('normalized_title', '')
            study_info = studied_map.get(normalized, {})
            
            pdfs.append({
                'filename': filename,
                'title': info.get('original_title', ''),
                'normalized_title': normalized,
                'size_mb': info.get('size_mb', 0),
                'references': info.get('references', []),
                # Study status
                'studied': bool(study_info),
                'studied_by': study_info.get('actors', []),
                'study_count': study_info.get('study_count', 0),
                'last_studied': study_info.get('last_studied', ''),
                'study_topics': study_info.get('topics', [])
            })

        # Sort by title
        pdfs.sort(key=lambda x: x.get('title', '').lower())

        # Count studied vs not studied
        studied_count = sum(1 for p in pdfs if p['studied'])

        return jsonify({
            'total': len(pdfs),
            'studied_count': studied_count,
            'not_studied_count': len(pdfs) - studied_count,
            'pdfs': pdfs
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-cache/search')
def api_pdf_cache_search():
    """Search for a PDF by title in the stock."""
    try:
        title = request.args.get('title', '')
        if not title:
            return jsonify({'error': 'title parameter required'}), 400

        result = find_pdf_in_stock(title)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf/file/<filename>')
def api_serve_pdf(filename):
    """Serve a cached PDF file."""
    try:
        pdf_cache_dir = get_pdf_cache_dir()
        pdf_path = pdf_cache_dir / filename

        if not pdf_path.exists():
            return jsonify({'error': 'PDF not found'}), 404

        # Security: ensure the path is within pdf_cache_dir
        if not str(pdf_path.resolve()).startswith(str(pdf_cache_dir.resolve())):
            return jsonify({'error': 'Invalid path'}), 403

        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
