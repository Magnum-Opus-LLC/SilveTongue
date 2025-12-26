"""
PDF cache routes for Silvertongue Dashboard.
"""

import json
from datetime import datetime
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file

from dashboard.utils import get_pdf_cache_dir, get_array_dir
from dashboard.pdf_stock import (
    get_pdf_stock, rebuild_pdf_stock, find_pdf_in_stock,
    clear_pdf_stock_cache, normalize_title
)
from dashboard.pdf_book import (
    get_pdf_book, rebuild_pdf_book, get_pdf_book_stats,
    find_pdf, get_all_pdfs, clear_pdf_book_cache
)

pdf_cache_bp = Blueprint('pdf_cache', __name__)


def get_all_studied_papers():
    """
    Scan all actors' cognitive_loops to build a map of studied papers.
    Returns dict mapping normalized_title -> {actors: [...], studied_at: ..., loop_topic: ...}
    Also maps by filename for PDFs that were actually downloaded.

    Enhanced to also check:
    - discovered_papers in state (papers found during learning)
    - history entries for study actions with timestamps
    - PDF file timestamps to correlate with learning sessions
    """
    array_dir = get_array_dir()
    actors_dir = array_dir / 'actors'
    studied_map = {}  # normalized_title -> study info
    filename_map = {}  # filename -> study info (for direct PDF matches)
    # Track (actor, start_time, end_time, topic) for timestamp matching
    learning_sessions = []

    if not actors_dir.exists():
        return studied_map, filename_map

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
                history = data.get('history', [])
                loop_topic = state.get('topic', 'Unknown')

                # Track learning session time range for PDF timestamp correlation
                started_at = state.get('started_at', '')
                updated_at = state.get('updated_at', '')
                if started_at and updated_at:
                    learning_sessions.append({
                        'actor': actor_name,
                        'start': started_at,
                        'end': updated_at,
                        'topic': loop_topic,
                        'total_studied': state.get('total_papers_studied', 0)
                    })

                # Process knowledge_bits for source_file references
                for kb in state.get('knowledge_bits', []):
                    source_file = kb.get('source_file', '')
                    if source_file and source_file.endswith('.pdf'):
                        filename = Path(source_file).name
                        if filename not in filename_map:
                            filename_map[filename] = {
                                'actors': [],
                                'study_count': 0,
                                'last_studied': '',
                                'topics': []
                            }
                        info = filename_map[filename]
                        if actor_name not in info['actors']:
                            info['actors'].append(actor_name)
                        info['study_count'] += 1
                        if loop_topic and loop_topic not in info['topics']:
                            info['topics'].append(loop_topic)

                # Process discovered_papers - these are papers found and potentially studied
                for paper in state.get('discovered_papers', []):
                    title = paper.get('title', '')
                    if not title:
                        continue
                    normalized = normalize_title(title)
                    if not normalized:
                        continue

                    # Mark as discovered (not confirmed studied, but likely if total_papers_studied > 0)
                    if state.get('total_papers_studied', 0) > 0:
                        if normalized not in studied_map:
                            studied_map[normalized] = {
                                'actors': [],
                                'study_count': 0,
                                'last_studied': updated_at,
                                'topics': [],
                                'original_title': title,
                                'status': 'discovered'  # May have been studied
                            }
                        info = studied_map[normalized]
                        if actor_name not in info['actors']:
                            info['actors'].append(actor_name)
                        if loop_topic and loop_topic not in info['topics']:
                            info['topics'].append(loop_topic)

                for paper_entry in state.get('studied_papers', []):
                    paper = paper_entry.get('paper', {}) or {}
                    title = paper.get('title', '')
                    studied_at = paper_entry.get('studied_at', '')

                    # Check for actual PDF file path in study_result
                    study_result = paper_entry.get('study_result', {})
                    if study_result:
                        sr_paper = study_result.get('paper', {})
                        if sr_paper:
                            file_path = sr_paper.get('file_path', '')
                            if file_path:
                                filename = Path(file_path).name
                                if filename not in filename_map:
                                    filename_map[filename] = {
                                        'actors': [],
                                        'study_count': 0,
                                        'last_studied': '',
                                        'topics': []
                                    }
                                info = filename_map[filename]
                                if actor_name not in info['actors']:
                                    info['actors'].append(actor_name)
                                info['study_count'] += 1
                                if studied_at > info['last_studied']:
                                    info['last_studied'] = studied_at
                                if loop_topic and loop_topic not in info['topics']:
                                    info['topics'].append(loop_topic)

                    # Also track by title for fuzzy matching
                    if not title:
                        continue

                    normalized = normalize_title(title)
                    if not normalized:
                        continue

                    if normalized not in studied_map:
                        studied_map[normalized] = {
                            'actors': [],
                            'study_count': 0,
                            'last_studied': '',
                            'topics': [],
                            'original_title': title,
                            'status': 'studied'
                        }

                    info = studied_map[normalized]
                    info['status'] = 'studied'  # Confirmed studied
                    if actor_name not in info['actors']:
                        info['actors'].append(actor_name)
                    info['study_count'] += 1
                    if studied_at > info['last_studied']:
                        info['last_studied'] = studied_at
                    if loop_topic and loop_topic not in info['topics']:
                        info['topics'].append(loop_topic)

            except Exception as e:
                continue

    # Now correlate PDF file timestamps with learning sessions
    # This helps identify PDFs that were downloaded during learning
    pdf_cache_dir = get_pdf_cache_dir()
    if pdf_cache_dir.exists() and learning_sessions:
        for pdf_file in pdf_cache_dir.glob('*.pdf'):
            try:
                # Get file modification time
                mtime = pdf_file.stat().st_mtime
                pdf_time = datetime.fromtimestamp(mtime).isoformat()

                # Check if this PDF was created during any learning session
                for session in learning_sessions:
                    if session['start'] <= pdf_time <= session['end']:
                        filename = pdf_file.name
                        if filename not in filename_map:
                            filename_map[filename] = {
                                'actors': [],
                                'study_count': 0,
                                'last_studied': pdf_time,
                                'topics': [],
                                'inferred': True  # Inferred from timestamp
                            }
                        info = filename_map[filename]
                        if session['actor'] not in info['actors']:
                            info['actors'].append(session['actor'])
                        if session['topic'] and session['topic'] not in info['topics']:
                            info['topics'].append(session['topic'])
                        if pdf_time > info['last_studied']:
                            info['last_studied'] = pdf_time
                        break  # Found matching session
            except Exception:
                continue

    return studied_map, filename_map


@pdf_cache_bp.route('/api/pdf-cache/stats')
def api_pdf_cache_stats():
    """Get statistics about the PDF cache - uses PDF Book for validation."""
    try:
        # Use PDF Book (has validation stats)
        book = get_pdf_book()
        book.load()
        stats = book.get_stats()

        # Get live study count from cognitive loops (more accurate than PDF Book cache)
        studied_map, filename_map = get_all_studied_papers()

        # Count PDFs that match studied papers
        pdf_cache_dir = get_pdf_cache_dir()
        studied_count = 0
        if pdf_cache_dir.exists():
            for pdf_file in pdf_cache_dir.glob('*.pdf'):
                filename = pdf_file.name
                # Check by filename match
                if filename in filename_map:
                    studied_count += 1
                    continue
                # Check by title match
                title = filename.split(
                    '_', 1)[-1].replace('.pdf', '').replace('_', ' ')
                normalized = normalize_title(title)
                if normalized in studied_map:
                    studied_count += 1

        valid_pdfs = stats.get('valid_pdfs', 0)
        duplicates = stats.get('duplicates', 0)

        return jsonify({
            'exists': True,
            'path': str(get_pdf_cache_dir()),
            'total_pdfs': stats.get('total_pdfs', 0),
            'valid_pdfs': valid_pdfs,
            'invalid_pdfs': stats.get('invalid_pdfs', 0),
            'duplicates': duplicates,
            'total_size_mb': stats.get('total_size_mb', 0),
            'studied_count': studied_count,
            'not_studied_count': max(0, valid_pdfs - studied_count - duplicates),
            'updated_at': book._data.get('updated_at')
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


# Track rebuild status
_rebuild_status = {
    'in_progress': False,
    'started_at': None,
    'result': None
}


@pdf_cache_bp.route('/api/pdf-cache/rebuild', methods=['POST'])
def api_pdf_cache_rebuild():
    """Rebuild the PDF index with validation and duplicate detection.

    OPTIONS:
    - cleanup: true to delete invalid and duplicate PDFs from disk
    - background: true for non-blocking rebuild
    - force: true to proceed even if learning is in progress

    WARNING: This can interrupt learning processes if run concurrently.
    """
    import threading
    from dashboard.utils import active_loops

    # Handle both JSON and form data
    if request.is_json:
        data = request.json or {}
    else:
        data = request.form.to_dict() if request.form else {}
        # Also check query params
        if not data:
            data = request.args.to_dict()

    background = data.get('background', False)
    cleanup = data.get('cleanup', False)

    # Convert string 'true'/'false' to bool (from form/query params)
    if isinstance(background, str):
        background = background.lower() == 'true'
    if isinstance(cleanup, str):
        cleanup = cleanup.lower() == 'true'

    # Warn if learning is in progress
    if active_loops:
        active_actors = list(active_loops.keys())
        force = data.get('force', False)
        if isinstance(force, str):
            force = force.lower() == 'true'
        if not force:
            return jsonify({
                'warning': f'Learning in progress for: {active_actors}. PDF scanning may interfere. Use force=true to proceed anyway.',
                'active_loops': active_actors
            }), 409  # Conflict

    if _rebuild_status['in_progress']:
        return jsonify({
            'status': 'already_running',
            'started_at': _rebuild_status['started_at']
        })

    def do_rebuild():
        _rebuild_status['in_progress'] = True
        _rebuild_status['started_at'] = datetime.now().isoformat()
        try:
            result = rebuild_pdf_book(
                validate=True, detect_duplicates=True, cleanup=cleanup)
            clear_pdf_stock_cache()
            _rebuild_status['result'] = {
                'success': True,
                'total_pdfs': result.get('total', 0),
                'valid_pdfs': result.get('valid', 0),
                'invalid_pdfs': result.get('invalid', 0),
                'duplicates': result.get('duplicates', 0),
                'deleted': result.get('deleted', 0),
                'protected': result.get('protected', 0),
                'cleanup_enabled': cleanup
            }
        except Exception as e:
            _rebuild_status['result'] = {'error': str(e)}
        finally:
            _rebuild_status['in_progress'] = False

    if background:
        threading.Thread(target=do_rebuild, daemon=True).start()
        return jsonify({
            'status': 'started',
            'message': 'Rebuild started in background. Check /api/pdf-cache/rebuild/status for progress.',
            'cleanup_enabled': cleanup
        })

    try:
        do_rebuild()
        return jsonify(_rebuild_status['result'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-cache/rebuild/status')
def api_pdf_cache_rebuild_status():
    """Get the status of a background rebuild."""
    return jsonify({
        'in_progress': _rebuild_status['in_progress'],
        'started_at': _rebuild_status['started_at'],
        'result': _rebuild_status['result']
    })


@pdf_cache_bp.route('/api/pdf-cache/list')
def api_pdf_cache_list():
    """List all valid, unique PDFs with study status (filters invalid/duplicates)."""
    try:
        # Use PDF Book (has validation info)
        book = get_pdf_book()
        book.load()

        # Get studied papers maps (by title and by filename)
        studied_map, filename_map = get_all_studied_papers()

        pdfs = []
        for filename, info in book._data.get('pdfs', {}).items():
            # Skip invalid PDFs
            if info.get('status') != 'valid':
                continue
            # Skip duplicates
            if info.get('is_duplicate'):
                continue

            normalized = info.get('normalized_title', '')

            # Check study status from multiple sources:
            # 1. Direct filename match (most reliable)
            # 2. Title-based match
            # 3. PDF Book's study_info
            filename_study = filename_map.get(filename, {})
            title_study = studied_map.get(normalized, {})
            book_study = info.get('study_info', {})

            is_studied = bool(filename_study) or bool(
                title_study) or book_study.get('studied', False)

            # Merge study info from all sources
            studied_by = list(set(
                filename_study.get('actors', []) +
                title_study.get('actors', []) +
                book_study.get('studied_by', [])
            ))
            study_count = max(
                filename_study.get('study_count', 0),
                title_study.get('study_count', 0),
                book_study.get('study_count', 0)
            )
            last_studied = max(
                filename_study.get('last_studied', ''),
                title_study.get('last_studied', ''),
                book_study.get('last_studied', '')
            )
            topics = list(set(
                filename_study.get('topics', []) +
                title_study.get('topics', []) +
                book_study.get('topics', [])
            ))

            pdfs.append({
                'filename': filename,
                'title': info.get('title', ''),
                'normalized_title': normalized,
                'size_mb': info.get('size_mb', 0),
                'status': info.get('status', 'unknown'),
                # Study status
                'studied': is_studied,
                'studied_by': studied_by,
                'study_count': study_count,
                'last_studied': last_studied,
                'study_topics': topics
            })

        # Sort by title
        pdfs.sort(key=lambda x: x.get('title', '').lower())

        # Get stats from PDF Book
        stats = book.get_stats()
        studied_count = sum(1 for p in pdfs if p['studied'])

        return jsonify({
            'total': len(pdfs),
            'studied_count': studied_count,
            'not_studied_count': len(pdfs) - studied_count,
            'invalid_count': stats.get('invalid_pdfs', 0),
            'duplicate_count': stats.get('duplicates', 0),
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


# ==================== PDF Book API Routes ====================
# These routes use the new PDF Book system for validation,
# duplicate detection, and comprehensive PDF management.


@pdf_cache_bp.route('/api/pdf-book/stats')
def api_pdf_book_stats():
    """Get comprehensive statistics from PDF Book including invalid/duplicate counts."""
    try:
        book = get_pdf_book()
        book.load()
        stats = book.get_stats()

        return jsonify({
            'success': True,
            'stats': stats,
            'book_exists': book.book_path.exists(),
            'book_path': str(book.book_path)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/rebuild', methods=['POST'])
def api_pdf_book_rebuild():
    """
    Rebuild the PDF Book by scanning and validating all PDFs.
    This detects invalid files, duplicates, and updates study info.
    """
    try:
        data = request.get_json() or {}
        validate = data.get('validate', True)
        detect_duplicates = data.get('detect_duplicates', True)

        result = rebuild_pdf_book(validate=validate,
                                  detect_duplicates=detect_duplicates)

        # Clear old pdf_stock cache too
        clear_pdf_stock_cache()

        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/list')
def api_pdf_book_list():
    """
    List all PDFs from PDF Book with validation status and study info.
    Query params:
        - include_invalid: bool (default false)
        - include_duplicates: bool (default false)
        - status_filter: 'valid' | 'invalid' | 'duplicate' | 'studied' | 'not_studied'
    """
    try:
        include_invalid = request.args.get(
            'include_invalid', 'false').lower() == 'true'
        include_duplicates = request.args.get(
            'include_duplicates', 'false').lower() == 'true'
        status_filter = request.args.get('status_filter', '')

        book = get_pdf_book()

        # Get base list
        if status_filter == 'invalid':
            pdfs = book.get_invalid_pdfs()
        elif status_filter == 'duplicate':
            pdfs = book.get_duplicates()
        else:
            pdfs = book.get_all(
                include_invalid=include_invalid or status_filter == 'invalid',
                include_duplicates=include_duplicates or status_filter == 'duplicate'
            )

        # Additional filtering
        if status_filter == 'studied':
            pdfs = [p for p in pdfs if p.get('study_info', {}).get('studied')]
        elif status_filter == 'not_studied':
            pdfs = [p for p in pdfs
                    if not p.get('study_info', {}).get('studied')
                    and p.get('status') == 'valid'
                    and not p.get('is_duplicate')]

        # Get stats
        stats = book.get_stats()

        return jsonify({
            'success': True,
            'total': len(pdfs),
            'stats': stats,
            'pdfs': pdfs
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/invalid')
def api_pdf_book_invalid():
    """Get list of invalid/corrupted PDFs."""
    try:
        book = get_pdf_book()
        invalid_pdfs = book.get_invalid_pdfs()

        return jsonify({
            'success': True,
            'count': len(invalid_pdfs),
            'pdfs': invalid_pdfs
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/duplicates')
def api_pdf_book_duplicates():
    """Get list of duplicate PDFs."""
    try:
        book = get_pdf_book()
        duplicates = book.get_duplicates()

        # Group by original file for easier display
        grouped = {}
        for dup in duplicates:
            original = dup.get('duplicate_of', 'unknown')
            if original not in grouped:
                grouped[original] = {
                    'original': original,
                    'duplicates': []
                }
            grouped[original]['duplicates'].append({
                'filename': dup.get('filename'),
                'size_mb': dup.get('size_mb'),
                'title': dup.get('title')
            })

        return jsonify({
            'success': True,
            'count': len(duplicates),
            'grouped': list(grouped.values()),
            'pdfs': duplicates
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/search')
def api_pdf_book_search():
    """Search for a PDF by title using PDF Book (with fuzzy matching)."""
    try:
        title = request.args.get('title', '')
        if not title:
            return jsonify({'success': False, 'error': 'title parameter required'}), 400

        result = find_pdf(title)

        if result:
            return jsonify({
                'success': True,
                'found': True,
                'pdf': result
            })
        else:
            return jsonify({
                'success': True,
                'found': False,
                'pdf': None
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/cleanup/invalid', methods=['POST'])
def api_pdf_book_cleanup_invalid():
    """
    Remove invalid PDFs from cache.
    POST body: { "dry_run": true|false }
    """
    try:
        data = request.get_json() or {}
        dry_run = data.get('dry_run', True)

        book = get_pdf_book()
        removed = book.remove_invalid(dry_run=dry_run)

        action = "Would remove" if dry_run else "Removed"

        return jsonify({
            'success': True,
            'dry_run': dry_run,
            'message': f"{action} {len(removed)} invalid PDFs",
            'removed': removed
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/cleanup/duplicates', methods=['POST'])
def api_pdf_book_cleanup_duplicates():
    """
    Remove duplicate PDFs from cache.
    POST body: { "dry_run": true|false }
    """
    try:
        data = request.get_json() or {}
        dry_run = data.get('dry_run', True)

        book = get_pdf_book()
        removed = book.remove_duplicates(dry_run=dry_run)

        action = "Would remove" if dry_run else "Removed"

        return jsonify({
            'success': True,
            'dry_run': dry_run,
            'message': f"{action} {len(removed)} duplicate PDFs",
            'removed': removed
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/pdf/<filename>')
def api_pdf_book_get_pdf(filename):
    """Get detailed info about a specific PDF from PDF Book."""
    try:
        book = get_pdf_book()
        pdf_info = book.get_pdf(filename)

        if not pdf_info:
            return jsonify({
                'success': False,
                'error': 'PDF not found in book'
            }), 404

        return jsonify({
            'success': True,
            'pdf': pdf_info
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/pdf/<filename>/metadata', methods=['POST'])
def api_pdf_book_update_metadata(filename):
    """
    Update metadata for a PDF.
    POST body: { "authors": str, "source": str, "url": str, "abstract": str }
    """
    try:
        data = request.get_json() or {}

        book = get_pdf_book()
        success = book.update_metadata(filename, data)

        if success:
            return jsonify({
                'success': True,
                'message': 'Metadata updated'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'PDF not found'
            }), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_cache_bp.route('/api/pdf-book/pdf/<filename>/mark-studied', methods=['POST'])
def api_pdf_book_mark_studied(filename):
    """
    Mark a PDF as studied by an actor.
    POST body: { "actor": str, "topic": str (optional) }
    """
    try:
        data = request.get_json() or {}
        actor = data.get('actor', '')
        topic = data.get('topic', '')

        if not actor:
            return jsonify({
                'success': False,
                'error': 'actor parameter required'
            }), 400

        book = get_pdf_book()
        success = book.mark_studied(filename, actor, topic)

        if success:
            return jsonify({
                'success': True,
                'message': f'Marked as studied by {actor}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'PDF not found'
            }), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
