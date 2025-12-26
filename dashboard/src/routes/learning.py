"""
Learning & cognitive loop routes for Silvertongue Dashboard.
"""

import json
import queue
import threading
from datetime import datetime
from flask import Blueprint, jsonify, request, Response

from ..utils import get_array_dir, active_loops, loop_outputs, import_project_module
from ..actors import get_actor_root
from ..pdf_stock import find_cached_pdf

learning_bp = Blueprint('learning', __name__)


@learning_bp.route('/api/actor/<actor_name>/sessions')
def api_actor_sessions(actor_name):
    """Get learning sessions for an actor."""
    try:
        array_dir = get_array_dir()
        sessions_dir = array_dir / 'data' / actor_name / 'learning_sessions'

        sessions = []
        if sessions_dir.exists():
            for session_dir in sorted(sessions_dir.iterdir(), reverse=True):
                if session_dir.is_dir():
                    meta_file = session_dir / 'session_meta.json'
                    if meta_file.exists():
                        with open(meta_file, 'r') as f:
                            meta = json.load(f)
                            sessions.append({
                                'session_id': meta.get('session_id', session_dir.name),
                                'topic': meta.get('topic', 'Unknown'),
                                'created_at': meta.get('started_at', ''),
                                'papers_studied': meta.get('total_papers_studied', 0),
                                'qa_pairs': meta.get('qa_pairs_count', 0)
                            })

        return jsonify(sessions[:20])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@learning_bp.route('/api/actor/<actor_name>/loops')
def api_actor_loops(actor_name):
    """Get cognitive loops for an actor."""
    try:
        array_dir = get_array_dir()
        loops_dir = array_dir / 'data' / actor_name / 'cognitive_loops'

        loops = []
        if loops_dir.exists():
            for loop_file in sorted(loops_dir.glob('loop_*.json'), reverse=True):
                try:
                    with open(loop_file, 'r') as f:
                        data = json.load(f)
                        state = data.get('state', {})
                        loops.append({
                            'loop_id': state.get('loop_id', ''),
                            'topic': state.get('topic', 'Unknown'),
                            'phase': state.get('phase', 'unknown'),
                            'iteration': state.get('iteration', 0),
                            'started_at': state.get('started_at', ''),
                            'papers_studied': state.get('total_papers_studied', 0)
                        })
                except:
                    continue

        return jsonify(loops[:20])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@learning_bp.route('/api/actor/<actor_name>/studied-papers')
def api_actor_studied_papers(actor_name):
    """Get all studied papers and their extracted concepts for an actor."""
    import sys
    try:
        array_dir = get_array_dir()
        loops_dir = array_dir / 'data' / actor_name / 'cognitive_loops'

        print(
            f"[studied-papers] Looking for papers in: {loops_dir}", flush=True)
        print(
            f"[studied-papers] Directory exists: {loops_dir.exists()}", flush=True)
        sys.stdout.flush()

        papers = []
        files_checked = 0
        files_with_papers = 0

        if loops_dir.exists():
            for loop_file in sorted(loops_dir.glob('loop_*.json'), reverse=True):
                files_checked += 1
                try:
                    with open(loop_file, 'r') as f:
                        data = json.load(f)

                    state = data.get('state', {})
                    loop_topic = state.get('topic', 'Unknown Topic')
                    loop_id = state.get('loop_id', loop_file.stem)

                    studied_papers = state.get('studied_papers', [])
                    if studied_papers:
                        files_with_papers += 1
                        print(
                            f"[studied-papers] {loop_file.name}: {len(studied_papers)} papers")

                    # Extract studied papers with their concepts
                    for paper_entry in studied_papers:
                        paper = paper_entry.get('paper', {}) or {}
                        study_result = paper_entry.get(
                            'study_result', {}) or {}

                        # Extract knowledge bits/concepts
                        extraction = study_result.get('extraction', {}) or {}
                        study_notes = study_result.get('study_notes', {}) or {}

                        if extraction:
                            # New format from StudyProcessor
                            knowledge_bits = extraction.get(
                                'knowledge_bits', []) or []
                            questions = extraction.get(
                                'questions_raised', []) or []
                            summary = study_notes.get('summary', '') or ''
                            key_concepts = study_notes.get(
                                'key_concepts', []) or []
                        else:
                            # Old format
                            knowledge_bits = study_result.get(
                                'knowledge_bits', []) or []
                            questions = study_result.get('questions', []) or []
                            notes = study_result.get('notes', {}) or {}
                            summary = notes.get('summary', notes.get(
                                'paper_summary', '')) if isinstance(notes, dict) else ''
                            key_concepts = notes.get(
                                'key_concepts', []) if isinstance(notes, dict) else []

                        title = paper.get('title', 'Unknown Paper')
                        # Check if PDF is cached (pass knowledge_bits for source_file matching)
                        pdf_info = find_cached_pdf(title, knowledge_bits)

                        papers.append({
                            'title': title,
                            'authors': paper.get('authors', paper.get('author', '')),
                            'url': paper.get('link', paper.get('url', '')),
                            'pdf_url': paper.get('pdf_url', paper.get('pdf_link', '')),
                            'abstract': paper.get('abstract', paper.get('summary', '')),
                            'source': paper.get('source', 'unknown'),
                            'studied_at': paper_entry.get('studied_at', ''),
                            'loop_topic': loop_topic,
                            'loop_id': loop_id,
                            'summary': summary,
                            'key_concepts': key_concepts,
                            'knowledge_bits': knowledge_bits,
                            'questions_raised': questions,
                            'pdf_cached': pdf_info.get('found', False),
                            'pdf_filename': pdf_info.get('filename', ''),
                            'pdf_size_mb': pdf_info.get('size_mb', 0),
                            'pdf_match_type': pdf_info.get('match_type', '')
                        })
                except Exception as e:
                    print(f"Error loading {loop_file.name}: {e}")
                    continue

        print(
            f"[studied-papers] Checked {files_checked} files, {files_with_papers} had papers, total {len(papers)} papers found")

        # Also check learning_sessions for additional paper data
        sessions_dir = array_dir / 'data' / actor_name / 'learning_sessions'
        if sessions_dir.exists():
            for session_file in sorted(sessions_dir.glob('session_*.json'), reverse=True):
                try:
                    with open(session_file, 'r') as f:
                        session = json.load(f)

                    session_topic = session.get('metadata', {}).get(
                        'topic', session.get('topic', 'Unknown'))

                    for note in session.get('study_notes', []):
                        # Avoid duplicates by checking title
                        title = note.get('title', '')
                        if title and not any(p['title'] == title for p in papers):
                            papers.append({
                                'title': title,
                                'authors': '',
                                'url': note.get('url', ''),
                                'pdf_url': '',
                                'abstract': '',
                                'source': note.get('source_type', 'session'),
                                'studied_at': note.get('timestamp', ''),
                                'loop_topic': session_topic,
                                'loop_id': session_file.stem,
                                'summary': note.get('summary', ''),
                                'key_concepts': note.get('key_points', []),
                                'knowledge_bits': [],
                                'questions_raised': note.get('questions', [])
                            })
                except Exception as e:
                    print(f"Error loading session {session_file.name}: {e}")
                    continue

        return jsonify({
            'total': len(papers),
            'papers': papers
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@learning_bp.route('/api/actor/<actor_name>/learn', methods=['POST'])
def api_start_learning(actor_name):
    """Start a cognitive learning loop."""
    data = request.json
    topic = data.get('topic', '')
    goal = data.get('goal', '')
    mode = data.get('mode', 'standard')
    # LoRA adapter to use (empty = base model)
    adapter = data.get('adapter', '')

    # Configuration options
    max_iterations = data.get('max_iterations', 10)
    fetch_pdfs = data.get('fetch_pdfs', True)
    save_session = data.get('save_session', True)
    auto_save = data.get('auto_save', True)
    max_papers_per_search = data.get('max_papers_per_search', 15)
    max_papers_to_study = data.get('max_papers_to_study', 3)
    search_depth = data.get('search_depth', 2)
    min_citations = data.get('min_citations', 10)
    year_from = data.get('year_from', 2018)
    year_to = data.get('year_to', None)
    goal_threshold = data.get('goal_threshold', 85)

    if not topic:
        return jsonify({'error': 'Topic is required'}), 400

    if not goal:
        goal = f"Develop comprehensive understanding of {topic}"

    # Create output queue for this loop
    loop_id = f"{actor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    loop_outputs[loop_id] = queue.Queue()

    def run_loop():
        try:
            # Lazy import
            cognitive_loop = import_project_module('src.actions.cognitive_loop')
            CognitiveLoop = cognitive_loop.CognitiveLoop

            loop = CognitiveLoop(actor_name, verbose=True)
            loop.configure(
                fetch_pdfs=fetch_pdfs,
                save_session=save_session,
                auto_save=auto_save,
                max_papers_per_search=max_papers_per_search,
                max_papers_to_study=max_papers_to_study,
                search_depth=search_depth,
                min_citations=min_citations,
                year_from=year_from,
                year_to=year_to,
                model_override=adapter if adapter else None  # LoRA adapter support
            )

            # Custom logging to capture output
            original_log = loop._log

            def custom_log(msg):
                loop_outputs[loop_id].put(msg)
                original_log(msg)
            loop._log = custom_log

            active_loops[loop_id] = loop

            # Adjust iterations based on mode
            if mode == 'nonstop':
                # Nonstop mode: truly unlimited - no goal threshold, just keep going
                result = loop.learn_nonstop(
                    topic, goal,
                    # Never stop based on goal (would need 100% which never happens)
                    goal_threshold=100,
                    max_iterations=999999,  # Essentially unlimited
                    checkpoint_interval=1  # Save state after every iteration
                )
            elif mode == 'quick':
                result = loop.learn(topic, goal, max_iterations=1)
            elif mode == 'fast':
                result = loop.learn(
                    topic, goal, max_iterations=min(max_iterations, 3))
            else:
                result = loop.learn(topic, goal, max_iterations)

            loop_outputs[loop_id].put(
                f"\n✅ COMPLETE: {json.dumps(result.get('statistics', {}))}")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            loop_outputs[loop_id].put(f"\n❌ ERROR: {str(e)}")
            loop_outputs[loop_id].put(f"\n📋 Traceback:\n{error_trace}")
        finally:
            if loop_id in active_loops:
                del active_loops[loop_id]

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    return jsonify({
        'loop_id': loop_id,
        'status': 'started',
        'topic': topic,
        'goal': goal,
        'mode': mode
    })


@learning_bp.route('/api/loop/<loop_id>/status')
def api_loop_status(loop_id):
    """Get status of a running loop."""
    if loop_id in active_loops:
        loop = active_loops[loop_id]
        state = loop.get_state()
        return jsonify({
            'running': True,
            'phase': state.phase.value if state else 'unknown',
            'iteration': state.iteration if state else 0,
            'papers_studied': state.total_papers_studied if state else 0,
            'knowledge_bits': state.total_knowledge_bits if state else 0
        })
    return jsonify({'running': False})


@learning_bp.route('/api/loop/<loop_id>/output')
def api_loop_output(loop_id):
    """Stream output from a running loop."""
    def generate():
        if loop_id not in loop_outputs:
            yield f"data: {json.dumps({'error': 'Loop not found'})}\n\n"
            return

        q = loop_outputs[loop_id]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps({'message': msg})}\n\n"
                if '✅ COMPLETE' in msg or '❌ ERROR' in msg:
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@learning_bp.route('/api/tasks/active')
def api_active_tasks():
    """Get all active learning tasks/loops with their status."""
    tasks = []
    for loop_id, loop in active_loops.items():
        try:
            state = loop.get_state()
            # Parse loop_id: actorname_timestamp
            parts = loop_id.split('_')
            actor = parts[0] if parts else 'unknown'
            timestamp = '_'.join(parts[1:]) if len(parts) > 1 else ''

            tasks.append({
                'loop_id': loop_id,
                'actor': actor,
                'started_at': timestamp,
                'topic': state.topic if state else 'unknown',
                'goal': state.goal if state else 'unknown',
                'phase': state.phase.value if state else 'unknown',
                'iteration': state.iteration if state else 0,
                'papers_studied': state.total_papers_studied if state else 0,
                'knowledge_bits': state.total_knowledge_bits if state else 0,
                'status': 'running'
            })
        except Exception as e:
            tasks.append({
                'loop_id': loop_id,
                'error': str(e),
                'status': 'error'
            })

    return jsonify({
        'active_count': len(tasks),
        'tasks': tasks
    })


@learning_bp.route('/api/tasks/stop/<loop_id>', methods=['POST'])
def api_stop_task(loop_id):
    """Stop a running task gracefully."""
    if loop_id in active_loops:
        try:
            loop = active_loops[loop_id]
            loop.pause()  # Gracefully pause the loop
            return jsonify({'success': True, 'message': 'Task paused gracefully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': False, 'error': 'Task not found'}), 404
