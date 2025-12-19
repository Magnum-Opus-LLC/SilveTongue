"""
Silvertongue Dashboard - Web interface for managing actors and cognitive loops.

Reads configuration from config.json including array_dir path.
"""

import sys
from pathlib import Path
import json
import yaml
from datetime import datetime
import threading
import queue

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify, request, Response

app = Flask(__name__)

# Global state for running loops
active_loops = {}
loop_outputs = {}  # Store output queues for each loop


def load_config():
    """Load configuration from config.json."""
    config_path = project_root / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def get_array_dir():
    """Get the array directory from config.json."""
    config = load_config()
    array_dir = config.get('array_dir', '')
    if array_dir:
        return Path(array_dir)
    return project_root / 'array'


def load_yaml_file(file_path):
    """Load a YAML file safely."""
    if file_path.exists():
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def get_actor_root(actor_name):
    """Get the root directory for an actor."""
    if actor_name == 'template':
        return project_root / 'actors' / actor_name
    return get_array_dir() / 'actors' / actor_name


def get_available_actors():
    """Get list of available actors."""
    array_dir = get_array_dir()
    actors_dir = array_dir / 'actors'
    
    actors = []
    if actors_dir.exists():
        for item in actors_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if (item / 'behavior.json').exists() or (item / 'characteristic').exists():
                    actors.append(item.name)
    
    # Also check template in project root
    template_dir = project_root / 'actors' / 'template'
    if template_dir.exists() and 'template' not in actors:
        actors.append('template')
    
    return sorted(actors)


def get_actor_details(actor_name):
    """Get detailed information about an actor (lightweight, no heavy imports)."""
    actor_root = get_actor_root(actor_name)
    
    # Define paths
    paths = {
        'principles': actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
        'adjustments': actor_root / 'expertise' / 'domain' / 'adjustment' / 'adjustments.yaml',
        'core_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
        'supporting_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
        'experiences': actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
        'challenges': actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
        'values': actor_root / 'characteristic' / 'value' / 'values.yaml',
        'learning_book': actor_root / 'learning_book.json'
    }
    
    def get_list(key):
        data = load_yaml_file(paths[key])
        return data.get(key, [])
    
    # Load learning book
    learning_book = {}
    if paths['learning_book'].exists():
        with open(paths['learning_book'], 'r') as f:
            learning_book = json.load(f)
    
    # Get archive history
    archive_root = actor_root / 'archive'
    archive_history = []
    if archive_root.exists():
        for key_dir in archive_root.iterdir():
            if key_dir.is_dir():
                for f in sorted(key_dir.glob('*'), reverse=True)[:3]:
                    archive_history.append({
                        'key': key_dir.name,
                        'file': f.name,
                        'path': str(f),
                        'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    })
    
    return {
        'name': actor_name,
        'path': str(actor_root),
        'values': get_list('values'),
        'principles': get_list('principles'),
        'experiences': get_list('experiences'),
        'core_knowledge': get_list('core_knowledge'),
        'supporting_knowledge': get_list('supporting_knowledge'),
        'challenges': get_list('challenges'),
        'adjustments': get_list('adjustments'),
        'mindset': learning_book.get('current_mindset', {}),
        'recent_learnings': learning_book.get('learning_history', [])[-10:],
        'archive_history': archive_history[:20]
    }


def add_to_yaml_list(file_path, key, item, source=None):
    """Add an item to a YAML list file."""
    data = load_yaml_file(file_path)
    items = data.get(key, [])
    
    if source:
        item = f"{item} [Source: {source}]"
    
    # Avoid duplicates
    item_lower = item.lower()[:50]
    for existing in items:
        if existing.lower()[:50] == item_lower:
            return False
    
    items.append(item)
    data[key] = items
    data['_meta'] = {
        'last_updated': datetime.now().isoformat(),
        'updated_by': 'dashboard'
    }
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return True


# ==================== Routes ====================

@app.route('/')
def index():
    """Main dashboard page."""
    actors = get_available_actors()
    return render_template('index.html', actors=actors)


@app.route('/api/actors')
def api_actors():
    """Get list of actors."""
    return jsonify(get_available_actors())


@app.route('/api/actor/<actor_name>')
def api_actor_details(actor_name):
    """Get actor details."""
    try:
        details = get_actor_details(actor_name)
        return jsonify(details)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/knowledge/<knowledge_type>')
def api_actor_knowledge(actor_name, knowledge_type):
    """Get specific knowledge type for an actor."""
    try:
        details = get_actor_details(actor_name)
        if knowledge_type in details:
            return jsonify({knowledge_type: details[knowledge_type]})
        return jsonify({'error': 'Invalid knowledge type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/knowledge/<knowledge_type>', methods=['POST'])
def api_add_knowledge(actor_name, knowledge_type):
    """Add new knowledge item."""
    actor_root = get_actor_root(actor_name)
    data = request.json
    content = data.get('content', '')
    source = data.get('source', 'dashboard')
    
    paths = {
        'principles': actor_root / 'expertise' / 'domain' / 'principal' / 'principles.yaml',
        'experiences': actor_root / 'expertise' / 'domain' / 'experience' / 'experiences.yaml',
        'core_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'core' / 'core_knowledge.yaml',
        'supporting_knowledge': actor_root / 'expertise' / 'domain' / 'knowledge' / 'supporting' / 'supporting_knowledge.yaml',
        'challenges': actor_root / 'expertise' / 'domain' / 'challenge' / 'challenges.yaml',
        'adjustments': actor_root / 'expertise' / 'domain' / 'adjustment' / 'adjustments.yaml'
    }
    
    if knowledge_type not in paths:
        return jsonify({'error': 'Invalid knowledge type'}), 400
    
    success = add_to_yaml_list(paths[knowledge_type], knowledge_type, content, source)
    return jsonify({'success': success})


@app.route('/api/actor/<actor_name>/sessions')
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


@app.route('/api/actor/<actor_name>/loops')
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


@app.route('/api/actor/<actor_name>/chat', methods=['POST'])
def api_chat(actor_name):
    """Chat with an actor - lightweight cognitive mode for chat only."""
    data = request.json
    message = data.get('message', '')
    images = data.get('images', [])  # Array of {data: base64, type: mime}
    history = data.get('history', [])
    use_cognitive = data.get('use_cognitive', True)
    
    if not message and not images:
        return jsonify({'error': 'No message or images provided'}), 400
    
    # Clean history - only keep role and content (remove extra fields like 'cognitive')
    clean_history = [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in history
        if msg.get("role") and msg.get("content")
    ]
    
    try:
        from src.utils.chat_utils import get_chat_completion
        from src.utils.knowledge_base import KnowledgeBase
        from src.thoughts.subconscious import SilvertongueSubconscious
        from src.thoughts.conscious import SilvertongueConscious
        
        # Initialize knowledge base and thought processors
        kb = KnowledgeBase(actor_name)
        subconscious = SilvertongueSubconscious(kb)
        conscious = SilvertongueConscious(kb)
        
        details = get_actor_details(actor_name)
        
        # Build system prompt from actor's knowledge - increase limits for better context
        values = details.get('values', [])[:15]
        principles = details.get('principles', [])[:20]
        experiences = details.get('experiences', [])[:15]
        core_knowledge = details.get('core_knowledge', [])[:30]
        supporting_knowledge = details.get('supporting_knowledge', [])[:15]
        recent_learnings = details.get('recent_learnings', [])[:10]
        
        if use_cognitive:
            # PROPER COGNITIVE MODE: Use subconscious -> conscious pipeline
            
            # Step 1: Subconscious analysis
            analysis = subconscious.analyze_request(message)
            
            # Step 2: Conscious processing with full context
            conscious_prompt = conscious.construct_conscious_prompt(analysis)
            
            # Step 3: Build final response with full knowledge context
            system_prompt = f"""{conscious_prompt}

You are responding to a chat message. Draw upon:
- Your principles and values
- The relevant experiences and knowledge identified by your subconscious
- Your accumulated learning

User's message: {message}

Provide a thoughtful, helpful response that reflects your knowledge and character."""

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(clean_history[-10:])
            
            # Add user message with images if present
            if images and len(images) > 0:
                user_content = []
                if message:
                    user_content.append({"type": "text", "text": message})
                for img in images:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": img['data']}
                    })
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": message})
            
            # Use base model from config (LoRA adapters used when available)
            response = get_chat_completion(messages, temperature=0.7)
            
            # Save chat history
            _save_chat_turn(actor_name, message, response)
            
            return jsonify({
                'response': response,
                'actor': actor_name,
                'cognitive': True,
                'system_prompt': system_prompt,
                'subconscious_analysis': analysis,
                'conscious_prompt': conscious_prompt,
                'analysis': analysis,
                'thought_history': [
                    {"step": "Subconscious Analysis", "content": f"Domain: {analysis.get('domain', 'Unknown')}"},
                    {"step": "Conscious Processing", "content": "Applied principles and relevant knowledge"}
                ]
            })
        else:
            # Non-cognitive mode: Still use subconscious for context but simpler processing
            analysis = subconscious.analyze_request(message)
            
            system_prompt = f"""You are {actor_name}.

Domain: {analysis.get('domain', 'General')}

Relevant Knowledge:
{chr(10).join(f'- {k}' for k in analysis.get('relevant_knowledge', [])[:10])}

Relevant Experiences:
{chr(10).join(f'- {e}' for e in analysis.get('relevant_experiences', [])[:5])}

Principles:
{chr(10).join(f'- {p}' for p in kb.get_principles()[:10])}

Respond naturally and helpfully."""

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(clean_history[-10:])
            
            # Add user message with images if present
            if images and len(images) > 0:
                user_content = []
                if message:
                    user_content.append({"type": "text", "text": message})
                for img in images:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": img['data']}
                    })
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": message})
            
            # Use base model from config
            response = get_chat_completion(messages, temperature=0.7)
            
            # Save chat history
            _save_chat_turn(actor_name, message, response)
            
            return jsonify({
                'response': response,
                'actor': actor_name,
                'cognitive': False
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _save_chat_turn(actor_name: str, user_msg: str, assistant_msg: str):
    """Save a chat turn to history file."""
    try:
        array_dir = get_array_dir()
        chat_dir = array_dir / 'data' / actor_name / 'chat_history'
        chat_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create today's chat file
        today = datetime.now().strftime('%Y%m%d')
        chat_file = chat_dir / f'chat_{today}.json'
        
        if chat_file.exists():
            with open(chat_file, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'actor_name': actor_name,
                'date': today,
                'turns': []
            }
        
        # Add turn
        data['turns'].append({
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'assistant': assistant_msg
        })
        
        # Save
        with open(chat_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")


@app.route('/api/actor/<actor_name>/chat/history')
def api_chat_history(actor_name):
    """Get chat history for an actor."""
    try:
        array_dir = get_array_dir()
        chat_dir = array_dir / 'data' / actor_name / 'chat_history'
        
        if not chat_dir.exists():
            return jsonify({'history': []})
        
        # Get all chat files
        chat_files = sorted(chat_dir.glob('chat_*.json'), reverse=True)
        
        # Return recent history (last 7 days)
        all_turns = []
        for chat_file in chat_files[:7]:
            with open(chat_file, 'r') as f:
                data = json.load(f)
                all_turns.extend(data.get('turns', []))
        
        return jsonify({
            'actor': actor_name,
            'total_turns': len(all_turns),
            'history': all_turns[-100:]  # Last 100 turns
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/learn', methods=['POST'])
def api_start_learning(actor_name):
    """Start a cognitive learning loop."""
    data = request.json
    topic = data.get('topic', '')
    goal = data.get('goal', '')
    mode = data.get('mode', 'standard')
    
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
            from src.actions.cognitive_loop import CognitiveLoop
            
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
                year_to=year_to
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
                # Nonstop mode: run until goal achieved (essentially unlimited)
                result = loop.learn_nonstop(
                    topic, goal, 
                    goal_threshold=goal_threshold,
                    max_iterations=9999  # Essentially unlimited, stops when goal achieved
                )
            elif mode == 'quick':
                result = loop.learn(topic, goal, max_iterations=1)
            elif mode == 'fast':
                result = loop.learn(topic, goal, max_iterations=min(max_iterations, 3))
            else:
                result = loop.learn(topic, goal, max_iterations)
            
            loop_outputs[loop_id].put(f"\n✅ COMPLETE: {json.dumps(result.get('statistics', {}))}")
            
        except Exception as e:
            loop_outputs[loop_id].put(f"\n❌ ERROR: {str(e)}")
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


@app.route('/api/loop/<loop_id>/status')
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


@app.route('/api/tasks/active')
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


@app.route('/api/tasks/stop/<loop_id>', methods=['POST'])
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


@app.route('/api/loop/<loop_id>/output')
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


def _save_chat_turn(actor_name: str, user_msg: str, assistant_msg: str):
    """Save a chat turn to history file."""
    try:
        array_dir = get_array_dir()
        chat_dir = array_dir / 'data' / actor_name / 'chat_history'
        chat_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create today's chat file
        today = datetime.now().strftime('%Y%m%d')
        chat_file = chat_dir / f'chat_{today}.json'
        
        if chat_file.exists():
            with open(chat_file, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'actor_name': actor_name,
                'date': today,
                'turns': []
            }
        
        # Add turn
        data['turns'].append({
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'assistant': assistant_msg
        })
        
        # Save
        with open(chat_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")


@app.route('/api/config')
def api_config():
    """Get current configuration."""
    config = load_config()
    array_dir = get_array_dir()
    return jsonify({
        'apiHost': config.get('apiHost', ''),
        'model': config.get('model', ''),
        'array_dir': str(array_dir),
        'array_exists': array_dir.exists()
    })


@app.route('/api/config', methods=['POST'])
def api_update_config():
    """Update configuration."""
    data = request.json
    config = load_config()
    
    # Update allowed fields
    if 'array_dir' in data:
        config['array_dir'] = data['array_dir']
    if 'apiHost' in data:
        config['apiHost'] = data['apiHost']
    if 'model' in data:
        config['model'] = data['model']
    
    # Save config
    config_path = project_root / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return jsonify({'success': True, 'config': config})


@app.route('/api/actor/create', methods=['POST'])
def api_create_actor():
    """Create a new actor from a description using AI generation."""
    data = request.json
    actor_name = data.get('name', '').strip().lower().replace(' ', '_')
    description = data.get('description', '').strip()
    preview_only = data.get('preview', False)
    
    if not actor_name:
        return jsonify({'error': 'Actor name is required'}), 400
    
    if not description:
        return jsonify({'error': 'Actor description is required'}), 400
    
    # Check if actor already exists (unless just previewing)
    array_dir = get_array_dir()
    actor_dir = array_dir / 'actors' / actor_name
    if not preview_only and actor_dir.exists():
        return jsonify({'error': f'Actor "{actor_name}" already exists'}), 400
    
    try:
        from src.utils.actor_generator import generate_actor_profile, save_actor_profile
        
        # Generate profile from description
        profile = generate_actor_profile(description)
        
        if "raw_response" in profile:
            return jsonify({'error': 'Failed to generate actor profile', 'details': profile['raw_response']}), 500
        
        # If only preview, return the profile without saving
        if preview_only:
            return jsonify({
                'success': True,
                'profile': profile,
                'message': 'Profile generated (preview only)'
            })
        
        # Save the actor
        save_actor_profile(profile, actor_name, project_root)
        
        return jsonify({
            'success': True,
            'actor': actor_name,
            'profile': profile,
            'message': f'Actor "{actor_name}" created successfully'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/create/manual', methods=['POST'])
def api_create_actor_manual():
    """Create a new actor manually with provided profile data."""
    data = request.json
    actor_name = data.get('name', '').strip().lower().replace(' ', '_')
    profile = data.get('profile', {})
    
    if not actor_name:
        return jsonify({'error': 'Actor name is required'}), 400
    
    # Check if actor already exists
    array_dir = get_array_dir()
    actor_dir = array_dir / 'actors' / actor_name
    if actor_dir.exists():
        return jsonify({'error': f'Actor "{actor_name}" already exists'}), 400
    
    try:
        from src.utils.actor_generator import save_actor_profile
        
        # Helper to convert string to list (split by newlines)
        def to_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str) and val.strip():
                return [line.strip() for line in val.strip().split('\n') if line.strip()]
            return []
        
        # Ensure profile has required keys with defaults
        default_profile = {
            'values': to_list(profile.get('values', [])),
            'principles': to_list(profile.get('principles', [])),
            'core_knowledge': to_list(profile.get('core_knowledge', [])),
            'supporting_knowledge': to_list(profile.get('supporting_knowledge', [])),
            'experiences': to_list(profile.get('experiences', [])),
            'adjustments': to_list(profile.get('adjustments', [])),
            'challenges': to_list(profile.get('challenges', []))
        }
        
        # Save the actor
        save_actor_profile(default_profile, actor_name, project_root)
        
        return jsonify({
            'success': True,
            'actor': actor_name,
            'message': f'Actor "{actor_name}" created successfully'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/sleep', methods=['POST'])
def api_actor_sleep(actor_name):
    """
    Put an actor to sleep - consolidate memories into LoRA training data.
    
    This processes all of the actor's accumulated knowledge, experiences,
    and learning sessions into training data that can be used to fine-tune
    a model to embody the actor's unique mind.
    """
    try:
        from src.actions.sleep_processor import SleepProcessor
        
        data = request.json or {}
        include_dreams = data.get('include_dreams', True)
        output_format = data.get('format', 'alpaca')  # alpaca, sharegpt, or full
        
        # Use verbose=True so we can see consolidation logs
        processor = SleepProcessor(actor_name, verbose=True)
        
        # Run complete sleep cycle (includes dreams by default)
        result = processor.sleep()
        
        return jsonify({
            'success': True,
            'actor': actor_name,
            'summary': result,
            'message': f'Sleep cycle complete! Generated {result["total_examples"]} training examples.'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/sleep/status', methods=['GET'])
def api_actor_sleep_status(actor_name):
    """Get the status of LoRA training data for an actor."""
    try:
        array_dir = get_array_dir()
        lora_dir = array_dir / 'data' / actor_name / 'lora_training'
        
        if not lora_dir.exists():
            return jsonify({
                'has_data': False,
                'message': 'No sleep data yet. Put the actor to sleep to generate LoRA training data.'
            })
        
        # Find dataset files
        dataset_files = list(lora_dir.glob('*.jsonl'))
        summary_files = list(lora_dir.glob('*.summary.json'))
        
        result = {
            'has_data': len(dataset_files) > 0,
            'datasets': []
        }
        
        for dataset_file in dataset_files:
            # Count lines (examples)
            with open(dataset_file, 'r') as f:
                example_count = sum(1 for _ in f)
            
            # Load summary if exists
            summary_file = dataset_file.with_suffix('.summary.json')
            summary = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            
            result['datasets'].append({
                'filename': dataset_file.name,
                'path': str(dataset_file),
                'examples': example_count,
                'generated_at': summary.get('generated_at', 'Unknown'),
                'by_category': summary.get('by_category', {}),
                'recommended_params': summary.get('recommended_training_params', {})
            })
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/actor/<actor_name>/datasets/merge', methods=['POST'])
def api_merge_datasets(actor_name):
    """
    Merge multiple LoRA datasets from different learning sessions.
    
    Request body:
    {
        "dataset_files": ["file1.jsonl", "file2.jsonl"],  // Filenames to merge
        "output_name": "merged_dataset.jsonl"  // Optional, auto-generates if not provided
    }
    """
    try:
        from src.actions.sleep_processor import merge_datasets
        
        data = request.json or {}
        dataset_files = data.get('dataset_files', [])
        output_name = data.get('output_name', None)
        
        if not dataset_files or len(dataset_files) < 2:
            return jsonify({
                'error': 'Must provide at least 2 datasets to merge'
            }), 400
        
        # Build full paths
        array_dir = get_array_dir()
        lora_dir = array_dir / 'data' / actor_name / 'lora_training'
        dataset_paths = [str(lora_dir / filename) for filename in dataset_files]
        
        # Merge datasets
        result = merge_datasets(actor_name, dataset_paths, output_name)
        
        return jsonify({
            'success': True,
            'actor': actor_name,
            'result': result,
            'message': f'Merged {result["sources_merged"]} datasets into {result["total_examples"]} examples'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== Training Server Integration ====================

def get_trainer_config():
    """Get trainer server configuration."""
    config = load_config()
    return config.get('trainer_server', {
        'host': 'localhost',
        'port': 5050,
        'enabled': False
    })


def get_trainer_url():
    """Get the trainer server URL."""
    trainer = get_trainer_config()
    return f"http://{trainer['host']}:{trainer['port']}"


def translate_path_for_server(local_path: str) -> str:
    """
    Translate a local Mac path to the server's view inside the Docker container.
    
    Mac mount:     /Users/ruihe/mnt/silvertongueArray/array/data/...
    DGX host:      /mnt/silvertongueArray/array/data/...
    Docker mount:  /datasets/...  (DATASET_DIR is mounted to /datasets)
    
    The launch script mounts: -v "$DATASET_DIR":/datasets
    Where DATASET_DIR=/mnt/silvertongueArray/array/data
    """
    # Get the local array_dir from config
    config = load_config()
    local_array_dir = config.get('array_dir', '')
    
    # The Docker container mounts DATASET_DIR (/mnt/silvertongueArray/array/data) to /datasets
    # So /mnt/silvertongueArray/array/data/steve/... becomes /datasets/steve/...
    
    # First, convert Mac path to what the DGX host sees
    # Mac: /Users/ruihe/mnt/silvertongueArray/array/...
    # DGX: /mnt/silvertongueArray/array/...
    mac_mount = '/Users/ruihe/mnt/silvertongueArray/array'
    dgx_mount = '/mnt/silvertongueArray/array'
    
    server_path = local_path
    if local_array_dir and local_path.startswith(local_array_dir):
        server_path = local_path.replace(local_array_dir, dgx_mount)
    elif local_path.startswith(mac_mount):
        server_path = local_path.replace(mac_mount, dgx_mount)
    
    # Now convert DGX host path to Docker container path
    # The DATASET_DIR (/mnt/silvertongueArray/array/data) is mounted as /datasets
    dgx_data_dir = '/mnt/silvertongueArray/array/data'
    docker_data_dir = '/datasets'
    
    if server_path.startswith(dgx_data_dir):
        server_path = server_path.replace(dgx_data_dir, docker_data_dir)
    
    return server_path


@app.route('/api/trainer/status', methods=['GET'])
def api_trainer_status():
    """Check if the remote trainer server is available."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({
            'enabled': False,
            'message': 'Remote trainer is not enabled in config.json'
        })
    
    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/health", timeout=5)
        if response.ok:
            data = response.json()
            return jsonify({
                'enabled': True,
                'connected': True,
                'base_model': data.get('base_model', 'Unknown'),
                'server': data,
                'url': get_trainer_url()
            })
    except Exception as e:
        return jsonify({
            'enabled': True,
            'connected': False,
            'error': str(e),
            'url': get_trainer_url()
        })


@app.route('/api/trainer/config', methods=['GET'])
def api_trainer_config():
    """Get trainer server configuration."""
    return jsonify(get_trainer_config())


@app.route('/api/trainer/config', methods=['PUT'])
def api_update_trainer_config():
    """Update trainer server configuration."""
    data = request.json
    config = load_config()
    
    trainer = config.get('trainer_server', {})
    if 'host' in data:
        trainer['host'] = data['host']
    if 'port' in data:
        trainer['port'] = data['port']
    if 'enabled' in data:
        trainer['enabled'] = data['enabled']
    
    config['trainer_server'] = trainer
    
    config_path = project_root / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return jsonify({'success': True, 'trainer_server': trainer})


@app.route('/api/actor/<actor_name>/train', methods=['POST'])
def api_start_training(actor_name):
    """Submit a LoRA training job to the remote trainer server."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({
            'error': 'Remote trainer is not enabled',
            'hint': 'Enable trainer_server in config.json or set up DGX Spark'
        }), 400
    
    # Check if training data exists - look for various dataset patterns
    array_dir = get_array_dir()
    lora_dir = array_dir / 'data' / actor_name / 'lora_training'
    
    # Find the best dataset file (prefer merged, then latest timestamped, then basic)
    train_file = None
    
    # Check request for specific dataset
    data = request.json or {}
    requested_dataset = data.get('dataset_file')
    
    if requested_dataset:
        train_file = lora_dir / requested_dataset
    else:
        # Auto-find best dataset
        # 1. Look for merged datasets (most comprehensive)
        merged_files = sorted(lora_dir.glob(f'{actor_name}_merged_*.jsonl'), reverse=True)
        if merged_files:
            train_file = merged_files[0]
        else:
            # 2. Look for timestamped datasets
            timestamped = sorted(lora_dir.glob(f'{actor_name}_lora_dataset_*.jsonl'), reverse=True)
            if timestamped:
                train_file = timestamped[0]
            else:
                # 3. Look for basic dataset
                basic = lora_dir / f"{actor_name}_lora_dataset.jsonl"
                if basic.exists():
                    train_file = basic
                else:
                    # 4. Fallback to old naming
                    old_style = lora_dir / f"{actor_name}_lora_train.jsonl"
                    if old_style.exists():
                        train_file = old_style
    
    if not train_file or not train_file.exists():
        return jsonify({
            'error': 'No training data found',
            'hint': 'Run the sleep cycle first to generate training data',
            'searched_dir': str(lora_dir)
        }), 404
    
    # Translate local path to server path
    server_dataset_path = translate_path_for_server(str(train_file))
    
    # Build training config - use server's expected field names
    training_config = {
        'actor_name': actor_name,
        'dataset_path': server_dataset_path,  # Use translated path for server
        'config': {
            'r': data.get('lora_r', 32),  # Server uses 'r' not 'lora_r'
            'lora_alpha': data.get('lora_alpha', 64),
            'lora_dropout': data.get('lora_dropout', 0.05),
            'learning_rate': data.get('learning_rate', 2e-4),
            'num_epochs': data.get('num_epochs', 3),
            'batch_size': data.get('batch_size', 4),
            'gradient_accumulation_steps': data.get('gradient_accumulation_steps', 4),
            'max_seq_length': data.get('max_seq_length', 2048),
        }
    }
    
    try:
        import requests
        response = requests.post(
            f"{get_trainer_url()}/api/train",
            json=training_config,
            timeout=30
        )
        
        if response.ok:
            result = response.json()
            return jsonify({
                'success': True,
                'job_id': result.get('job_id'),
                'status': 'started',
                'message': f'Training job submitted for {actor_name}',
                'dataset': str(train_file.name),
                'config': training_config
            })
        else:
            return jsonify({
                'error': 'Failed to submit training job',
                'details': response.text
            }), response.status_code
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to trainer server',
            'hint': f'Make sure the trainer server is running at {get_trainer_url()}'
        }), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trainer/jobs', methods=['GET'])
def api_trainer_jobs():
    """Get list of training jobs from the remote server."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({'jobs': [], 'enabled': False})
    
    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/api/jobs", timeout=10)
        if response.ok:
            return jsonify(response.json())
        return jsonify({'jobs': [], 'error': response.text})
    except Exception as e:
        return jsonify({'jobs': [], 'error': str(e)})


@app.route('/api/trainer/jobs/<job_id>', methods=['GET'])
def api_trainer_job_status(job_id):
    """Get status of a specific training job."""
    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/api/jobs/{job_id}", timeout=10)
        if response.ok:
            return jsonify(response.json())
        return jsonify({'error': 'Job not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trainer/models', methods=['GET'])
def api_trainer_models():
    """Get list of trained models/adapters from the remote server."""
    trainer = get_trainer_config()
    
    if not trainer.get('enabled', False):
        return jsonify({'models': [], 'enabled': False})
    
    try:
        import requests
        response = requests.get(f"{get_trainer_url()}/api/models", timeout=10)
        if response.ok:
            return jsonify(response.json())
        return jsonify({'models': [], 'error': response.text})
    except Exception as e:
        return jsonify({'models': [], 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5050, threaded=True)
