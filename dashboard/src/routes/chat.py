# filepath: /Users/ruihe/mnt/silvertongueArray/silvertongue/dashboard/src/routes/chat.py
"""
Chat routes for the Silvertongue Dashboard.

Handles: /api/actor/<name>/chat, /api/actor/<name>/chat/history
"""

import json
from datetime import datetime
from flask import Blueprint, jsonify, request

from ..utils import get_array_dir, import_project_module
from ..actors import get_actor_details

chat_bp = Blueprint('chat', __name__)


def _save_chat_turn(actor_name: str, user_msg: str, assistant_msg: str):
    """Save a chat turn to history file."""
    try:
        array_dir = get_array_dir()
        chat_dir = array_dir / 'data' / actor_name / 'chat_history'
        chat_dir.mkdir(parents=True, exist_ok=True)

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

        data['turns'].append({
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'assistant': assistant_msg
        })

        with open(chat_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")


@chat_bp.route('/api/actor/<actor_name>/chat', methods=['POST'])
def api_chat(actor_name):
    """Chat with an actor - lightweight cognitive mode for chat only."""
    data = request.json
    message = data.get('message', '')
    images = data.get('images', [])
    history = data.get('history', [])
    use_cognitive = data.get('use_cognitive', True)
    adapter = data.get('adapter', '')

    if not message and not images:
        return jsonify({'error': 'No message or images provided'}), 400

    clean_history = [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in history
        if msg.get("role") and msg.get("content")
    ]

    try:
        # Import from project root's src (not dashboard's src)
        chat_utils = import_project_module('src.utils.chat_utils')
        knowledge_base_mod = import_project_module('src.utils.knowledge_base')
        subconscious_mod = import_project_module('src.thoughts.subconscious')
        conscious_mod = import_project_module('src.thoughts.conscious')
        
        get_chat_completion = chat_utils.get_chat_completion
        KnowledgeBase = knowledge_base_mod.KnowledgeBase
        SilvertongueSubconscious = subconscious_mod.SilvertongueSubconscious
        SilvertongueConscious = conscious_mod.SilvertongueConscious

        kb = KnowledgeBase(actor_name)
        subconscious = SilvertongueSubconscious(kb)
        conscious = SilvertongueConscious(kb)

        details = get_actor_details(actor_name)

        if use_cognitive:
            analysis = subconscious.analyze_request(message)
            conscious_prompt = conscious.construct_conscious_prompt(analysis)

            system_prompt = f"""{conscious_prompt}

You are responding to a chat message. Draw upon:
- Your principles and values
- The relevant experiences and knowledge identified by your subconscious
- Your accumulated learning

User's message: {message}

Provide a thoughtful, helpful response that reflects your knowledge and character."""

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(clean_history[-10:])

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

            model_override = adapter if adapter else None
            response = get_chat_completion(
                messages, temperature=0.7, model_override=model_override)

            _save_chat_turn(actor_name, message, response)

            thought_history = [
                {"step": "Request Analysis",
                    "content": f"Analyzing request in domain: {analysis.get('domain', 'General')}"},
                {"step": "Intent Detection",
                    "content": f"Detected intent: {analysis.get('intent', 'Unknown')} | Complexity: {analysis.get('complexity', 'medium')}"},
            ]

            knowledge_count = len(analysis.get('relevant_knowledge', []))
            if knowledge_count > 0:
                thought_history.append({
                    "step": "Knowledge Retrieval",
                    "content": f"Retrieved {knowledge_count} relevant knowledge items from memory"
                })

            exp_count = len(analysis.get('relevant_experiences', []))
            if exp_count > 0:
                thought_history.append({
                    "step": "Experience Matching",
                    "content": f"Found {exp_count} relevant past experiences to apply"
                })

            principles_count = len(analysis.get('applicable_principles', []))
            if principles_count > 0:
                thought_history.append({
                    "step": "Principle Application",
                    "content": f"Applying {principles_count} guiding principles to response"
                })

            thought_history.append({
                "step": "Response Generation",
                "content": "Synthesizing response using retrieved context and character traits"
            })

            return jsonify({
                'response': response,
                'actor': actor_name,
                'cognitive': True,
                'system_prompt': system_prompt,
                'subconscious_analysis': analysis,
                'conscious_prompt': conscious_prompt,
                'analysis': analysis,
                'thought_history': thought_history
            })
        else:
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

            model_override = adapter if adapter else None
            response = get_chat_completion(
                messages, temperature=0.7, model_override=model_override)

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


@chat_bp.route('/api/actor/<actor_name>/chat/history')
def api_chat_history(actor_name):
    """Get chat history for an actor."""
    try:
        array_dir = get_array_dir()
        chat_dir = array_dir / 'data' / actor_name / 'chat_history'

        if not chat_dir.exists():
            return jsonify({'history': []})

        chat_files = sorted(chat_dir.glob('chat_*.json'), reverse=True)

        all_turns = []
        for chat_file in chat_files[:7]:
            with open(chat_file, 'r') as f:
                data = json.load(f)
                all_turns.extend(data.get('turns', []))

        return jsonify({
            'actor': actor_name,
            'total_turns': len(all_turns),
            'history': all_turns[-100:]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
