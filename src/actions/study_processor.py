"""
Study Processor

Processes papers through LLM to extract knowledge bits, generate notes,
and create structured learning artifacts including enlightenment analysis.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import uuid

from src.utils.chat_utils import get_chat_completion
from src.utils.paper_parser import paper_parser
from src.utils.knowledge_base import KnowledgeBase

# Optional PDF fetching support
try:
    from src.utils.pdf_fetcher import PaperContentExtractor, PDFFetcher
    HAS_PDF_FETCHER = True
except ImportError:
    HAS_PDF_FETCHER = False


EXTRACTION_PROMPT = """You are a meticulous academic researcher extracting knowledge from a paper/article.

PAPER CONTENT (Chunk {chunk_num}/{total_chunks}):
---
{content}
---

EXTRACTION TARGETS:
{extraction_targets}

FOCUS AREAS: {focus_areas}

Extract knowledge bits from this content. For each piece of knowledge, provide:
1. A unique, descriptive title
2. The type (from extraction targets)
3. A clear, concise description (2-4 sentences)
4. Key terms/concepts involved
5. Confidence level (high/medium/low) based on how clearly stated it is
6. Direct quotes if relevant (short)

Return a JSON object with this structure:
{{
    "knowledge_bits": [
        {{
            "title": "string",
            "type": "concept|method|finding|application|reference",
            "description": "string",
            "key_terms": ["term1", "term2"],
            "confidence": "high|medium|low",
            "quote": "optional direct quote",
            "page_hint": "if known"
        }}
    ],
    "chunk_summary": "2-3 sentence summary of this chunk",
    "questions_raised": ["question1", "question2"]
}}

Be thorough but selective - extract only meaningful, actionable knowledge.
"""

SYNTHESIS_PROMPT = """You are synthesizing knowledge from a complete paper study session.

PAPER: {paper_name}
SOURCE: {file_path}

ALL EXTRACTED KNOWLEDGE BITS:
{all_bits_json}

CHUNK SUMMARIES:
{chunk_summaries}

QUESTIONS RAISED:
{questions}

Create a comprehensive study note that:
1. Synthesizes the key insights from all chunks
2. Identifies the main contributions and takeaways
3. Notes connections between concepts
4. Highlights practical applications
5. Lists follow-up questions for deeper study

Return a JSON object:
{{
    "paper_summary": "3-5 paragraph comprehensive summary",
    "main_contributions": ["contribution1", "contribution2"],
    "key_concepts": [
        {{
            "concept": "name",
            "definition": "clear definition",
            "importance": "why it matters"
        }}
    ],
    "methodology_notes": "if applicable, describe the methods used",
    "findings_summary": "key findings in bullet points",
    "applications": ["practical application 1", "application 2"],
    "connections": ["connection to other concepts or fields"],
    "follow_up_questions": ["question for deeper study"],
    "difficulty_level": "beginner|intermediate|advanced",
    "recommended_prerequisites": ["prereq1", "prereq2"]
}}
"""

ENLIGHTENMENT_PROMPT = """You are evaluating a paper's contribution to an actor's intellectual growth.

PAPER: {paper_name}
PAPER SUMMARY:
{paper_summary}

KEY FINDINGS FROM PAPER:
{key_findings}

ACTOR'S CURRENT KNOWLEDGE BASE:
---
Core Principles:
{principles}

Core Knowledge:
{core_knowledge}

Current Experiences:
{experiences}

Known Challenges:
{challenges}

Current Mindset:
{current_mindset}
---

Evaluate this paper's ENLIGHTENMENT value - how does it relate to the actor's current understanding?

Categories:
1. CONFIRMS - Paper validates/reinforces existing knowledge or principles
2. EXPANDS - Paper adds new perspectives, methods, or evidence to current understanding  
3. CHALLENGES - Paper contradicts or questions current beliefs (requires reflection)
4. REVOLUTIONARY - Paper introduces fundamentally new paradigms worth adopting
5. LOW_VALUE - Paper offers little new insight, is poorly reasoned, or irrelevant

Be critical and honest. Not every paper is valuable.

Return a JSON object:
{{
    "enlightenment_category": "CONFIRMS|EXPANDS|CHALLENGES|REVOLUTIONARY|LOW_VALUE",
    "confidence": "high|medium|low",
    "value_score": 1-10,
    "summary": "2-3 sentence summary of enlightenment assessment",
    "confirmations": [
        {{
            "existing_belief": "what the actor already knew/believed",
            "paper_evidence": "how the paper confirms this",
            "strengthened_confidence": true
        }}
    ],
    "expansions": [
        {{
            "current_understanding": "what the actor currently knows",
            "new_perspective": "what the paper adds",
            "integration_suggestion": "how to integrate this"
        }}
    ],
    "challenges": [
        {{
            "challenged_belief": "what is being questioned",
            "paper_argument": "the paper's counter-argument",
            "resolution_needed": "what reflection/research is needed",
            "severity": "minor|moderate|fundamental"
        }}
    ],
    "revolutionary_ideas": [
        {{
            "new_paradigm": "description of the new idea",
            "implications": "what this means for the actor",
            "adoption_recommendation": "adopt|investigate|monitor"
        }}
    ],
    "low_value_reasons": [
        "reason why paper lacks value (if applicable)"
    ],
    "recommended_actions": [
        {{
            "action": "what to do",
            "priority": "high|medium|low",
            "rationale": "why"
        }}
    ],
    "mindset_update": {{
        "should_update": true,
        "suggested_changes": ["change1", "change2"]
    }}
}}

Be intellectually rigorous. Challenge weak arguments. Celebrate genuine insights.
"""


class StudyProcessor:
    """Process papers and extract structured knowledge."""
    
    def __init__(self, actor_name: str = 'template'):
        self.actor_name = actor_name
        self.project_root = Path(__file__).parent.parent.parent
        self.actor_root = self.project_root / 'actors' / actor_name
        self.schedule_root = self.actor_root / 'learning_schedule'
        self.kb = KnowledgeBase(actor_name)
    
    def study_paper(self, file_path: Path, study_config: Dict) -> Dict:
        """
        Study a single paper and extract knowledge.
        
        Args:
            file_path: Path to the paper
            study_config: Study configuration from schedule
        
        Returns:
            Complete study result with notes and knowledge bits
        """
        print(f"\n📖 Studying: {file_path.name}")
        
        # Parse the paper
        parsed = paper_parser.parse_file(file_path)
        print(f"   Parsed {parsed['metadata']['word_count']} words, {len(parsed['chunks'])} chunks")
        
        # Extract knowledge from each chunk
        all_bits = []
        chunk_summaries = []
        all_questions = []
        
        extraction_targets = "\n".join([
            f"- {t['type']}: {t['description']}" 
            for t in study_config.get('extraction_targets', [])
        ])
        focus_areas = ", ".join(study_config.get('focus_areas', ['key concepts']))
        
        for i, chunk in enumerate(parsed['chunks']):
            print(f"   Processing chunk {i+1}/{len(parsed['chunks'])}...")
            
            prompt = EXTRACTION_PROMPT.format(
                chunk_num=i+1,
                total_chunks=len(parsed['chunks']),
                content=chunk[:8000],  # Limit chunk size
                extraction_targets=extraction_targets,
                focus_areas=focus_areas
            )
            
            response = get_chat_completion([
                {"role": "system", "content": "You are a knowledge extraction assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            
            try:
                # Parse response
                chunk_result = self._parse_json_response(response)
                
                # Add source info to each bit
                for bit in chunk_result.get('knowledge_bits', []):
                    bit['source_file'] = str(file_path)
                    bit['source_chunk'] = i + 1
                    bit['extracted_at'] = datetime.now().isoformat()
                    bit['bit_id'] = str(uuid.uuid4())[:8]
                    all_bits.append(bit)
                
                if chunk_result.get('chunk_summary'):
                    chunk_summaries.append(f"Chunk {i+1}: {chunk_result['chunk_summary']}")
                
                all_questions.extend(chunk_result.get('questions_raised', []))
                
            except Exception as e:
                print(f"   ⚠️ Error processing chunk {i+1}: {e}")
                continue
        
        print(f"   Extracted {len(all_bits)} knowledge bits")
        
        # Synthesize overall notes
        study_notes = None
        if study_config.get('generate_summary', True) and all_bits:
            print("   Synthesizing study notes...")
            study_notes = self._synthesize_notes(
                paper_name=file_path.name,
                file_path=str(file_path),
                all_bits=all_bits,
                chunk_summaries=chunk_summaries,
                questions=all_questions
            )
        
        # Perform enlightenment analysis
        enlightenment = None
        if study_config.get('analyze_enlightenment', True) and study_notes:
            print("   💡 Analyzing enlightenment...")
            enlightenment = self._analyze_enlightenment(
                paper_name=file_path.name,
                study_notes=study_notes,
                all_bits=all_bits
            )
            if enlightenment:
                category = enlightenment.get('enlightenment_category', 'UNKNOWN')
                score = enlightenment.get('value_score', '?')
                emoji = {
                    'CONFIRMS': '✅',
                    'EXPANDS': '🌱',
                    'CHALLENGES': '⚡',
                    'REVOLUTIONARY': '🚀',
                    'LOW_VALUE': '📉'
                }.get(category, '❓')
                print(f"   {emoji} Enlightenment: {category} (value: {score}/10)")
        
        # Build final result
        result = {
            "study_id": str(uuid.uuid4()),
            "studied_at": datetime.now().isoformat(),
            "paper": {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": parsed['file_type'],
                "metadata": parsed['metadata']
            },
            "extraction": {
                "total_chunks": len(parsed['chunks']),
                "knowledge_bits": all_bits,
                "chunk_summaries": chunk_summaries,
                "questions_raised": list(set(all_questions))
            },
            "study_notes": study_notes,
            "enlightenment": enlightenment,
            "config_used": {
                "depth": study_config.get('depth', 'standard'),
                "focus_areas": study_config.get('focus_areas', [])
            }
        }
        
        return result
    
    def study_from_url(self, paper_info: Dict, study_config: Dict) -> Dict:
        """
        Study a paper by fetching its PDF from a URL.
        
        Args:
            paper_info: Paper metadata dict with keys like 'url', 'title', 'abstract', 'doi'
            study_config: Study configuration from schedule
        
        Returns:
            Complete study result with notes and knowledge bits
        """
        if not HAS_PDF_FETCHER:
            # Fallback to abstract-only study
            print(f"\n📖 Studying (abstract only - PDF fetcher not available): {paper_info.get('title', 'Unknown')}")
            return self._study_abstract_only(paper_info, study_config)
        
        title = paper_info.get('title', 'Unknown Paper')
        print(f"\n📖 Studying: {title}")
        
        # Try to fetch full PDF content
        extractor = PaperContentExtractor()
        extracted = extractor.extract_from_paper(paper_info)
        
        if extracted['success'] and extracted.get('metadata', {}).get('content_source') == 'full_pdf':
            print(f"   ✅ Full PDF obtained ({extracted['metadata'].get('word_count', 0)} words)")
            return self._study_extracted_content(extracted, paper_info, study_config)
        else:
            # Fallback to abstract
            if extracted.get('error'):
                print(f"   ⚠️ {extracted['error']}")
            return self._study_abstract_only(paper_info, study_config)
    
    def _study_extracted_content(self, extracted: Dict, paper_info: Dict, study_config: Dict) -> Dict:
        """
        Study from extracted PDF content.
        """
        print(f"   Parsing {len(extracted.get('chunks', []))} chunks...")
        
        all_bits = []
        chunk_summaries = []
        all_questions = []
        
        extraction_targets = "\n".join([
            f"- {t['type']}: {t['description']}" 
            for t in study_config.get('extraction_targets', [])
        ])
        focus_areas = ", ".join(study_config.get('focus_areas', ['key concepts']))
        
        chunks = extracted.get('chunks', [])
        
        for i, chunk in enumerate(chunks):
            print(f"   Processing chunk {i+1}/{len(chunks)}...")
            
            prompt = EXTRACTION_PROMPT.format(
                chunk_num=i+1,
                total_chunks=len(chunks),
                content=chunk[:8000],
                extraction_targets=extraction_targets,
                focus_areas=focus_areas
            )
            
            response = get_chat_completion([
                {"role": "system", "content": "You are a knowledge extraction assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            
            try:
                chunk_result = self._parse_json_response(response)
                
                for bit in chunk_result.get('knowledge_bits', []):
                    bit['source_url'] = paper_info.get('url', '')
                    bit['source_title'] = paper_info.get('title', '')
                    bit['source_chunk'] = i + 1
                    bit['extracted_at'] = datetime.now().isoformat()
                    bit['bit_id'] = str(uuid.uuid4())[:8]
                    all_bits.append(bit)
                
                if chunk_result.get('chunk_summary'):
                    chunk_summaries.append(f"Chunk {i+1}: {chunk_result['chunk_summary']}")
                
                all_questions.extend(chunk_result.get('questions_raised', []))
                
            except Exception as e:
                print(f"   ⚠️ Error processing chunk {i+1}: {e}")
                continue
        
        print(f"   Extracted {len(all_bits)} knowledge bits from full PDF")
        
        # Synthesize notes
        study_notes = None
        if study_config.get('generate_summary', True) and all_bits:
            print("   Synthesizing study notes...")
            study_notes = self._synthesize_notes(
                paper_name=paper_info.get('title', 'Unknown'),
                file_path=paper_info.get('url', ''),
                all_bits=all_bits,
                chunk_summaries=chunk_summaries,
                questions=all_questions
            )
        
        # Enlightenment analysis
        enlightenment = None
        if study_config.get('analyze_enlightenment', True) and study_notes:
            print("   💡 Analyzing enlightenment...")
            enlightenment = self._analyze_enlightenment(
                paper_name=paper_info.get('title', 'Unknown'),
                study_notes=study_notes,
                all_bits=all_bits
            )
            if enlightenment:
                category = enlightenment.get('enlightenment_category', 'UNKNOWN')
                score = enlightenment.get('value_score', '?')
                emoji = {
                    'CONFIRMS': '✅', 'EXPANDS': '🌱', 'CHALLENGES': '⚡',
                    'REVOLUTIONARY': '🚀', 'LOW_VALUE': '📉'
                }.get(category, '❓')
                print(f"   {emoji} Enlightenment: {category} (value: {score}/10)")
        
        return {
            "study_id": str(uuid.uuid4()),
            "studied_at": datetime.now().isoformat(),
            "paper": {
                "title": paper_info.get('title', 'Unknown'),
                "url": paper_info.get('url', ''),
                "authors": paper_info.get('authors', []),
                "year": paper_info.get('year'),
                "abstract": paper_info.get('abstract', ''),
                "doi": paper_info.get('doi'),
                "content_source": "full_pdf",
                "pdf_path": extracted.get('metadata', {}).get('pdf_path'),
                "metadata": extracted.get('metadata', {})
            },
            "extraction": {
                "total_chunks": len(chunks),
                "knowledge_bits": all_bits,
                "chunk_summaries": chunk_summaries,
                "questions_raised": list(set(all_questions))
            },
            "study_notes": study_notes,
            "enlightenment": enlightenment,
            "config_used": {
                "depth": study_config.get('depth', 'standard'),
                "focus_areas": study_config.get('focus_areas', [])
            }
        }
    
    def _study_abstract_only(self, paper_info: Dict, study_config: Dict) -> Dict:
        """
        Study using only the paper's abstract when PDF is unavailable.
        """
        abstract = paper_info.get('abstract', '')
        title = paper_info.get('title', 'Unknown')
        
        if not abstract:
            print(f"   ⚠️ No abstract available for: {title}")
            return {
                "study_id": str(uuid.uuid4()),
                "studied_at": datetime.now().isoformat(),
                "paper": {
                    "title": title,
                    "url": paper_info.get('url', ''),
                    "content_source": "none",
                    "error": "No content available"
                },
                "extraction": {"knowledge_bits": [], "questions_raised": []},
                "study_notes": None,
                "enlightenment": None
            }
        
        print(f"   📝 Studying from abstract ({len(abstract)} chars)")
        
        extraction_targets = "\n".join([
            f"- {t['type']}: {t['description']}" 
            for t in study_config.get('extraction_targets', [])
        ])
        focus_areas = ", ".join(study_config.get('focus_areas', ['key concepts']))
        
        # Enhanced prompt for abstract-only study
        prompt = f"""You are extracting knowledge from a paper abstract.

PAPER TITLE: {title}

ABSTRACT:
{abstract}

AUTHORS: {', '.join(paper_info.get('authors', []))}
YEAR: {paper_info.get('year', 'Unknown')}
CITATIONS: {paper_info.get('citations', 'Unknown')}

EXTRACTION TARGETS:
{extraction_targets}

FOCUS AREAS: {focus_areas}

Since we only have the abstract, extract the most important knowledge bits available.
Be aware that abstracts contain condensed information - extract key claims, methods mentioned, and findings.

Return a JSON object:
{{
    "knowledge_bits": [
        {{
            "title": "string",
            "type": "concept|method|finding|application|claim",
            "description": "string",
            "key_terms": ["term1", "term2"],
            "confidence": "high|medium|low",
            "source_note": "from abstract"
        }}
    ],
    "abstract_summary": "1-2 sentence core summary",
    "main_claim": "the paper's main claim or contribution",
    "methodology_hint": "methodology mentioned (if any)",
    "questions_raised": ["question needing full paper to answer"],
    "relevance_assessment": "high|medium|low with reason"
}}
"""
        
        response = get_chat_completion([
            {"role": "system", "content": "You are a knowledge extraction assistant. Extract insights from paper abstracts. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.3)
        
        try:
            result = self._parse_json_response(response)
            
            all_bits = []
            for bit in result.get('knowledge_bits', []):
                bit['source_url'] = paper_info.get('url', '')
                bit['source_title'] = title
                bit['source_type'] = 'abstract'
                bit['extracted_at'] = datetime.now().isoformat()
                bit['bit_id'] = str(uuid.uuid4())[:8]
                all_bits.append(bit)
            
            print(f"   Extracted {len(all_bits)} knowledge bits from abstract")
            
            # Create study notes from abstract analysis
            study_notes = {
                "paper_summary": result.get('abstract_summary', ''),
                "main_contributions": [result.get('main_claim', '')] if result.get('main_claim') else [],
                "methodology_notes": result.get('methodology_hint', ''),
                "key_concepts": [{"concept": t, "from": "abstract"} for bit in all_bits for t in bit.get('key_terms', [])[:2]],
                "relevance": result.get('relevance_assessment', ''),
                "note": "Based on abstract only - full paper analysis requires PDF access"
            }
            
            # Simplified enlightenment for abstract
            enlightenment = None
            if study_config.get('analyze_enlightenment', True) and all_bits:
                enlightenment = self._analyze_enlightenment(
                    paper_name=title,
                    study_notes=study_notes,
                    all_bits=all_bits
                )
                if enlightenment:
                    category = enlightenment.get('enlightenment_category', 'UNKNOWN')
                    print(f"   💡 Enlightenment (from abstract): {category}")
            
            return {
                "study_id": str(uuid.uuid4()),
                "studied_at": datetime.now().isoformat(),
                "paper": {
                    "title": title,
                    "url": paper_info.get('url', ''),
                    "authors": paper_info.get('authors', []),
                    "year": paper_info.get('year'),
                    "abstract": abstract,
                    "doi": paper_info.get('doi'),
                    "content_source": "abstract_only"
                },
                "extraction": {
                    "total_chunks": 1,
                    "knowledge_bits": all_bits,
                    "chunk_summaries": [result.get('abstract_summary', '')],
                    "questions_raised": result.get('questions_raised', [])
                },
                "study_notes": study_notes,
                "enlightenment": enlightenment,
                "config_used": {
                    "depth": "abstract_only",
                    "focus_areas": study_config.get('focus_areas', [])
                }
            }
            
        except Exception as e:
            print(f"   ⚠️ Error studying abstract: {e}")
            return {
                "study_id": str(uuid.uuid4()),
                "studied_at": datetime.now().isoformat(),
                "paper": {"title": title, "url": paper_info.get('url', ''), "content_source": "error"},
                "extraction": {"knowledge_bits": [], "error": str(e)},
                "study_notes": None,
                "enlightenment": None
            }

    def _synthesize_notes(self, paper_name: str, file_path: str, 
                          all_bits: List[Dict], chunk_summaries: List[str],
                          questions: List[str]) -> Optional[Dict]:
        """Synthesize knowledge bits into comprehensive notes."""
        
        # Limit bits for prompt
        bits_for_prompt = all_bits[:30]  # Limit to avoid token limits
        
        prompt = SYNTHESIS_PROMPT.format(
            paper_name=paper_name,
            file_path=file_path,
            all_bits_json=json.dumps(bits_for_prompt, indent=2),
            chunk_summaries="\n".join(chunk_summaries),
            questions="\n".join(f"- {q}" for q in questions[:20])
        )
        
        response = get_chat_completion([
            {"role": "system", "content": "You are a knowledge synthesis assistant. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.4)
        
        try:
            return self._parse_json_response(response)
        except Exception as e:
            print(f"   ⚠️ Error synthesizing notes: {e}")
            return None
    
    def _analyze_enlightenment(self, paper_name: str, study_notes: Dict, 
                                all_bits: List[Dict]) -> Optional[Dict]:
        """
        Analyze paper's enlightenment value against actor's current knowledge.
        
        Returns assessment of whether paper:
        - CONFIRMS existing beliefs
        - EXPANDS current understanding  
        - CHALLENGES held beliefs
        - REVOLUTIONARY - introduces new paradigms
        - LOW_VALUE - offers little insight
        """
        # Gather actor's current knowledge
        principles = self.kb.get_principles()
        core_knowledge = self.kb.get_core_knowledge()
        experiences = self.kb.get_experiences()
        challenges = self.kb.get_challenges()
        current_mindset = self.kb.get_current_mindset()
        
        # Build paper summary for analysis
        paper_summary = study_notes.get('paper_summary', '')
        key_findings = []
        
        # Add main contributions
        for contrib in study_notes.get('main_contributions', []):
            key_findings.append(f"- Contribution: {contrib}")
        
        # Add key concepts
        for concept in study_notes.get('key_concepts', []):
            if isinstance(concept, dict):
                key_findings.append(f"- Concept: {concept.get('concept', '')} - {concept.get('importance', '')}")
            else:
                key_findings.append(f"- Concept: {concept}")
        
        # Add top findings from bits
        high_confidence_bits = [b for b in all_bits if b.get('confidence') == 'high'][:10]
        for bit in high_confidence_bits:
            key_findings.append(f"- {bit.get('type', 'Finding').title()}: {bit.get('title', '')} - {bit.get('description', '')[:100]}")
        
        # Format current knowledge for prompt
        def format_list(items, max_items=10):
            if not items:
                return "(none recorded)"
            items = items[:max_items]
            if isinstance(items[0], dict):
                return "\n".join([f"- {json.dumps(item)}" for item in items])
            return "\n".join([f"- {item}" for item in items])
        
        mindset_str = json.dumps(current_mindset, indent=2) if current_mindset else "(no current mindset recorded)"
        
        prompt = ENLIGHTENMENT_PROMPT.format(
            paper_name=paper_name,
            paper_summary=paper_summary,
            key_findings="\n".join(key_findings),
            principles=format_list(principles),
            core_knowledge=format_list(core_knowledge),
            experiences=format_list(experiences),
            challenges=format_list(challenges),
            current_mindset=mindset_str
        )
        
        response = get_chat_completion([
            {"role": "system", "content": "You are a critical intellectual assessor. Evaluate papers rigorously against existing knowledge. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ], temperature=0.5)
        
        try:
            return self._parse_json_response(response)
        except Exception as e:
            print(f"   ⚠️ Error analyzing enlightenment: {e}")
            return None
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response, handling common issues."""
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from response: {response[:200]}...")
    
    def save_study_result(self, result: Dict, schedule_name: str) -> Dict[str, Path]:
        """
        Save study results to appropriate locations.
        
        Returns paths to saved files.
        """
        notes_dir = self.schedule_root / 'notes' / schedule_name
        knowledge_dir = self.schedule_root / 'knowledge' / schedule_name
        
        notes_dir.mkdir(parents=True, exist_ok=True)
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames based on paper name
        base_name = Path(result['paper']['file_name']).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full study result as notes
        notes_path = notes_dir / f"{base_name}_{timestamp}_notes.json"
        with open(notes_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save knowledge bits separately with metadata
        knowledge_path = knowledge_dir / f"{base_name}_{timestamp}_knowledge.json"
        knowledge_data = {
            "source": result['paper'],
            "extracted_at": result['studied_at'],
            "study_id": result['study_id'],
            "knowledge_bits": result['extraction']['knowledge_bits'],
            "enlightenment": result.get('enlightenment'),
            "metadata": {
                "total_bits": len(result['extraction']['knowledge_bits']),
                "by_type": self._count_by_type(result['extraction']['knowledge_bits']),
                "by_confidence": self._count_by_confidence(result['extraction']['knowledge_bits']),
                "enlightenment_category": result.get('enlightenment', {}).get('enlightenment_category') if result.get('enlightenment') else None,
                "value_score": result.get('enlightenment', {}).get('value_score') if result.get('enlightenment') else None
            }
        }
        with open(knowledge_path, 'w') as f:
            json.dump(knowledge_data, f, indent=2)
        
        print(f"   💾 Saved notes to: {notes_path.name}")
        print(f"   💾 Saved knowledge to: {knowledge_path.name}")
        
        return {
            "notes": notes_path,
            "knowledge": knowledge_path
        }
    
    def _count_by_type(self, bits: List[Dict]) -> Dict[str, int]:
        """Count knowledge bits by type."""
        counts = {}
        for bit in bits:
            t = bit.get('type', 'unknown')
            counts[t] = counts.get(t, 0) + 1
        return counts
    
    def _count_by_confidence(self, bits: List[Dict]) -> Dict[str, int]:
        """Count knowledge bits by confidence level."""
        counts = {}
        for bit in bits:
            c = bit.get('confidence', 'unknown')
            counts[c] = counts.get(c, 0) + 1
        return counts


# Global instance
study_processor = StudyProcessor()
