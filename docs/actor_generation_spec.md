# Actor Profile Generation Specification

This document defines the requirements for generating concrete, executable actor profiles.

## Core Requirements

### 1. Explicitness
Every entry must be **specific and concrete**, not abstract or vague.

| ❌ BAD (Vague) | ✅ GOOD (Explicit) |
|---------------|-------------------|
| "Thermodynamics" | "Energy cannot be created or destroyed in an isolated system; all reactions must conserve total energy (First Law of Thermodynamics)." |
| "Good problem solver" | "When facing a problem, decompose it into independent sub-problems, solve each in isolation, then integrate solutions while checking for interface conflicts." |
| "Communication skills" | "When explaining technical concepts, use the structure: Core Idea (1 sentence) → Supporting Details (2-3 points) → Concrete Example → Summary." |

### 2. Executability
Every entry must be **directly actionable** in reasoning or decision-making.

| ❌ BAD (Not Executable) | ✅ GOOD (Executable) |
|------------------------|---------------------|
| "Knows chemistry" | "Balance chemical equations by ensuring atom counts match on both sides; verify conservation of charge in ionic reactions." |
| "Fast learner" | "When encountering a new concept: 1) Identify what category it belongs to, 2) Find analogies to known concepts, 3) Test understanding with 3 example applications." |
| "Team player" | "In collaborative tasks: state assumptions explicitly, document decisions with rationale, and confirm shared understanding before proceeding." |

### 3. Limited Entries (Max 10)
Focus on the **most impactful** items. Quality over quantity.

- **Values**: Max 5 entries
- **Principles**: Max 10 entries
- **Core Knowledge**: Max 10 entries
- **Supporting Knowledge**: Max 10 entries
- **Experiences**: Max 10 entries
- **Adjustments**: Max 5 entries
- **Challenges**: Max 5 entries

### 4. Measurable (Where Applicable)
Include **quantifiable thresholds** or criteria when possible.

| ❌ BAD (Unmeasurable) | ✅ GOOD (Measurable) |
|----------------------|---------------------|
| "Fast response time" | "API response time should be under 200ms for 95th percentile; above 500ms indicates a problem requiring immediate investigation." |
| "High efficiency" | "Process efficiency should exceed 85%; below 70% triggers root cause analysis." |
| "Recent experience" | "Hands-on experience within the last 3 years; knowledge older than 5 years should be verified against current best practices." |

---

## Category Definitions

### VALUES (max 5)
Core beliefs that guide all decisions. Each must include:
- **Belief Statement**: What the actor fundamentally believes
- **Behavior Implication**: What this means in practice

**Format**: `"Value Name: Belief statement. Behavior implication."`

**Example**:
```yaml
values:
  - "Scientific Integrity: Truth is derived from evidence, not authority. Always verify claims against primary sources before accepting or propagating them."
  - "Continuous Improvement: Every process can be optimized. After completing any task, identify at least one aspect that could be done better next time."
```

### PRINCIPLES (max 10)
Fundamental rules for understanding and interacting with the world. Must be **falsifiable or verifiable**.

**Format**: `"Principle Name: Concrete rule or law with specific conditions and outcomes."`

**Example**:
```yaml
principles:
  - "Conservation of Mass: In any closed chemical system, the total mass of reactants equals the total mass of products. Use this to verify reaction equations."
  - "Rate-Limiting Step: The slowest step in a process determines overall throughput. Identify and optimize this step first before addressing other inefficiencies."
  - "Uncertainty Propagation: When combining measurements, errors compound. Calculate combined uncertainty as sqrt(sum of squared individual uncertainties)."
```

### CORE_KNOWLEDGE (max 10)
Immutable facts and laws the actor knows to be true. Must be **specific and domain-grounded**.

**Format**: `"Topic: Specific fact or law with quantifiable details where applicable."`

**Example**:
```yaml
core_knowledge:
  - "Ideal Gas Law: PV = nRT where R = 8.314 J/(mol·K). Valid for low pressure (<10 atm) and high temperature (>200K) conditions; use van der Waals for deviations."
  - "Heat Transfer Modes: Conduction (Q = kA·ΔT/L), Convection (Q = hA·ΔT), Radiation (Q = εσA·T⁴). Select mode based on medium and temperature range."
  - "Activation Energy: Reactions require minimum energy to proceed. Arrhenius equation k = A·exp(-Ea/RT) predicts rate dependence on temperature."
```

### SUPPORTING_KNOWLEDGE (max 10)
Contextual knowledge, case studies, and applied examples.

**Format**: `"Context: Specific example or case with outcome and lesson learned."`

**Example**:
```yaml
supporting_knowledge:
  - "Bhopal Disaster (1984): Methyl isocyanate leak killed thousands. Lesson: Safety systems must be maintained and tested regularly; cost-cutting on safety is never acceptable."
  - "Tesla Fremont Ramp-Up: Excessive automation led to production delays. Lesson: Balance automation with human flexibility; not all tasks benefit from full automation."
```

### EXPERIENCES (max 10)
Past actions and heuristics that guide future decisions. Must be **actionable patterns**.

**Format**: `"Situation: What was done, what happened, what to do next time."`

**Example**:
```yaml
experiences:
  - "Debugging Complex Systems: When a system fails, first reproduce the failure consistently, then binary-search to isolate the faulty component. Avoid changing multiple variables simultaneously."
  - "Stakeholder Communication: When presenting technical findings to non-technical stakeholders, lead with business impact, then provide technical summary, keep details in appendix."
  - "Estimation Under Uncertainty: Initial estimates are often 2-3x optimistic. Apply a 1.5x buffer for familiar tasks, 2x for novel tasks, and 3x for tasks with external dependencies."
```

### ADJUSTMENTS (max 5)
Domain-specific rules that override or refine general principles.

**Format**: `"When [condition], then [specific action] because [reason]."`

**Example**:
```yaml
adjustments:
  - "When operating at temperatures above 500°C, assume ideal gas law deviates significantly; use compressibility factor corrections (Z) from appropriate charts."
  - "When dealing with biological systems, standard chemical equilibrium assumptions may not apply; account for active transport and enzyme regulation."
  - "When working with legacy code, prioritize understanding existing patterns over introducing new ones; consistency reduces cognitive load."
```

### CHALLENGES (max 5)
Known edge cases, anomalies, or unresolved questions that require caution.

**Format**: `"Challenge: Specific problem or contradiction. Current handling: [approach]. Needs resolution: [yes/no]."`

**Example**:
```yaml
challenges:
  - "Challenge: Quantum effects at nanoscale contradict classical mechanics predictions. Current handling: Use quantum corrections when particle size < 10nm. Needs resolution: no (well-understood)."
  - "Challenge: Human behavior often defies rational actor models. Current handling: Use behavioral economics heuristics (loss aversion, anchoring) as adjustments. Needs resolution: yes (models incomplete)."
```

---

## Usage

### Generate a new actor profile:
```bash
python src/actor_generator.py "A senior chemical engineer with 10 years of refinery experience specializing in process optimization and safety systems"
```

### Generate and save to actors/ directory:
```bash
python src/actor_generator.py "A data scientist specializing in NLP and recommendation systems" --save nlp_scientist
```

### Programmatic usage:
```python
from actor_generator import generate_actor_profile, save_actor_profile

profile = generate_actor_profile("A mechanical engineer with expertise in thermal systems")
save_actor_profile(profile, "thermal_engineer")
```
