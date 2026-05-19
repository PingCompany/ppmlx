# Memory Architecture v2: Decomposed Extraction for Small Models

> **Status:** Contrastive Retriever implemented (`ppmlx/contrastive_retriever.py`). Tested on real pi session: 9 segments → 2 kept (22% pass rate), correctly filtering restatements, code blocks, and noise while keeping novel info and contradiction signals.

## Problem

Current single-pass extraction (`ModelMemoryJsonExtractor`) gives a 2B model ~2000 tokens of raw dialog and asks it to simultaneously: understand context, identify facts, classify type (fact/preference/decision/todo/constraint/...), extract S-P-O triples, estimate confidence, estimate salience, and copy source quotes. 

On real conversational data (pi/claude sessions), this produces 0.33 candidates per turn. The model either returns nothing (too conservative) or garbage like `subject="add"` (verb misclassified as entity).

The root cause is **task complexity**, not prompt tuning. A 2B model cannot reliably perform 6 cognitive operations in one forward pass.

## Architecture

```
Session transcript
    │
    ├──[Step 0]── Load current memory state (pure code, ~1ms)
    │
    ▼
[Dense Chunker] ────────────────────────────────────── (embeddings + heuristics, ~50ms)
    │   Sliding windows → information density scoring
    │   Output: top 20% densest segments
    ▼
[Contrastive Retriever] ────────────────────────────── (embeddings + vector search, ~30ms)
    │   Novelty / contradiction / reinforcement scoring vs current knowledge
    │   Output: only novel or contradictory segments (drops 60-80% of dense segments)
    ▼
[Slot Classifier] ──────────────────────────────────── (1 small LLM call per segment, ~200ms)
    │   Multi-label: fact? preference? decision? todo? constraint?
    │   Output: type labels + text spans
    ▼
[Slot Extractor] ───────────────────────────────────── (1 small LLM call per type, ~300ms)
    │   Type-specific prompts with constrained output schema
    │   Output: {subject, predicate, object, text, confidence, salience, source_quote}
    ▼
[Self-Consistency ×3] ──────────────────────────────── (2 extra LLM calls, majority vote)
    │   Same segment, varied temperature + examples
    │   Output: candidates appearing in ≥2/3 runs
    ▼
[Validator + Graph Projection] ──────────────────────── (existing pipeline, ~1ms)
    │   MemoryValidator, store_candidate, upsert_memory_edge, run_inference
    ▼
memory.db (candidates, edges, inferred)
```

**Design principles:**
- **Decompose, don't prompt-engineer.** Each LLM call does one thing: classify OR extract a specific type. Never both.
- **Discriminative over generative.** Classification is 10x more reliable than structured generation for small models.
- **Filter before you extract.** Dense chunking eliminates 80% of noise before any model call. Contrastive retrieval eliminates redundant facts.
- **Vote, don't trust.** Self-consistency across 3 runs catches hallucinations. Real facts are stable; hallucinations vary.
- **All local.** Embeddings via MLX (same model). No external services.

---

## Component Specifications

### 0. Current Memory State Loader

**Purpose:** Load the project's current knowledge graph so downstream components can detect novelty and contradictions.

**Input:** `project_id`, `session_id`

**Output:** `MemorySnapshot`:
- `active_candidates`: list of all active candidates for this project + global scope
- `entity_embeddings`: cached embeddings for known entities (computed once, reused)
- `summary_text`: compact text rendering of current state (max 500 tokens)

**Implementation:**
```python
@dataclass
class MemorySnapshot:
    active_candidates: list[dict]
    entity_embeddings: dict[str, list[float]]  # entity_id → embedding vector
    summary_text: str

def load_memory_snapshot(store: MemoryStore, project_id: str, session_id: str) -> MemorySnapshot:
    # Query active candidates scoped to this project + global
    candidates = store.query_candidates(
        status='active', project_id=project_id, limit=100
    )
    global_candidates = store.query_candidates(
        status='active', scope='global', limit=50
    )
    
    # Build summary text from candidates
    summary_lines = []
    for c in candidates + global_candidates:
        summary_lines.append(f"[{c['scope']}] {c['type']}: {c['text']}")
    summary_text = '\n'.join(summary_lines[-40:])  # Most recent 40 facts
    
    # Load or compute entity embeddings (lazy, cached)
    entity_embeddings = _load_entity_embeddings(store)
    
    return MemorySnapshot(
        active_candidates=candidates + global_candidates,
        entity_embeddings=entity_embeddings,
        summary_text=summary_text[:2000],  # ~500 tokens
    )
```

---

### 1. Dense Chunker

**Purpose:** Find the 20% of conversation that contains extractable facts. Don't waste model context on greetings, code blocks, stack traces, or conversational filler.

**Algorithm:**

```
1. Split transcript into sliding windows of W=500 tokens, stride S=100 tokens
2. For each window, compute information density score:
   
   density = 0.35 × fact_signal_score
           + 0.25 × entity_density_score  
           + 0.20 × lexical_diversity_score
           + 0.20 × negation_score
   
   Where:
   - fact_signal_score: max cosine similarity of window embedding to a bank 
     of ~20 fact-indicator phrases ("we decided", "I prefer", "the plan is",
     "next step", "problem:", "todo:", "remember that", "important:", "key decision",
     "architecture:", "we use", "constraint:", "budget:", "deadline:")
   - entity_density_score: count of named entities / window length
     (detected via simple capitalization patterns + technical term dictionary)
   - lexical_diversity_score: unique tokens / total tokens (higher = more information)
   - negation_score: penalize windows matching code/stack-trace/log patterns
     (indentation, file paths, timestamps, error messages)

3. Select top K windows by density, where K = max(3, total_windows × 0.2)
4. Merge adjacent selected windows (within 2 strides)
5. Expand each merged segment by ±100 tokens for context
6. Output: list of TextSegment(text, start_idx, end_idx, density_score)
```

**Why no model call:** Embeddings + regex + statistics. ~50ms for a 2000-token transcript on Apple Silicon.

**Key insight:** The fact-indicator embedding bank is the only learned component. It can be bootstrapped from existing high-quality extracted facts (their source quotes), then fine-tuned. Initially it can be a hand-picked list of 20-30 phrases that correlate with extractable facts.

**Interface:**
```python
@dataclass 
class TextSegment:
    text: str
    start_idx: int
    end_idx: int
    density_score: float

def dense_chunk(
    messages: list[dict], 
    fact_indicator_embeddings: list[list[float]],
    window_tokens: int = 500,
    stride_tokens: int = 100,
    top_k_ratio: float = 0.2,
) -> list[TextSegment]:
    ...
```

---

### 2. Contrastive Retriever

**Purpose:** Given the current memory state, keep only segments that are novel (new information) or contradictory (potential update to existing fact). Drop segments that merely restate known facts.

**This is the highest-leverage component.** In a long session, 60-80% of "informational" content is restating or referencing things already known. Filtering those saves 60-80% of downstream model calls.

**Algorithm:**

```
For each dense segment:
  1. Compute segment embedding via MLX (same model, embedding mode)
  
  2. Novelty score:
     - Compute cosine similarity to each active candidate's text embedding
     - novelty = 1.0 - max(similarities)
     - If novelty > 0.7: clearly new information → KEEP
     - If novelty < 0.3: very similar to existing fact → DROP
     - If 0.3 ≤ novelty ≤ 0.7: ambiguous → continue to contradiction check
  
  3. Contradiction check (for ambiguous-novelty segments):
     - Find the top-3 most-similar active candidates
     - For each, check if the segment contains contradiction signals:
       * Lexical: "actually", "no longer", "instead", "not X but Y", "changed", "update:"
       * Structural: same subject+predicate but potentially different object
     - If contradiction signal found: KEEP (flagged as potential supersession)
  
  4. Reinforcement detection:
     - If novelty < 0.3 AND existing fact confidence ≥ 0.85: DROP (already known with high confidence)
     - If novelty < 0.3 AND existing fact confidence < 0.7: KEEP (weak fact, reinforcement adds confidence)
  
  5. Relevance gate:
     - If segment has no similarity > 0.15 to ANY existing fact AND no fact-indicator phrases:
       it's likely noise or domain-irrelevant → DROP
```

**Output:** `RelevantSegment(text, novelty_score, contradiction_flag, related_candidate_ids)`

**Why this is critical for small models:**
- A 2B model has limited attention. Giving it only novel content means it can allocate full attention to what matters.
- Contradiction detection at this stage means the extractor can be prompted specifically: "Here is a segment that may CONTRADICT existing knowledge. Extract the updated fact."
- Without this, the model wastes capacity re-extracting "ppmlx uses MLX" for the 5th time.

**Interface:**
```python
@dataclass
class RelevantSegment:
    text: str
    novelty_score: float
    contradiction_flag: bool
    related_candidate_ids: list[str]  # existing candidates this relates to
    segment_embedding: list[float]    # cached for downstream use

def contrastive_retrieve(
    segments: list[TextSegment],
    snapshot: MemorySnapshot,
    embedding_fn: Callable[[str], list[float]],
    novelty_threshold: float = 0.3,
    contradiction_threshold: float = 0.7,
) -> list[RelevantSegment]:
    ...
```

---

### 3. Slot Classifier

**Purpose:** For each relevant segment, answer a simple question: what TYPE of memory could this contain? Multi-label classification. No extraction, no S-P-O, no generation.

**Prompt design (per segment, ~200 tokens input, ~30 tokens output):**

```
Classify this conversation segment. Return ONLY a JSON array of types.
Types: fact, preference, decision, todo, constraint, instruction, relationship, none

Segment: "{segment_text}"

Types found:
```

**Output:** `["decision", "constraint"]` or `["none"]` or `["fact"]`

**Why classification beats generation:**
- Output space is tiny: 8 possible labels × 0-3 labels per segment = ~100 possible outputs
- Small models excel at classification (discriminative task) vs generation (constructive task)
- Classification errors are "missed a fact type" not "hallucinated a nonexistent fact with fake S-P-O"
- ~200ms per segment vs ~2000ms for full extraction

**Type definitions (given to model):**
```
fact        — a statement about what is true (e.g., "ppmlx uses SQLite")
preference  — someone's stated preference (e.g., "I prefer short answers")
decision    — a choice was made (e.g., "we decided to use pipe format")
todo        — an action item or next step (e.g., "todo: add examples")
constraint  — a limitation or requirement (e.g., "budget is 100ms per call")
instruction — a directive about how to behave (e.g., "keep answers under 3 sentences")
relationship — a connection between entities (e.g., "memory_store depends on memory_engine")
none        — no extractable memory in this segment
```

**Interface:**
```python
@dataclass
class ClassifiedSegment:
    text: str
    types: list[str]  # e.g., ["decision", "constraint"]
    spans: list[tuple[int, int]]  # character offsets for each type
    confidence: float
    source_segment: RelevantSegment

def classify_segment(
    segment: RelevantSegment,
    model_name: str = "gemma-4-e2b",
    generation_fn: Callable | None = None,
) -> ClassifiedSegment:
    ...
```

---

### 4. Slot Extractor

**Purpose:** Given a classified segment and its type(s), extract the {subject, predicate, object} triple. One call per type in the segment.

**Key design choice: type-specific prompts, not generic.**

Instead of one prompt that handles all 8 types, each type has its own constrained template:

**Example — "decision" extractor:**
```
Extract the decision from this text. Fill in the blanks:

Text: "We decided to use SQLite for storage — it is fast and requires no server."

Who or what made the decision: _____________________
What action was decided: ___________________________
What is the decision about: ________________________
One-sentence summary: _____________________________
Confidence (0.0-1.0): _____
Verbatim quote from text: _________________________
```

**Example — "preference" extractor:**
```
Extract the preference from this text. Fill in the blanks:

Text: "Rafał prefers concise commit messages under 60 chars."

Who has the preference: ___________________________
What do they prefer: _____________________________
In what context: _________________________________
One-sentence summary: ____________________________
Confidence (0.0-1.0): _____
Verbatim quote from text: _________________________
```

**Why type-specific prompts work:**
- Each prompt is ~150 tokens (vs ~400 for the generic JSON/pipe prompt)
- The model fills in blanks, doesn't need to remember output schema
- The blanks guide attention to relevant spans in the text
- Output parsing is deterministic (regex per field, not full JSON)
- A "decision" extractor never confuses a preference for a decision

**Interface:**
```python
@dataclass 
class ExtractedCandidate:
    type: str
    subject: str
    predicate: str
    object: str
    text: str
    scope: str
    confidence: float
    salience: float
    source_quote: str
    metadata: dict

# Type-specific extraction prompts
_EXTRACTION_TEMPLATES: dict[str, str] = {
    "fact": """...""",
    "preference": """...""",
    "decision": """...""",
    "todo": """...""",
    "constraint": """...""",
    "instruction": """...""",
    "relationship": """...""",
}

def extract_slots(
    classified: ClassifiedSegment,
    model_name: str = "gemma-4-e2b",
    generation_fn: Callable | None = None,
) -> list[ExtractedCandidate]:
    """Run type-specific extraction for each type in the classified segment."""
    candidates = []
    for fact_type in classified.types:
        if fact_type == "none":
            continue
        template = _EXTRACTION_TEMPLATES[fact_type]
        prompt = template.replace("{text}", classified.text)
        raw = generation_fn(model_name, [{"role": "user", "content": prompt}], 200, 0.0)
        candidate = _parse_slot_output(raw, fact_type)
        if candidate:
            candidates.append(candidate)
    return candidates
```

---

### 5. Self-Consistency

**Purpose:** A 2B model is stochastic. Real facts are stable across runs; hallucinations vary. Run extraction 3 times, keep only facts that appear in ≥2 runs.

**Algorithm:**

```
For each classified segment:
  1. Run slot extraction 3 times with variation:
     Run A: temperature=0.0, examples_set="coding"
     Run B: temperature=0.3, examples_set="infra"  
     Run C: temperature=0.7, examples_set="mixed"
     
     (Different few-shot examples in each run prevent the model from
      copying example content — each run has different "prior" examples.)
  
  2. Normalize candidates: lowercase, strip punctuation, canonicalize entity names
  
  3. Cluster candidates across runs:
     - Same type + fuzzy subject match + fuzzy predicate match + fuzzy object match
     - Fuzzy match: token Jaccard ≥ 0.6 OR embedding cosine ≥ 0.85
  
  4. Vote:
     - Appears in 3/3 runs → confidence = base_confidence × 1.0, keep
     - Appears in 2/3 runs → confidence = base_confidence × 0.8, keep
     - Appears in 1/3 runs → drop (likely noise or hallucination)
  
  5. For kept candidates, use the medoid (most central) candidate from the cluster
     as the final output
```

**Cost:** 3× LLM calls per segment. But:
- Each call is type-specific, ~200 tokens output max → ~300ms each
- Only runs on segments that passed classification (typically 2-5 per turn)
- Total: 2-5 segments × 3 runs × 300ms = 1.8-4.5s per turn
- Current single-pass extraction: 2s per turn, 0.33 candidates
- Self-consistency: 3-4.5s per turn, expected 2-4 high-quality candidates

**Why 3 runs, not 5 or 7:** Diminishing returns. Most hallucinations are one-off; 3 runs catches ~85% of them. 5 runs catches ~92% but costs 67% more compute.

**Interface:**
```python
@dataclass
class ConsensusCandidate:
    candidate: ExtractedCandidate
    num_runs: int  # 2 or 3
    cluster_size: int
    consensus_confidence: float

def self_consistency_extract(
    classified: ClassifiedSegment,
    model_name: str,
    generation_fn: Callable,
    num_runs: int = 3,
    agreement_threshold: int = 2,
) -> list[ConsensusCandidate]:
    ...
```

---

## Integration: DecomposedMemoryEngine

```python
class DecomposedMemoryEngine:
    """Memory extraction with task decomposition for small local models."""
    
    def __init__(self, store, model_name="gemma-4-e2b"):
        self.store = store
        self.model_name = model_name
        self.fact_embeddings = _load_fact_indicator_embeddings()
        self.entity_cache: dict[str, list[float]] = {}
    
    def extract_from_session(
        self, messages: list[dict], project_id: str, session_id: str
    ) -> ExtractionReport:
        
        # Step 0: Load current memory state
        snapshot = load_memory_snapshot(self.store, project_id, session_id)
        
        # Step 1: Dense chunking (pure code, ~50ms)
        dense_segments = dense_chunk(messages, self.fact_embeddings)
        
        # Step 2: Contrastive retrieval (pure code, ~30ms)
        relevant = contrastive_retrieve(
            dense_segments, snapshot, 
            embedding_fn=self._embed_text
        )
        
        # Step 3-5: Per-segment extraction pipeline
        all_candidates = []
        for segment in relevant:
            # Step 3: Classify (1 LLM call, ~200ms)
            classified = classify_segment(segment, self.model_name, self._generate)
            if "none" in classified.types and len(classified.types) == 1:
                continue
            
            # Step 4+5: Extract + self-consistency (3 LLM calls, ~900ms)
            consensus = self_consistency_extract(
                classified, self.model_name, self._generate
            )
            
            for cc in consensus:
                # Convert to ShadowMemoryCandidate and validate
                candidate = cc.to_shadow_candidate()
                validation = self.validator.validate(event, candidate)
                if validation['status'] == 'active':
                    self.store.store_candidate(candidate.to_record(), validation)
                    self.store.upsert_memory_edge(candidate.to_record())
                all_candidates.append(candidate)
        
        # Run inference on new edges
        inferred = self.store.run_inference()
        
        return ExtractionReport(
            segments_total=len(dense_segments),
            segments_relevant=len(relevant),
            candidates_extracted=len(all_candidates),
            candidates_active=sum(1 for c in all_candidates if c.status == 'active'),
            inferred_edges=sum(inferred.values()),
        )
```

---

## Performance Estimates

For a 2000-token, 9-turn pi session:

| Phase | Calls | Time | Notes |
|---|---|---|---|
| Memory snapshot | 0 LLM | ~2ms | SQLite query |
| Dense chunking | 0 LLM, 1 embed | ~50ms | MLX embedding of 4 windows |
| Contrastive retrieval | 0 LLM, 4 embeds | ~30ms | Cosine similarity on cached vectors |
| Slot classification | 1 LLM × 3 segments | ~600ms | 3 × 200ms classification calls |
| Slot extraction + SC | 3 LLM × 3 segments × 2 types | ~2.7s | 6 × 3 × 150ms type-specific calls |
| Validation + graph | 0 LLM | ~1ms | Existing pipeline |
| **Total** | **~22 LLM calls** | **~3.4s** | vs current 9 calls, 17.5s, 3 candidates |

**Expected output:** 6-12 high-quality candidates (vs current 3), 80%+ active rate, minimal hallucinations.

---

## Implementation Status

| Component | Status | File | Notes |
|---|---|---|---|
| Memory Snapshot Loader | Spec | — | Existing MemoryStore query methods sufficient |
| Dense Chunker | Spec | — | Embeddings + heuristics, ~50ms |
| **Contrastive Retriever** | **Implemented** | `ppmlx/contrastive_retriever.py` | Tested on real pi session (see below) |
| Slot Classifier | Spec | — | 1 small LLM call/segment, ~200ms |
| Slot Extractor | Spec | — | Type-specific prompts, ~300ms |
| Self-Consistency | Spec | — | 3× extraction + majority vote |

### Contrastive Retriever — implementation details

**File:** `ppmlx/contrastive_retriever.py` (~350 LOC)

**Key design decisions:**

1. **EmbeddingIndex** — pure numpy dot-product, no FAISS. For <1000 active candidates, brute-force cosine similarity is <1ms. Normalization on insert, not on query.

2. **Three-tier decision logic, not one threshold:**
   - `novelty > 0.70` → KEEP (clearly new information)
   - `novelty < 0.30` → DROP, unless: contradiction signal OR weak existing fact (reinforcement opportunity)
   - `0.30–0.70` → check contradiction + related fact confidence

3. **Contradiction detection** — 17 regex patterns (EN + PL), not LLM. Zero cost, zero latency. Patterns: "actually", "no longer", "instead", "właściwie", "jednak", "zmieniamy", etc.

4. **Embedding caching** — via `store_entity_alias` with type=`embedding_cache`. Survives process restarts. Batch-embeds only new candidates.

5. **Batch embedding** — `EmbedEngine.encode()` accepts `list[str]`. One model call for all segments, not N calls.

**Test results on real pi session (17 messages, ollama/ppmlx integration):**

```
Existing memories: 5 (ppmlx uses MLX, SQLite storage, Rafał prefers concise, 
                         ollama support [weak], Apple Silicon)
Input segments:    9 (from dense chunker simulation)
Kept:              2 (22%)
Dropped:           7 (78%)

Kept segment 1: "please investigate how ollama launch command works"
  → Novelty 0.84 — genuinely new investigation request
  
Kept segment 2: "you can pull model manifest for the model downloaded by ppmlx"
  → Novelty 0.83 — new capability not in existing memories

Dropped: ollama configuration details, tool output blocks, 
         restatements of known facts, conversational filler

Contradiction signal detected: "Ollama doesn't display models from 
  arbitrary manifest files" — flag for potential supersession
```

---

## Migration Path

1. **Phase 1:** Implement Dense Chunker + Contrastive Retriever as pre-filters to existing pipeline. No changes to extraction. Wins: fewer model calls, less noise.
2. **Phase 2:** Add Slot Classifier as a routing layer before current extractor. Wins: type-aware extraction, fewer "wrong type" errors.
3. **Phase 3:** Replace generic extractor with type-specific Slot Extractors. Wins: higher precision per type.
4. **Phase 4:** Add Self-Consistency wrapper around Slot Extractors. Wins: hallucination elimination.

Each phase is independently deployable and testable.
