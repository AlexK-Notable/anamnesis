# TOON Integration Design Document

## Anamnesis Semantic Code Intelligence Server

**Document Status:** Design Rationale
**Scope:** Conceptual analysis of TOON adoption for MCP tool responses

---

## Executive Summary

Anamnesis is a strong candidate for TOON (Token-Oriented Object Notation) adoption. The server's primary data exports—semantic concepts, pattern metadata, and metrics breakdowns—exhibit high tabular eligibility (70-80%), with consistent schemas across array elements and predominantly primitive values at the queryable surface.

Expected token savings: **25-40%** on targeted endpoints, with the highest gains on concept array responses and metrics breakdowns.

---

## 1. Why TOON for Anamnesis?

### The Token Pressure Problem

Anamnesis exists to give LLMs deep understanding of codebases. This creates an inherent tension:

- **Rich semantic data** requires verbose structures (concepts, relationships, patterns)
- **Context window limits** constrain how much intelligence can be surfaced per query
- **Token costs** scale with response verbosity

Every concept returned consumes tokens. A codebase with 500 functions, 80 classes, and 200 patterns generates substantial response payloads. JSON's structural overhead—repeated keys, quoted strings, braces—compounds across hundreds of array elements.

### Why TOON Specifically?

TOON's design philosophy aligns with anamnesis's data model:

1. **Tabular array optimization**: Anamnesis returns arrays of concepts with identical field structures. TOON declares fields once in a header, then represents each concept as a row. This eliminates per-element key repetition.

2. **Schema consistency**: Semantic concepts follow a strict schema (`id`, `concept_name`, `concept_type`, `confidence`, `file_path`, `line_start`, `line_end`). TOON's tabular format assumes and exploits this consistency.

3. **Primitive-heavy values**: At the query surface, most concept fields are strings, numbers, or enums—exactly what TOON handles efficiently. Nested structures like `relationships` can be deferred to separate queries or serialized as JSON strings within TOON cells.

4. **High volume, repetitive structure**: Returning 50-200 concepts per query is common. The more elements in an array, the greater TOON's relative advantage over JSON.

---

## 2. Data Structure Analysis

### Tier 1: Excellent TOON Fit

These structures should adopt TOON encoding as the primary format.

#### Semantic Concept Arrays (Flattened)

**Current JSON structure:**
```json
{
  "insights": [
    {"id": "abc123", "concept_name": "DatabaseConnection", "concept_type": "class", "confidence": 0.95, "file_path": "src/db.py", "line_start": 42, "line_end": 128},
    {"id": "def456", "concept_name": "execute_query", "concept_type": "function", "confidence": 0.88, "file_path": "src/db.py", "line_start": 130, "line_end": 145},
    ...
  ]
}
```

**Why it fits TOON:**
- Homogeneous array: every element has identical fields
- All values are primitives (strings, numbers)
- High element count (50-200 typical)
- Field names repeat for every element in JSON

**TOON representation:**
```
insights[N]{id,concept_name,concept_type,confidence,file_path,line_start,line_end}:
  abc123,DatabaseConnection,class,0.95,src/db.py,42,128
  def456,execute_query,function,0.88,src/db.py,130,145
  ...
```

**Token impact:** ~35-40% reduction. Field names declared once instead of N times.

---

#### Pattern Metadata (Without Examples)

**Current JSON structure:**
```json
{
  "recommendations": [
    {"pattern_id": "singleton_001", "pattern_type": "singleton", "frequency": 12, "confidence": 0.87},
    {"pattern_id": "factory_002", "pattern_type": "factory", "frequency": 8, "confidence": 0.82},
    ...
  ]
}
```

**Why it fits TOON:**
- Simple, flat objects
- All primitives
- Consistent schema across all patterns

**TOON representation:**
```
recommendations[N]{pattern_id,pattern_type,frequency,confidence}:
  singleton_001,singleton,12,0.87
  factory_002,factory,8,0.82
  ...
```

**Token impact:** ~35% reduction.

---

#### Intelligence Metrics Breakdown

**Current JSON structure:**
```json
{
  "breakdown": {
    "concepts_by_type": {"function": 145, "class": 43, "interface": 8, "variable": 203},
    "patterns_by_type": {"singleton": 12, "factory": 8, "dependency_injection": 15}
  }
}
```

**Why it fits TOON:**
- Key-value aggregations map naturally to two-column tables
- All values are integers
- Deterministic field ordering

**TOON representation:**
```
breakdown:
  concepts_by_type[4]{type,count}:
    function,145
    class,43
    interface,8
    variable,203
  patterns_by_type[3]{type,count}:
    singleton,12
    factory,8
    dependency_injection,15
```

**Token impact:** ~25% reduction (smaller absolute savings due to lower element counts).

---

### Tier 2: Conditional TOON Fit

These structures benefit from TOON only under specific conditions.

#### Feature Maps

**Structure:** `{feature_name: [file1, file2, file3]}`

**Analysis:**
- Outer structure: object with string keys → TOON handles via indentation
- Inner structure: arrays of strings → TOON primitive arrays

**Condition for TOON benefit:** Feature maps with >5 features and >3 files per feature.

**TOON representation:**
```
feature_map:
  authentication[3]: src/auth/handler.py,src/auth/middleware.py,src/models/user.py
  database[2]: src/database/connection.py,src/database/migrations.py
```

**Token impact:** ~20% reduction when conditions met.

---

#### Session Lists

**Structure:** Arrays of `{id, project_path, session_start, session_end}`

**Analysis:**
- Consistent schema ✓
- Primitive values ✓
- BUT: typically low volume (5-20 sessions)

**Condition for TOON benefit:** Only when returning >10 sessions.

**Token impact:** Marginal (10-15%) due to low typical volumes.

---

### Tier 3: Poor TOON Fit

These structures should remain JSON-encoded.

#### Full Pattern Objects with Examples

**Structure:**
```json
{
  "pattern_id": "singleton_001",
  "examples": [
    {"file": "src/cache.py", "lines": [10, 25], "code": "class Cache:\n    _instance = None\n    ..."}
  ]
}
```

**Why TOON struggles:**
- `examples[]` contains nested objects
- `code` field contains multi-line strings with special characters
- Nested arrays within objects break tabular eligibility

**Recommendation:** Keep as JSON. The complexity of escaping multi-line code strings in TOON cells outweighs potential savings.

---

#### Relationship Graphs

**Structure:** Concepts with `relationships: {uses: [...], extends: [...], implements: [...]}`

**Why TOON struggles:**
- Relationships are objects containing arrays
- Link targets may themselves be complex objects
- Graph structures are inherently non-tabular

**Recommendation:** Keep as JSON, or split into separate flattened queries:
- One query for concept metadata (TOON)
- One query for relationships (JSON or edge-list TOON)

---

#### Developer Profiles

**Structure:** Complex nested object with `patterns{}`, `naming_conventions{}`, `expertise[]`, `work_context{}`

**Why TOON struggles:**
- Deep nesting (3+ levels)
- Heterogeneous child structures
- Low query frequency (called once per session)

**Recommendation:** Keep as JSON. Single-use complex objects don't benefit from tabular optimization.

---

## 3. Tool-by-Tool Recommendations

| Tool | TOON Adoption | Rationale |
|------|---------------|-----------|
| `get_semantic_insights` | **Yes** | Core use case. Concept arrays are ideal TOON targets. |
| `get_pattern_recommendations` | **Partial** | TOON for metadata summary; JSON for full examples. |
| `get_intelligence_metrics` | **Yes** | Breakdown tables map directly to TOON format. |
| `predict_coding_approach` | **Partial** | TOON for file routing lists; JSON for approach narrative. |
| `get_project_blueprint` | **No** | Nested structure, single-use response. |
| `get_developer_profile` | **No** | Complex nesting, low frequency. |
| `search_codebase` | **Partial** | TOON for file match lists; JSON for context snippets. |
| `analyze_codebase` | **No** | Deep AST structures, nested metrics. |
| `contribute_insights` | **No** | Input operation, not response optimization. |
| `auto_learn_if_needed` | **No** | Status response, not data-heavy. |
| `list_sessions` | **Conditional** | TOON only if >10 sessions returned. |

---

## 4. Architectural Considerations

### Response Format Negotiation

Consider supporting both formats with client preference:

```
# Client indicates TOON preference via parameter
get_semantic_insights(query="database", format="toon")
get_semantic_insights(query="database", format="json")
```

Alternatively, detect LLM capability from client metadata and auto-select format.

### Hybrid Responses

Some tools benefit from mixed encoding:

```
# TOON for tabular data, JSON for complex nested data
response:
  concepts[50]{id,name,type,confidence,file}:
    abc123,DatabaseConnection,class,0.95,src/db.py
    ...
  relationship_graph: {"edges": [...]}  # Embedded JSON
```

This preserves TOON benefits for eligible data while avoiding awkward encoding of complex structures.

### LLM Parsing Reliability

TOON assumes LLMs can parse the format. Considerations:

1. **System prompt augmentation**: Include TOON format description in MCP server instructions
2. **Example-based learning**: Provide TOON parsing examples in initial context
3. **Fallback strategy**: If LLM struggles, degrade to JSON gracefully

---

## 5. Expected Outcomes

### Token Savings Projections

| Response Type | Current JSON (tokens) | TOON (tokens) | Savings |
|---------------|----------------------|---------------|---------|
| 100 concepts | ~4,500 | ~2,800 | 38% |
| 50 patterns (metadata) | ~1,800 | ~1,150 | 36% |
| Metrics breakdown | ~450 | ~340 | 24% |
| Feature map (10 features) | ~800 | ~600 | 25% |

### Qualitative Benefits

1. **Increased context density**: More concepts per query within same token budget
2. **Reduced API costs**: Direct cost savings on token-priced APIs
3. **Faster time-to-first-token**: Smaller payloads transmit faster
4. **Better LLM comprehension**: Tabular format may improve LLM pattern recognition on structured data

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| LLM parsing errors | Provide format examples in system prompt; implement fallback |
| Maintenance complexity | Abstract encoding behind response formatter; single point of change |
| Client compatibility | Support format negotiation; default to JSON for unknown clients |
| Edge cases in string escaping | Use TOON's delimiter options (tab, pipe) for data with embedded commas |

---

## 6. Decision Summary

**Adopt TOON for:**
- Semantic concept arrays (flattened)
- Pattern metadata listings
- Metrics breakdowns
- File path arrays in predictions

**Keep JSON for:**
- Full pattern objects with code examples
- Relationship graphs
- Developer profiles
- Complex nested structures
- Low-volume responses (<10 elements)

**Implementation priority:**
1. `get_semantic_insights` - highest volume, best fit
2. `get_intelligence_metrics` - clean tabular mapping
3. `get_pattern_recommendations` - hybrid approach (TOON metadata + JSON examples)

---

*This document captures design rationale. Implementation details, encoding functions, and API changes are out of scope.*
