# TOON Reference for Anamnesis Integration

## Quick Reference

### Installation
```bash
pip install git+https://github.com/toon-format/toon-python.git
```

### Core API
```python
from toon_format import encode, decode

# Encode Python objects to TOON
toon_str = encode({"name": "Alice", "age": 30})

# Decode TOON back to Python
data = decode("items[2]: apple,banana")
```

### Encode Options
- `delimiter`: "," (default), "\t", or "|"
- `indent`: Spaces per indentation level (default: 2)

### Decode Options
- `indent`: Expected indentation size (default: 2)
- `strict`: Validate syntax and constraints (default: True)

---

## Format Syntax

### Objects
```
key: value
nested:
  child: value
```

### Primitive Arrays
```
tags[3]: admin,ops,dev
```

### Tabular Arrays (uniform objects)
```
items[2]{id,name,price}:
  1,Widget,9.99
  2,Gadget,14.50
```

### Mixed Arrays
```
items[3]:
  - 1
  - a: 1
  - text
```

---

## Quoting Rules

Strings require quotes when they:
- Are empty
- Have leading/trailing whitespace
- Equal `true`/`false`/`null`
- Resemble numbers
- Contain `:`, `"`, `\`, brackets, braces
- Match the active delimiter

---

## Escape Sequences

Only five escapes in quoted strings:
- `\\` (backslash)
- `\"` (quote)
- `\n` (newline)
- `\r` (carriage return)
- `\t` (tab)

---

## LLM Usage Best Practices

1. **Show, don't describe** - Present TOON in fenced code blocks
2. **Use tab delimiters** for better token efficiency: `delimiter='\t'`
3. **Always validate** model-generated TOON with `strict=True`
4. **Keep examples minimal** - 2-5 rows demonstrate the pattern

---

## Token Savings

| Response Type | JSON (tokens) | TOON (tokens) | Savings |
|---------------|---------------|---------------|---------|
| 100 concepts | ~4,500 | ~2,800 | 38% |
| 50 patterns | ~1,800 | ~1,150 | 36% |
| Metrics breakdown | ~450 | ~340 | 24% |

---

## Anamnesis Integration Tiers

### Tier 1: Adopt TOON (best fit)
- `manage_concepts` (query) - arrays of uniform concept data
- `get_coding_guidance` - pattern metadata and recommendations

### Tier 2: Conditional
- `search_codebase` - file match lists
- `get_sessions` - session lists when >10 entries

### Tier 3: Keep JSON
- Full pattern objects with code examples
- Relationship graphs
- Developer profiles (deep nesting)
