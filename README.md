# sinciput

`sinciput` /sĭn′sə-pət/ _(noun)_ - The upper half of the cranium, especially the anterior portion above and including the *forehead.*

multi-task, ~~fore~~ _four_ headed multi-task NLP model that performs **Domain Classification**, **Named Entity Recognition**, **Dependency Parsing**, and **Coreference Resolution** in a single forward pass.

Built on `nreimers/MiniLM-L6-H384-uncased` (~22.7M params, ~23MB quantized), targeting deployment on resource-constrained edge devices via ONNX runtime in Rust.

~~fore~~ _for_ anyone dissatisfied with the naming of this project, be grateful that I resisted the plebeian choice to refer to this multi-headed project with something as pedestrian as _hydra._ 
## Architecture

A shared MiniLM-L6 encoder feeds into four isolated task heads:

| Task | Head Type | Complexity | Description |
|------|-----------|------------|-------------|
| Domain Classification | Linear on `[CLS]` | O(1) | 300 domain classes |
| NER | Token classifier | O(N) | IOB2 schema: 8 entity types (PERSON, ORG, LOC, DATE, QUANTITY, EVENT, PRODUCT, OTHER) |
| Dependency Parsing | Biaffine attention | O(N^2) | Syntactic head arcs per token |
| Coreference Resolution | Span-pair scoring | O(N^4) | Pronoun-antecedent linking via marginalized log-likelihood |

## Project Structure

```
sinciput/
├── build/                  # Data generation & processing pipeline
│   ├── build.sh            # Run full pipeline (all 5 steps)
│   ├── models.py           # Pydantic models for all pipeline stages
│   ├── generate_texts.py   # Step 1: Generate raw text blocks via LLM
│   ├── extract_entities.py # Step 2: Extract entities & coreferences via LLM
│   ├── tokenize_blocks.py  # Step 3: WordPiece tokenization (local)
│   ├── build_labels.py     # Step 4: Map spans to NER labels & coref clusters (local)
│   └── parse_deps.py       # Step 5: Dependency parsing via spaCy (local)
├── data/                   # Generated data (gitignored)
│   ├── raw_texts/          # Step 1 output: {domain}.jsonl
│   ├── extracted/          # Step 2 output: {domain}.jsonl
│   ├── tokenized/          # Step 3 output: {domain}.jsonl
│   ├── labeled/            # Step 4 output: {domain}.jsonl
│   └── training/           # Step 5 output: final training examples
├── train/                  # Training scripts (TODO)
├── PROJ.md                 # Full project specification & training schema
├── RESEARCH.md             # Technical deep-dive on model architecture
└── requirements.txt        # Python dependencies
```

## Data Pipeline

The pipeline generates ~150,000 synthetic training examples (300 domains x 500 blocks) using a hybrid approach: LLM for semantic content, deterministic Python for mathematical precision.

### Why Hybrid?

LLMs cannot reliably produce exact token indices. The pipeline splits the problem: the LLM generates text and identifies entities/coreferences as **strings**, then deterministic scripts map those strings to exact **token-level integer labels** for loss computation.

### Pipeline Steps

| Step | Script | Method | Input | Output |
|------|--------|--------|-------|--------|
| 1 | `generate_texts.py` | Groq API (Qwen3-32B) | Domain list | Raw text blocks |
| 2 | `extract_entities.py` | Groq API (Qwen3-32B) | Raw text | Entities + coreference links |
| 3 | `tokenize_blocks.py` | Local (HuggingFace) | Extracted blocks | WordPiece tokens + char-to-token map |
| 4 | `build_labels.py` | Local (Python) | Tokenized blocks | IOB2 NER labels + coref span bounds |
| 5 | `parse_deps.py` | Local (spaCy) | Labeled blocks | Dependency arcs (final training format) |

### Block Type Distribution

Each domain generates 500 blocks with this composition:

- **20% Single sentences** - Syntactic foundation, parser training
- **40% Contextual (3-4 sentences)** - High entity density for NER
- **20% Long-distance (5-7 sentences)** - Coreference span memory stress test
- **10% Adversarial (1-2 sentences)** - Winograd-style ambiguous pronouns
- **10% Noisy (1-3 sentences)** - Informal/fragmented text resilience

## Setup

```bash
# Create and activate virtualenv
python -m venv s
source s/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Set Groq API key (get from https://console.groq.com/keys)
export GROQ_API_KEY="your-key-here"
```

## Usage

```bash
# Run full pipeline for all domains
./build/build.sh

# Run for specific domains only
./build/build.sh --domains cloud_computing cybersecurity

# Run individual steps
PYTHONPATH=. python build/generate_texts.py --domains cloud_computing
PYTHONPATH=. python build/extract_entities.py --domains cloud_computing
PYTHONPATH=. python build/tokenize_blocks.py --domains cloud_computing
PYTHONPATH=. python build/build_labels.py --domains cloud_computing
PYTHONPATH=. python build/parse_deps.py --domains cloud_computing
```

Every step has **resume support** - it skips domains that already have output files, so you can safely interrupt and re-run.

## Training Data Schema

The final output in `data/training/` matches this structure (see PROJ.md for full spec):

```json
{
  "id": "cloud_computing-0000",
  "raw_text": "Amazon Web Services launched a new region in Tokyo...",
  "domain": "cloud_computing",
  "block_type": "long_distance",
  "tokens": ["[CLS]", "amazon", "web", "services", "launched", "..."],
  "domain_class": 0,
  "ner_labels": [-100, 3, 4, 4, 0, "..."],
  "parsing_arcs": [-100, 3, 3, 4, 0, "..."],
  "coref_clusters": [
    {"mentions": [
      {"start": 1, "end": 3, "text": "Amazon Web Services"},
      {"start": 24, "end": 24, "text": "This"}
    ]}
  ]
}
```

Key conventions:
- `-100` masks `[CLS]`, `[SEP]`, and subword continuations (`##`) - PyTorch `CrossEntropyLoss` ignores these
- NER uses IOB2 format: `B-TYPE` for first token, `I-TYPE` for continuation
- Parsing arcs point to head token index; `0` = root
- Coref clusters group mentions with token-level `[start, end]` span bounds

## Key Technical Decisions

- **Base model**: `nreimers/MiniLM-L6-H384-uncased` (NOT `sentence-transformers/all-MiniLM-L6-v2` - its contrastive pre-conditioning breaks token-level NER)
- **LLM for data gen**: Qwen3-32B via Groq Cloud (`reasoning_effort="none"` to disable thinking mode)
- **Batched generation**: 25 blocks per API call with `===N===` separators, 4 concurrent requests
- **Entity extraction**: 10 blocks per batch, temperature 0.2 for precise structured output, entity spans validated against source text
- **512 token ceiling**: MiniLM's hard max context window - blocks are truncated at tokenization
- **Dependency trees**: Cannot span multiple sentences - spaCy handles per-sentence root resets

## Hardware

- **Training**: Mac mini M4 Pro, 64GB Unified Memory, PyTorch `mps` backend
- **Deployment**: Edge device, ONNX runtime via Rust with `tokenizers` crate for WordPiece

## Status

- [x] Phase 1: Data pipeline (build scripts complete, tested on 2 domains)
- [ ] Phase 1: Full data generation (300 domains, ~150k blocks)
- [ ] Phase 2: Model training
- [ ] Phase 3: ONNX export & Rust deployment
