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
├── train/                  # Training infrastructure
│   ├── train.py            # Multi-task training loop
│   ├── model.py            # MultiTaskModel (shared encoder + 4 task heads)
│   ├── dataset.py          # SinciputDataset & collate_fn
│   └── overfit_test.py     # Sanity check (overfit on small batch)
├── data/                   # Generated data (gitignored)
│   ├── raw_texts/          # Step 1 output: {domain}.jsonl
│   ├── extracted/          # Step 2 output: {domain}.jsonl
│   ├── tokenized/          # Step 3 output: {domain}.jsonl
│   ├── labeled/            # Step 4 output: {domain}.jsonl
│   └── training/           # Step 5 output: final training examples
├── output/                 # Training outputs (gitignored)
│   ├── checkpoints/        # Per-epoch checkpoints (model + optimizer + scheduler)
│   ├── runs/               # TensorBoard logs
│   └── model_final.pt      # Final model weights only (saved on completion)
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

## Training

### Training Data

| Stat | Value |
|------|-------|
| Total records | 183,859 |
| Domains | 369 |
| Blocks per domain | ~500 |
| Task heads | 4 (domain, NER, parsing, coreference) |

### Model & Optimizations

- **Base encoder**: `nreimers/MiniLM-L6-H384-uncased` (~23.9M params, ~18.6M trainable)
- **Layer freezing**: Encoder layers 0-2 frozen, layers 3-5 trainable
- **Differential learning rates**: Encoder 1e-5, task heads 1e-4 to 3e-4
- **Per-task gradient clipping**: parsing=5.0, all others=1.0
- **Dynamic padding**: Pad to batch max length, not fixed 512
- **Vectorized span gather & pairwise scoring** in coreference head (6x speedup)

### Running Training

```bash
# Start a full 30-epoch run
python -m train.train --data-dir data/training --epochs 30 --output-dir output

# Run in sessions: train 5 epochs, then stop
python -m train.train --data-dir data/training --epochs 30 --stop-after 5 --output-dir output

# Resume and run 5 more epochs
python -m train.train --epochs 30 --stop-after 5 --resume output/checkpoints/checkpoint_epoch5.pt

# Resume and finish remaining epochs
python -m train.train --epochs 30 --resume output/checkpoints/checkpoint_epoch10.pt
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/training` | Training data directory |
| `--epochs` | `20` | Total number of training epochs |
| `--stop-after` | _(all)_ | Stop after N epochs this session (checkpoint & resume later) |
| `--resume` | _(none)_ | Path to checkpoint to resume from |
| `--output-dir` | `output` | Base output directory |
| `--checkpoint-dir` | `<output-dir>/checkpoints` | Checkpoint directory |
| `--log-dir` | `<output-dir>/runs` | TensorBoard log directory |
| `--base-lr` | `1e-5` | Base learning rate for encoder |
| `--device` | _(auto)_ | Device: `mps`, `cuda`, `cpu` (auto-detected) |
| `--fp16` | off | Mixed precision training (CUDA only, falls back to fp32 on MPS) |
| `--keep-last-n` | `3` | Keep only the last N checkpoints (plus epoch 1) |
| `--num-workers` | _(auto)_ | DataLoader workers (0 for MPS/CPU, 4 for CUDA) |
| `--log-interval` | `10` | Log metrics every N steps |
| `--domains` | _(all)_ | Comma-separated domain filter |

### Output Structure

```
output/
├── checkpoints/
│   ├── checkpoint_epoch1.pt    # ~232MB each (model + optimizer + scheduler)
│   ├── checkpoint_epoch2.pt
│   └── ...
├── runs/                       # TensorBoard logs (tensorboard --logdir output/runs)
└── model_final.pt              # ~96MB (model weights only, saved on completion)
```

Checkpoint retention (`--keep-last-n 3`) automatically cleans up old checkpoints, always preserving epoch 1 and the latest 3.

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

- [x] Phase 1: Data pipeline (build scripts complete)
- [x] Phase 1: Full data generation (369 domains, 183,859 blocks)
- [x] Phase 2: Training infrastructure (multi-task loop, checkpointing, resume)
- [ ] Phase 2: Full training run (30 epochs)
- [ ] Phase 3: ONNX export & Rust deployment
