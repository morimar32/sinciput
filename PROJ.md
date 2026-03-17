# Project Overview: Edge-Optimized Multi-Task NLP Pipeline (MiniLM-L6)

## 1. Executive Summary
The objective of this project is to build a highly optimized, low-latency Natural Language Processing (NLP) model capable of executing four distinct tasks (Domain Classification, Named Entity Recognition, Dependency Parsing, and Coreference Resolution) in a single forward pass. The target deployment environment is a resource-constrained edge device utilizing an ONNX runtime executed via Rust.

To achieve this without catastrophic latency or memory bloat, we will fine-tune a heavily distilled Small Language Model (SLM) using a Multi-Task Learning (MTL) architecture. Training and synthetic data generation will be executed locally on Apple Silicon (Mac mini M4 Pro, 64GB Unified Memory).

---

## 2. Core Architecture

### Foundation Model
* **Base Model:** `microsoft/MiniLM-L6-H384-uncased`
* **Size:** 6 layers ($L=6$), 384 hidden dimension ($H=384$), ~22.7 million parameters.
* **Deployment Footprint:** ~23 MB (when quantized to INT8 via ONNX).
* **Why:** Achieves a 5.3x inference speedup over standard BERT-Base while maintaining semantic capacity. It provides single-pass contextual embeddings that feed simultaneously into four isolated task heads.

### The Four Task Heads
1. **Domain Context (Sequence Classification):** A linear classifier attached to the `[CLS]` token to determine the overarching domain/context of the text block.
2. **Named Entity Recognition (Token Classification):** A linear classifier scoring individual tokens using the strict IOB2 schema (e.g., `B-PER`, `I-ORG`, `O`).
3. **Dependency Parsing (Biaffine Attention):** Evaluates $O(N^2)$ relational pairs. Uses multi-layer perceptrons (MLPs) to isolate head vs. dependent features, followed by a Biaffine Scorer to map directed syntactic arcs between words.
4. **Coreference Resolution (Span-Pair Scoring):** Evaluates $O(N^4)$ pairwise combinations of text spans. Uses attention-weighted span representations and calculates the Marginalized Log-Likelihood to link pronouns to their correct antecedents across the document.

---

## 3. Data Engineering Pipeline (The "Divide & Conquer" Method)

LLMs cannot reliably execute strict token-index math. Therefore, data generation uses a hybrid pipeline to ensure 100% mathematical integrity for the downstream loss functions.

### The Generation Pipeline
1. **Semantic Generation:** Local `Qwen3.5-35B-A3B` (quantized) generates raw text and identifies domain classifications based on programmatic prompts.
2. **Entity & Coref Extraction:** `Qwen3.5-35B-A3B` reads the text and extracts string representations of entities and pronoun-noun links (e.g., `{"John": "he"}`).
3. **Tokenization (Deterministic):** A Python script runs the raw text through the Hugging Face `WordPiece` tokenizer to map exact subwords.
4. **Index Mapping (Deterministic):** Python calculates the exact token start/end integer indices for the LLM's extracted strings.
5. **Syntax Parsing (Deterministic):** Python passes the raw text through `spaCy` (or `Stanza`) to deterministically generate flawless dependency tree arrays. 

### Data Volume & Distribution
* **Total Volume:** ~150,000 text blocks (300 domains × 500 blocks per domain).
* **Block Composition Strategy:**
    * **20% (Single Sentences):** Syntactic foundation; perfect for parser training and simulated edge commands.
    * **40% (3-4 Sentences):** Contextual sweet spot; high entity density for NER disambiguation.
    * **20% (5-7 Sentences):** Long-distance stress test; trains the Coreference head's span memory across broader document context.
    * **10% (1-2 Sentences, Adversarial):** "Winograd" style sentences to force semantic (rather than heuristic) coreference resolution.
    * **10% (1-3 Sentences, Noisy):** Fragmented, informal logs or texts to build resilience against typos and missing punctuation.

---

## 4. JSON Training Data Schema

This payload format ensures perfectly aligned data for the interleaved batch training process.

```json
{
  "id": "domain-142-block-089",
  "raw_text": "The baker closed his shop.",
  
  // 1. TOKENIZATION (Includes [CLS], [SEP], and WordPiece subwords)
  "tokens": ["[CLS]", "the", "bak", "##er", "closed", "his", "shop", ".", "[SEP]"],
  
  // 2. DOMAIN CONTEXT (Targeting the [CLS] token)
  "domain_class": 42, 
  
  // 3. NER LABELS (IOB2 format)
  // CRITICAL CAVEAT: Intermediate subwords ("##er") and padding MUST be masked 
  // with -100 to prevent boundary gradient destabilization during cross-entropy loss.
  "ner_labels": [
    -100, // [CLS]
       0, // "the" (O)
       1, // "bak" (B-PERSON)
    -100, // "##er" (IGNORED SUBWORD)
       0, // "closed" (O)
       0, // "his" (O)
       0, // "shop" (O)
       0, // "." (O)
    -100  // [SEP]
  ],

  // 4. DEPENDENCY PARSING (Biaffine Arcs)
  // Points to the index of the syntactic head. 0 = Root.
  "parsing_arcs": [
    -100, // [CLS]
       2, // "the"    -> points to "bak"
       4, // "bak"    -> points to "closed"
    -100, // "##er"   -> ignored
       0, // "closed" -> Root
       6, // "his"    -> points to "shop"
       4, // "shop"   -> points to "closed"
       4, // "."      -> points to "closed"
    -100  // [SEP]
  ],

  // 5. COREFERENCE CLUSTERS (Span Boundaries)
  "coref_clusters": [
    [
      {"span_bounds": [1, 3], "text": "the baker"}, 
      {"span_bounds": [5, 5], "text": "his"}        
    ]
  ]
}
```

---

## 5. Training Mechanics & Hardware Profile

Training will be conducted entirely on the Mac mini M4 Pro utilizing PyTorch's `mps` (Metal Performance Shaders) backend to leverage the 64GB of Unified Memory. This completely eliminates the $O(N^4)$ Out-of-Memory (OOM) errors associated with standard GPUs during Coreference evaluation.

### Strict Fine-Tuning Constraints
To prevent the compressed representation manifolds of MiniLM from rapidly degrading (catastrophic forgetting), the training loop MUST implement:
* **Interleaved Batching:** Batches should be task-stratified (e.g., Batch 1 = NER, Batch 2 = Coref) with gradients accumulating into the shared base model.
* **Layer Freezing:** Layers 0 through 2 of the MiniLM encoder must remain frozen to preserve foundational linguistic feature extractors.
* **Differential Learning Rates:** The base transformer layers must use a severely constrained learning rate ($1 \times 10^{-5}$ to $2 \times 10^{-5}$), while the newly initialized task heads require a 5x-10x higher rate ($1 \times 10^{-4}$ to $3 \times 10^{-4}$).
* **Warmup Schedule:** A linear warmup over the initial 5-10% of total steps is required to prevent gradient shock across uninitialized sequence heads.

---

## 6. Caveats & Gotchas

1. **The Anisotropic Trap:** Do NOT use `sentence-transformers/all-MiniLM-L6-v2` as the base model. Its contrastive pre-conditioning heavily biases the embeddings for sentence-level similarity, breaking token-level boundary detection for NER. Stick to `microsoft/MiniLM-L6-H384-uncased`.
2. **The 512-Token Ceiling:** MiniLM has a hard maximum context window of 512 tokens. Python generation scripts must explicitly drop or truncate synthetic paragraphs that exceed this limit before adding them to the JSON dataset.
3. **Cross-Sentence Parsing Errors:** Dependency trees cannot span multiple sentences. The Python parsing script must ensure that `parsing_arcs` reset roots correctly for multi-sentence blocks; otherwise, the Biaffine Scorer will learn impossible grammar.
4. **Rust Tokenization in Production:** ONNX only executes tensor math. The final Rust deployment binary MUST implement the Hugging Face `tokenizers` crate to handle raw string-to-WordPiece conversions before feeding the integer arrays into the `.onnx` model session.
