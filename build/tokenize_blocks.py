#!/usr/bin/env python3
"""
Phase 1, Step 3: Tokenize extracted blocks using the MiniLM WordPiece tokenizer.

Runs each raw text through the HuggingFace tokenizer for
nreimers/MiniLM-L6-H384-uncased, producing:
  - WordPiece token list (with [CLS] and [SEP])
  - Character-to-token index mapping for span alignment in Step 4

Blocks exceeding 512 tokens are truncated per PROJ.md Section 6.2.

Input:  data/extracted/{domain}.jsonl
Output: data/tokenized/{domain}.jsonl
"""

import argparse
import time
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from build.models import ExtractedBlock, TokenizedBlock


# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
MAX_TOKENS = 512  # MiniLM hard ceiling

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
TOKENIZED_DIR = DATA_DIR / "tokenized"


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_block(block: ExtractedBlock, tokenizer: AutoTokenizer) -> TokenizedBlock:
    """Tokenize a single block and build char-to-token mapping."""

    encoding = tokenizer(
        block.raw_text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_offsets_mapping=True,
    )

    token_ids = encoding["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    offsets = encoding["offset_mapping"]

    # Build char-to-token map: for each character in raw_text,
    # which token index does it belong to?
    text_len = len(block.raw_text)
    char_to_token = [None] * text_len

    for token_idx, (start, end) in enumerate(offsets):
        # Skip special tokens ([CLS], [SEP]) which have (0, 0) offsets
        if start == end:
            continue
        for char_idx in range(start, min(end, text_len)):
            char_to_token[char_idx] = token_idx

    return TokenizedBlock(
        **block.model_dump(),
        tokens=tokens,
        char_to_token=char_to_token,
    )


# ============================================================================
# I/O
# ============================================================================

def load_extracted_blocks(domain: str) -> list[ExtractedBlock]:
    """Load extracted blocks for a domain."""
    path = EXTRACTED_DIR / f"{domain}.jsonl"
    blocks = []
    with open(path) as f:
        for line in f:
            blocks.append(ExtractedBlock.model_validate_json(line))
    return blocks


def save_tokenized_blocks(domain: str, blocks: list[TokenizedBlock]) -> Path:
    """Save tokenized blocks to JSONL."""
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TOKENIZED_DIR / f"{domain}.jsonl"
    with open(output_path, "w") as f:
        for block in blocks:
            f.write(block.model_dump_json() + "\n")
    return output_path


def load_completed_domains() -> set[str]:
    """Check which domains have already been tokenized."""
    if not TOKENIZED_DIR.exists():
        return set()
    completed = set()
    for path in TOKENIZED_DIR.glob("*.jsonl"):
        if path.stat().st_size > 1000:
            completed.add(path.stem)
    return completed


def get_available_domains() -> list[str]:
    """Get domains that have extracted files."""
    if not EXTRACTED_DIR.exists():
        return []
    return sorted(p.stem for p in EXTRACTED_DIR.glob("*.jsonl") if p.stat().st_size > 1000)


# ============================================================================
# Main
# ============================================================================

def main(selected_domains: list[str] | None = None):
    available = get_available_domains()

    if selected_domains:
        domains = [d for d in selected_domains if d in available]
        invalid = [d for d in selected_domains if d not in available]
        if invalid:
            print(f"WARNING: No extracted data found for: {invalid}")
    else:
        domains = available

    print("=" * 60)
    print("Phase 1, Step 3: WordPiece Tokenization")
    print("=" * 60)
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Domains: {len(domains)}")
    print(f"Input directory: {EXTRACTED_DIR}")
    print(f"Output directory: {TOKENIZED_DIR}")
    print("=" * 60)

    # Resume support
    completed = load_completed_domains()
    if completed:
        completed_in_scope = completed & set(domains)
        if completed_in_scope:
            print(f"\nResuming: {len(completed_in_scope)} domains already tokenized")

    remaining = [d for d in domains if d not in completed]

    if not remaining:
        print("\nAll domains complete!")
        return

    print(f"Domains to process: {len(remaining)}")

    # Load tokenizer once
    print(f"\nLoading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded.\n")

    overall_start = time.perf_counter()
    total_blocks = 0
    total_truncated = 0

    for domain in tqdm(remaining, desc="Domains"):
        blocks = load_extracted_blocks(domain)
        domain_start = time.perf_counter()

        tokenized = []
        truncated = 0
        for block in blocks:
            tb = tokenize_block(block, tokenizer)
            if len(tb.tokens) == MAX_TOKENS:
                truncated += 1
            tokenized.append(tb)

        domain_time = time.perf_counter() - domain_start
        total_blocks += len(tokenized)
        total_truncated += truncated

        output_path = save_tokenized_blocks(domain, tokenized)
        tqdm.write(f"\n{domain}:")
        tqdm.write(f"  Blocks: {len(tokenized)} | Truncated: {truncated}")
        tqdm.write(f"  Time: {domain_time:.1f}s | Saved -> {output_path.name}")

    overall_time = time.perf_counter() - overall_start

    print("\n" + "=" * 60)
    print("Tokenization Complete - Summary")
    print("=" * 60)
    print(f"Domains processed: {len(remaining)}")
    print(f"Total blocks: {total_blocks}")
    print(f"Truncated to {MAX_TOKENS} tokens: {total_truncated}")
    print(f"Overall time: {overall_time:.1f}s ({overall_time / 60:.1f} min)")
    print(f"Rate: {total_blocks / overall_time:.0f} blocks/sec")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize extracted blocks with MiniLM WordPiece tokenizer")
    parser.add_argument("--domains", nargs="+", help="Specific domain(s) to process (default: all available)")
    args = parser.parse_args()
    main(selected_domains=args.domains)
