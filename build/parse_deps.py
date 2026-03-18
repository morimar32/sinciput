#!/usr/bin/env python3
"""
Phase 1, Step 5: Add dependency parsing arcs via spaCy.

Runs each raw text through spaCy's dependency parser, then maps the
spaCy token-level arcs onto the WordPiece token indices.

Per PROJ.md Section 6.3: dependency trees cannot span multiple sentences,
so roots are reset per sentence.

Subword continuations (##), [CLS], and [SEP] are masked with -100.

Input:  data/labeled/{domain}.jsonl
Output: data/training/{domain}.jsonl  (final TrainingExample format)
"""

import argparse
import time
from pathlib import Path

import spacy
from tqdm import tqdm

from build.models import (
    LabeledBlock,
    TrainingExample,
    IGNORE_INDEX,
)


# ============================================================================
# Configuration
# ============================================================================

SPACY_MODEL = "en_core_web_sm"

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LABELED_DIR = DATA_DIR / "labeled"
TRAINING_DIR = DATA_DIR / "training"


# ============================================================================
# Dependency Arc Mapping
# ============================================================================

def build_parsing_arcs(block: LabeledBlock, nlp: spacy.language.Language) -> list[int]:
    """
    Build dependency arc array mapped to WordPiece token indices.

    For each WordPiece token that corresponds to the start of a spaCy word,
    the arc points to the WordPiece token index of that word's syntactic head.
    Root tokens point to 0.
    """

    num_tokens = len(block.tokens)
    arcs = [IGNORE_INDEX] * num_tokens  # Default: ignore everything

    doc = nlp(block.raw_text)

    # Build a mapping from character offset to spaCy token
    # spaCy token -> (head spaCy token, is_root)
    # Then map spaCy tokens to WordPiece tokens via char positions

    # For each spaCy token, find its corresponding WordPiece token index
    # (the first non-## WordPiece token covering that character span)
    spacy_to_wp: dict[int, int] = {}  # spaCy token index -> WordPiece token index

    for spacy_tok in doc:
        char_start = spacy_tok.idx
        char_end = char_start + len(spacy_tok.text)

        # Find the first WordPiece token for this spaCy token
        wp_idx = None
        for char_idx in range(char_start, min(char_end, len(block.raw_text))):
            if char_idx < len(block.raw_text):
                # Use char_to_token from tokenized block
                # We need to reconstruct this from the tokens
                tok_idx = _char_to_token_lookup(block, char_idx)
                if tok_idx is not None:
                    # Find the first non-subword token in this span
                    if not block.tokens[tok_idx].startswith("##"):
                        wp_idx = tok_idx
                        break
                    elif wp_idx is None:
                        # If we only find subwords, walk back to find the head token
                        for back_idx in range(tok_idx - 1, 0, -1):
                            if not block.tokens[back_idx].startswith("##"):
                                wp_idx = back_idx
                                break

        if wp_idx is not None:
            spacy_to_wp[spacy_tok.i] = wp_idx

    # Now build arcs: for each spaCy token that maps to a WordPiece token,
    # point to the WordPiece token of its syntactic head
    for spacy_tok in doc:
        if spacy_tok.i not in spacy_to_wp:
            continue

        wp_idx = spacy_to_wp[spacy_tok.i]

        if spacy_tok.dep_ == "ROOT":
            arcs[wp_idx] = 0  # Root points to 0
        else:
            head_wp = spacy_to_wp.get(spacy_tok.head.i)
            if head_wp is not None:
                arcs[wp_idx] = head_wp

    return arcs


def _char_to_token_lookup(block: LabeledBlock, char_idx: int) -> int | None:
    """
    Look up which WordPiece token a character belongs to.
    Reconstructs from token offsets since LabeledBlock doesn't carry char_to_token.
    """
    # We need to rebuild char_to_token from the tokens list
    # This is done by walking the tokens and tracking character positions
    if not hasattr(block, '_char_map_cache'):
        block._char_map_cache = _build_char_map(block)
    return block._char_map_cache.get(char_idx)


def _build_char_map(block: LabeledBlock) -> dict[int, int]:
    """Rebuild character-to-token mapping from tokens and raw_text."""
    char_map = {}
    text_lower = block.raw_text.lower()
    text_pos = 0

    for tok_idx, token in enumerate(block.tokens):
        # Skip special tokens
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        # Handle subword tokens
        tok_text = token[2:] if token.startswith("##") else token

        # Find this token's text in the raw text starting from current position
        found_pos = text_lower.find(tok_text, text_pos)
        if found_pos == -1:
            # Try from a broader search window
            found_pos = text_lower.find(tok_text, max(0, text_pos - 5))

        if found_pos != -1:
            for ci in range(found_pos, found_pos + len(tok_text)):
                char_map[ci] = tok_idx
            # Only advance text_pos if we found it at or after current position
            if found_pos >= text_pos:
                text_pos = found_pos + len(tok_text)

    return char_map


# ============================================================================
# Build Final Training Example
# ============================================================================

def build_training_example(block: LabeledBlock, nlp: spacy.language.Language) -> TrainingExample:
    """Convert a labeled block into the final training example."""

    parsing_arcs = build_parsing_arcs(block, nlp)

    return TrainingExample(
        id=block.id,
        raw_text=block.raw_text,
        domain=block.domain,
        block_type=block.block_type,
        tokens=block.tokens,
        domain_class=block.domain_class,
        ner_labels=block.ner_labels,
        parsing_arcs=parsing_arcs,
        coref_clusters=block.coref_clusters,
    )


# ============================================================================
# I/O
# ============================================================================

def load_labeled_blocks(domain: str) -> list[LabeledBlock]:
    path = LABELED_DIR / f"{domain}.jsonl"
    blocks = []
    with open(path) as f:
        for line in f:
            blocks.append(LabeledBlock.model_validate_json(line))
    return blocks


def save_training_examples(domain: str, examples: list[TrainingExample]) -> Path:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_DIR / f"{domain}.jsonl"
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
    return output_path


def load_completed_domains() -> set[str]:
    if not TRAINING_DIR.exists():
        return set()
    return {p.stem for p in TRAINING_DIR.glob("*.jsonl") if p.stat().st_size > 1000}


def get_available_domains() -> list[str]:
    if not LABELED_DIR.exists():
        return []
    return sorted(p.stem for p in LABELED_DIR.glob("*.jsonl") if p.stat().st_size > 1000)


# ============================================================================
# Main
# ============================================================================

def main(selected_domains: list[str] | None = None):
    available = get_available_domains()

    if selected_domains:
        domains = [d for d in selected_domains if d in available]
        invalid = [d for d in selected_domains if d not in available]
        if invalid:
            print(f"WARNING: No labeled data found for: {invalid}")
    else:
        domains = available

    print("=" * 60)
    print("Phase 1, Step 5: Dependency Parsing (spaCy)")
    print("=" * 60)
    print(f"spaCy model: {SPACY_MODEL}")
    print(f"Domains: {len(domains)}")
    print(f"Input directory: {LABELED_DIR}")
    print(f"Output directory: {TRAINING_DIR}")
    print("=" * 60)

    completed = load_completed_domains()
    if completed:
        completed_in_scope = completed & set(domains)
        if completed_in_scope:
            print(f"\nResuming: {len(completed_in_scope)} domains already parsed")

    remaining = [d for d in domains if d not in completed]

    if not remaining:
        print("\nAll domains complete!")
        return

    print(f"Domains to process: {len(remaining)}")

    # Load spaCy model once
    print(f"\nLoading spaCy model: {SPACY_MODEL}...")
    nlp = spacy.load(SPACY_MODEL)
    print("spaCy loaded.\n")

    overall_start = time.perf_counter()
    total_blocks = 0

    for domain in tqdm(remaining, desc="Domains"):
        blocks = load_labeled_blocks(domain)
        domain_start = time.perf_counter()

        examples = []
        for block in blocks:
            ex = build_training_example(block, nlp)
            examples.append(ex)

        domain_time = time.perf_counter() - domain_start
        total_blocks += len(examples)

        output_path = save_training_examples(domain, examples)
        tqdm.write(f"\n{domain}:")
        tqdm.write(f"  Blocks: {len(examples)} | Time: {domain_time:.1f}s")
        tqdm.write(f"  Saved -> {output_path.name}")

    overall_time = time.perf_counter() - overall_start

    print("\n" + "=" * 60)
    print("Dependency Parsing Complete - Summary")
    print("=" * 60)
    print(f"Domains processed: {len(remaining)}")
    print(f"Total training examples: {total_blocks}")
    print(f"Overall time: {overall_time:.1f}s ({overall_time / 60:.1f} min)")
    print(f"Rate: {total_blocks / overall_time:.0f} blocks/sec")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add dependency parsing arcs to produce final training examples")
    parser.add_argument("--domains", nargs="+", help="Specific domain(s) to process (default: all available)")
    args = parser.parse_args()
    main(selected_domains=args.domains)
