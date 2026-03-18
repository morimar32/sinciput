#!/usr/bin/env python3
"""
Phase 1, Step 4: Map extracted entities and coreferences to token-level labels.

For each tokenized block, this script:
  1. Builds IOB2 NER label arrays from entity string spans
     - [CLS], [SEP], and subword continuations (##) are masked with -100
  2. Maps coreference links to token-level span boundaries
  3. Sets domain_class from the domain_index

Input:  data/tokenized/{domain}.jsonl
Output: data/labeled/{domain}.jsonl
"""

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from build.models import (
    TokenizedBlock,
    LabeledBlock,
    SpanBounds,
    CoreferenceCluster,
    ENTITY_TYPE_TO_NER,
    NER_LABEL_TO_ID,
    IGNORE_INDEX,
)


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TOKENIZED_DIR = DATA_DIR / "tokenized"
LABELED_DIR = DATA_DIR / "labeled"


# ============================================================================
# NER Label Building
# ============================================================================

def build_ner_labels(block: TokenizedBlock) -> list[int]:
    """Build IOB2 NER label array from entity spans and char_to_token mapping."""

    num_tokens = len(block.tokens)
    # Default: O (outside) for all tokens
    labels = [NER_LABEL_TO_ID["O"]] * num_tokens

    # Mask special tokens and subword continuations
    for i, token in enumerate(block.tokens):
        if token in ("[CLS]", "[SEP]", "[PAD]") or token.startswith("##"):
            labels[i] = IGNORE_INDEX

    # Map each entity span to token indices
    for entity in block.entities:
        # Find all occurrences of entity text in raw_text
        # Use the first occurrence (most common case)
        start_char = block.raw_text.find(entity.text)
        if start_char == -1:
            continue

        end_char = start_char + len(entity.text)

        # Map char positions to token indices
        token_start = None
        token_end = None

        for char_idx in range(start_char, min(end_char, len(block.char_to_token))):
            tok_idx = block.char_to_token[char_idx]
            if tok_idx is not None:
                if token_start is None:
                    token_start = tok_idx
                token_end = tok_idx

        if token_start is None or token_end is None:
            continue

        # Determine NER label type
        ner_type = ENTITY_TYPE_TO_NER.get(entity.entity_type.upper(), "OTHER")

        # Apply B-/I- labels, skipping subword tokens
        first = True
        for tok_idx in range(token_start, token_end + 1):
            # Skip subword continuations - they keep -100
            if block.tokens[tok_idx].startswith("##"):
                continue
            if first:
                labels[tok_idx] = NER_LABEL_TO_ID[f"B-{ner_type}"]
                first = False
            else:
                labels[tok_idx] = NER_LABEL_TO_ID[f"I-{ner_type}"]

    return labels


# ============================================================================
# Coreference Span Mapping
# ============================================================================

def build_coref_clusters(block: TokenizedBlock) -> list[CoreferenceCluster]:
    """Map coreference links to token-level span boundaries."""

    if not block.coreferences:
        return []

    # Group by antecedent to form clusters
    clusters: dict[str, list[SpanBounds]] = {}

    for coref in block.coreferences:
        for text_span in [coref.antecedent, coref.referent]:
            start_char = block.raw_text.find(text_span)
            if start_char == -1:
                continue

            end_char = start_char + len(text_span)

            # Map to token indices
            token_start = None
            token_end = None
            for char_idx in range(start_char, min(end_char, len(block.char_to_token))):
                tok_idx = block.char_to_token[char_idx]
                if tok_idx is not None:
                    if token_start is None:
                        token_start = tok_idx
                    token_end = tok_idx

            if token_start is None or token_end is None:
                continue

            span = SpanBounds(start=token_start, end=token_end, text=text_span)

            key = coref.antecedent
            if key not in clusters:
                clusters[key] = []

            # Avoid duplicate spans in the same cluster
            existing_spans = {(s.start, s.end) for s in clusters[key]}
            if (span.start, span.end) not in existing_spans:
                clusters[key].append(span)

    # Convert to CoreferenceCluster objects, only keep clusters with 2+ mentions
    result = []
    for mentions in clusters.values():
        if len(mentions) >= 2:
            result.append(CoreferenceCluster(mentions=mentions))

    return result


# ============================================================================
# Label a Single Block
# ============================================================================

def label_block(block: TokenizedBlock) -> LabeledBlock:
    """Build NER labels and coref clusters for a single block."""

    ner_labels = build_ner_labels(block)
    coref_clusters = build_coref_clusters(block)

    return LabeledBlock(
        id=block.id,
        raw_text=block.raw_text,
        domain=block.domain,
        domain_index=block.domain_index,
        block_type=block.block_type,
        tokens=block.tokens,
        domain_class=block.domain_index,
        ner_labels=ner_labels,
        coref_clusters=coref_clusters,
    )


# ============================================================================
# I/O
# ============================================================================

def load_tokenized_blocks(domain: str) -> list[TokenizedBlock]:
    path = TOKENIZED_DIR / f"{domain}.jsonl"
    blocks = []
    with open(path) as f:
        for line in f:
            blocks.append(TokenizedBlock.model_validate_json(line))
    return blocks


def save_labeled_blocks(domain: str, blocks: list[LabeledBlock]) -> Path:
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LABELED_DIR / f"{domain}.jsonl"
    with open(output_path, "w") as f:
        for block in blocks:
            f.write(block.model_dump_json() + "\n")
    return output_path


def load_completed_domains() -> set[str]:
    if not LABELED_DIR.exists():
        return set()
    return {p.stem for p in LABELED_DIR.glob("*.jsonl") if p.stat().st_size > 1000}


def get_available_domains() -> list[str]:
    if not TOKENIZED_DIR.exists():
        return []
    return sorted(p.stem for p in TOKENIZED_DIR.glob("*.jsonl") if p.stat().st_size > 1000)


# ============================================================================
# Main
# ============================================================================

def main(selected_domains: list[str] | None = None):
    available = get_available_domains()

    if selected_domains:
        domains = [d for d in selected_domains if d in available]
        invalid = [d for d in selected_domains if d not in available]
        if invalid:
            print(f"WARNING: No tokenized data found for: {invalid}")
    else:
        domains = available

    print("=" * 60)
    print("Phase 1, Step 4: NER & Coreference Label Mapping")
    print("=" * 60)
    print(f"Domains: {len(domains)}")
    print(f"Input directory: {TOKENIZED_DIR}")
    print(f"Output directory: {LABELED_DIR}")
    print("=" * 60)

    completed = load_completed_domains()
    if completed:
        completed_in_scope = completed & set(domains)
        if completed_in_scope:
            print(f"\nResuming: {len(completed_in_scope)} domains already labeled")

    remaining = [d for d in domains if d not in completed]

    if not remaining:
        print("\nAll domains complete!")
        return

    print(f"Domains to process: {len(remaining)}\n")

    overall_start = time.perf_counter()
    total_blocks = 0
    total_entities_mapped = 0
    total_coref_clusters = 0

    for domain in tqdm(remaining, desc="Domains"):
        blocks = load_tokenized_blocks(domain)
        domain_start = time.perf_counter()

        labeled = []
        domain_entities = 0
        domain_clusters = 0

        for block in blocks:
            lb = label_block(block)
            labeled.append(lb)

            # Count non-O, non-IGNORE NER labels
            domain_entities += sum(
                1 for l in lb.ner_labels
                if l not in (NER_LABEL_TO_ID["O"], IGNORE_INDEX) and "B-" in (
                    next((k for k, v in NER_LABEL_TO_ID.items() if v == l), "")
                )
            )
            domain_clusters += len(lb.coref_clusters)

        domain_time = time.perf_counter() - domain_start
        total_blocks += len(labeled)
        total_entities_mapped += domain_entities
        total_coref_clusters += domain_clusters

        output_path = save_labeled_blocks(domain, labeled)
        tqdm.write(f"\n{domain}:")
        tqdm.write(f"  Blocks: {len(labeled)} | Entities mapped: {domain_entities} | Coref clusters: {domain_clusters}")
        tqdm.write(f"  Time: {domain_time:.1f}s | Saved -> {output_path.name}")

    overall_time = time.perf_counter() - overall_start

    print("\n" + "=" * 60)
    print("Label Mapping Complete - Summary")
    print("=" * 60)
    print(f"Domains processed: {len(remaining)}")
    print(f"Total blocks: {total_blocks}")
    print(f"Total entities mapped: {total_entities_mapped}")
    print(f"Total coref clusters: {total_coref_clusters}")
    print(f"Overall time: {overall_time:.1f}s ({overall_time / 60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map entity/coref extractions to token-level labels")
    parser.add_argument("--domains", nargs="+", help="Specific domain(s) to process (default: all available)")
    args = parser.parse_args()
    main(selected_domains=args.domains)
