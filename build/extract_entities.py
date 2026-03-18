#!/usr/bin/env python3
"""
Phase 1, Step 2: Extract entities and coreference links from raw text blocks.

Reads raw text blocks from data/raw_texts/{domain}.jsonl and sends them back
to Qwen via Groq to extract:
  - Named entities (PERSON, ORG, LOC, DATE, QUANTITY, EVENT, PRODUCT, OTHER)
  - Coreference links (antecedent → referent pairs)

Output: data/extracted/{domain}.jsonl
"""

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from groq import AsyncGroq
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from build.models import (
    RawTextBlock,
    EntityMention,
    CoreferenceLink,
    ExtractedBlock,
)


# ============================================================================
# Configuration
# ============================================================================

MODEL = "qwen/qwen3-32b"

# Concurrency control
MAX_CONCURRENT_REQUESTS = 4

# Batch size - extract from multiple blocks per API call
BLOCKS_PER_BATCH = 10

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_TEXTS_DIR = DATA_DIR / "raw_texts"
EXTRACTED_DIR = DATA_DIR / "extracted"


# ============================================================================
# Timing Stats
# ============================================================================

@dataclass
class TimingStats:
    """Track timing for an extraction run."""
    domain: str
    total_blocks: int = 0
    successful_blocks: int = 0
    failed_blocks: int = 0
    total_time_sec: float = 0.0
    request_times: list[float] = field(default_factory=list)

    @property
    def avg_request_time(self) -> float:
        return sum(self.request_times) / len(self.request_times) if self.request_times else 0.0

    @property
    def blocks_per_minute(self) -> float:
        return (self.successful_blocks / self.total_time_sec) * 60 if self.total_time_sec > 0 else 0.0

    def summary(self) -> str:
        return (
            f"  Blocks: {self.successful_blocks}/{self.total_blocks} "
            f"({self.failed_blocks} failed)\n"
            f"  Total time: {self.total_time_sec:.1f}s | "
            f"Avg request: {self.avg_request_time:.2f}s | "
            f"Rate: {self.blocks_per_minute:.1f} blocks/min"
        )


# ============================================================================
# Prompt & Parsing
# ============================================================================

def build_extraction_prompt(blocks: list[RawTextBlock]) -> str:
    """Build a prompt requesting entity/coref extraction for multiple blocks."""

    block_sections = []
    for i, block in enumerate(blocks, 1):
        block_sections.append(f"===TEXT {i}===\n{block.raw_text}")

    texts = "\n\n".join(block_sections)

    prompt = f"""Extract named entities and coreference links from each text below.

For each text, output a JSON object with:
- "entities": list of {{"text": "exact span", "entity_type": "TYPE"}}
  Types: PERSON, ORG, LOC, DATE, QUANTITY, EVENT, PRODUCT, OTHER
- "coreferences": list of {{"antecedent": "full noun phrase", "referent": "pronoun or short reference"}}
  Only include coreferences where a pronoun/short reference refers back to a specific named entity or noun phrase.

Rules:
- Entity "text" must be the EXACT substring from the passage (case-sensitive match)
- Only extract clear, unambiguous coreference links
- If no entities or coreferences exist, use empty lists

{texts}

Output format - use exactly this structure:
===RESULT 1===
{{"entities": [...], "coreferences": [...]}}
===RESULT 2===
{{"entities": [...], "coreferences": [...]}}
...and so on for all {len(blocks)} texts.

Output ONLY valid JSON results in the format above. No commentary."""

    return prompt


def parse_extraction_response(response_text: str, expected_count: int) -> list[dict | None]:
    """Parse the batched extraction response into individual results."""

    # Strip thinking tags
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'<think>.*', '', response_text, flags=re.DOTALL)

    # Split by ===RESULT N=== pattern
    pattern = r'===\s*RESULT\s+\d+\s*===\s*'
    parts = re.split(pattern, response_text)

    results = []
    for part in parts:
        text = part.strip()
        if not text:
            continue

        # Try to extract JSON from the part
        try:
            # Find the JSON object in the text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                results.append(data)
            else:
                results.append(None)
        except json.JSONDecodeError:
            results.append(None)

    # Pad with None if we got fewer results than expected
    while len(results) < expected_count:
        results.append(None)

    return results[:expected_count]


def parse_single_result(data: dict | None) -> tuple[list[EntityMention], list[CoreferenceLink]]:
    """Parse a single extraction result dict into typed objects."""
    if data is None:
        return [], []

    entities = []
    for e in data.get("entities", []):
        if not isinstance(e, dict):
            continue
        try:
            entities.append(EntityMention(
                text=e["text"],
                entity_type=e.get("entity_type", e.get("type", "OTHER")),
            ))
        except (KeyError, ValueError):
            continue

    coreferences = []
    for c in data.get("coreferences", []):
        if not isinstance(c, dict):
            continue
        try:
            coreferences.append(CoreferenceLink(
                antecedent=c["antecedent"],
                referent=c["referent"],
            ))
        except (KeyError, ValueError):
            continue

    return entities, coreferences


# ============================================================================
# Extraction Logic
# ============================================================================

class EntityExtractor:
    def __init__(self):
        self.client = AsyncGroq()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def extract_batch(
        self,
        blocks: list[RawTextBlock],
        batch_index: int,
        domain: str,
    ) -> tuple[list[ExtractedBlock], float]:
        """Extract entities/coref from a batch of blocks."""

        prompt = build_extraction_prompt(blocks)

        async with self.semaphore:
            start_time = time.perf_counter()
            try:
                response = await self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,  # Low temp for structured extraction
                    max_tokens=8192,
                    reasoning_effort="none",
                )
                elapsed = time.perf_counter() - start_time

                raw_response = (response.choices[0].message.content or "").strip()
                parsed_results = parse_extraction_response(raw_response, len(blocks))

                extracted = []
                for block, result in zip(blocks, parsed_results):
                    entities, coreferences = parse_single_result(result)

                    # Validate entity spans actually exist in the text
                    valid_entities = [
                        e for e in entities if e.text in block.raw_text
                    ]

                    extracted.append(ExtractedBlock(
                        **block.model_dump(),
                        entities=valid_entities,
                        coreferences=coreferences,
                    ))

                return extracted, elapsed

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                print(f"\nError extracting batch {batch_index} for {domain}: {e}")
                # Return blocks with empty extractions so we don't lose them
                fallback = [
                    ExtractedBlock(**block.model_dump(), entities=[], coreferences=[])
                    for block in blocks
                ]
                return fallback, elapsed

    async def extract_domain(
        self,
        domain: str,
        blocks: list[RawTextBlock],
    ) -> tuple[list[ExtractedBlock], TimingStats]:
        """Extract entities/coref for all blocks in a domain."""

        stats = TimingStats(domain=domain, total_blocks=len(blocks))

        # Split into batches
        batches = []
        for i in range(0, len(blocks), BLOCKS_PER_BATCH):
            batch = blocks[i:i + BLOCKS_PER_BATCH]
            batches.append((i // BLOCKS_PER_BATCH, batch))

        domain_start = time.perf_counter()

        tasks = [
            self.extract_batch(batch, batch_idx, domain)
            for batch_idx, batch in batches
        ]

        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"  {domain} (batches)",
            leave=False,
        )

        stats.total_time_sec = time.perf_counter() - domain_start

        all_extracted = []
        for batch_extracted, req_time in results:
            stats.request_times.append(req_time)
            stats.successful_blocks += sum(
                1 for b in batch_extracted if b.entities or b.coreferences
            )
            all_extracted.extend(batch_extracted)

        stats.failed_blocks = stats.total_blocks - stats.successful_blocks

        return all_extracted, stats


# ============================================================================
# I/O
# ============================================================================

def load_raw_blocks(domain: str) -> list[RawTextBlock]:
    """Load raw text blocks for a domain."""
    path = RAW_TEXTS_DIR / f"{domain}.jsonl"
    blocks = []
    with open(path) as f:
        for line in f:
            blocks.append(RawTextBlock.model_validate_json(line))
    return blocks


def save_extracted_blocks(domain: str, blocks: list[ExtractedBlock]) -> Path:
    """Save extracted blocks to JSONL."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXTRACTED_DIR / f"{domain}.jsonl"
    with open(output_path, "w") as f:
        for block in blocks:
            f.write(block.model_dump_json() + "\n")
    return output_path


def load_completed_domains() -> set[str]:
    """Check which domains have already been extracted."""
    if not EXTRACTED_DIR.exists():
        return set()
    completed = set()
    for path in EXTRACTED_DIR.glob("*.jsonl"):
        if path.stat().st_size > 1000:
            completed.add(path.stem)
    return completed


def get_available_domains() -> list[str]:
    """Get domains that have raw text files."""
    if not RAW_TEXTS_DIR.exists():
        return []
    return sorted(p.stem for p in RAW_TEXTS_DIR.glob("*.jsonl") if p.stat().st_size > 1000)


# ============================================================================
# Main
# ============================================================================

async def main(selected_domains: list[str] | None = None):
    available = get_available_domains()

    if selected_domains:
        domains = [d for d in selected_domains if d in available]
        invalid = [d for d in selected_domains if d not in available]
        if invalid:
            print(f"WARNING: No raw text found for: {invalid}")
    else:
        domains = available

    print("=" * 60)
    print("Phase 1, Step 2: Entity & Coreference Extraction")
    print("=" * 60)
    print(f"Domains: {len(domains)}")
    print(f"Blocks per batch: {BLOCKS_PER_BATCH}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Input directory: {RAW_TEXTS_DIR}")
    print(f"Output directory: {EXTRACTED_DIR}")
    print("=" * 60)

    # Resume support
    completed = load_completed_domains()
    if completed:
        completed_in_scope = completed & set(domains)
        if completed_in_scope:
            print(f"\nResuming: {len(completed_in_scope)} domains already extracted")

    remaining = [d for d in domains if d not in completed]

    if not remaining:
        print("\nAll domains complete!")
        return

    print(f"Domains to process: {len(remaining)}\n")

    extractor = EntityExtractor()
    all_stats: list[TimingStats] = []
    overall_start = time.perf_counter()

    for domain in tqdm(remaining, desc="Domains"):
        blocks = load_raw_blocks(domain)
        extracted, stats = await extractor.extract_domain(domain, blocks)
        all_stats.append(stats)

        output_path = save_extracted_blocks(domain, extracted)
        tqdm.write(f"\n{domain}:")
        tqdm.write(stats.summary())
        tqdm.write(f"  Saved -> {output_path.name}")

    overall_time = time.perf_counter() - overall_start

    # Summary
    print("\n" + "=" * 60)
    print("Extraction Complete - Summary")
    print("=" * 60)
    total_blocks = sum(s.total_blocks for s in all_stats)
    total_with_extractions = sum(s.successful_blocks for s in all_stats)
    total_requests = sum(len(s.request_times) for s in all_stats)
    avg_request = sum(sum(s.request_times) for s in all_stats) / total_requests if total_requests else 0

    print(f"Domains processed: {len(all_stats)}")
    print(f"Total blocks: {total_blocks}")
    print(f"Blocks with extractions: {total_with_extractions}")
    print(f"Overall time: {overall_time:.1f}s ({overall_time / 60:.1f} min)")
    print(f"Average request time: {avg_request:.2f}s")
    print(f"Overall rate: {(total_blocks / overall_time) * 60:.1f} blocks/min")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entities and coreferences from raw text blocks")
    parser.add_argument("--domains", nargs="+", help="Specific domain(s) to process (default: all available)")
    args = parser.parse_args()
    asyncio.run(main(selected_domains=args.domains))
