#!/usr/bin/env python3
"""
Phase 1, Step 1: Generate raw text blocks across ~300 domains using Qwen3.5.

This script iterates through domain categories and generates varied text blocks
with different compositional structures as defined in PROJ.md:
  - 20% single sentences (syntactic foundation)
  - 40% 3-4 sentences (contextual sweet spot)
  - 20% 5-7 sentences (long-distance stress test)
  - 10% adversarial Winograd-style (1-2 sentences)
  - 10% noisy/fragmented (1-3 sentences)

Output: data/raw_texts/{domain_id}.jsonl
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from groq import AsyncGroq
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from build.domains import DOMAINS, DOMAIN_IDS, DOMAIN_BY_ID
from build.models import RawTextBlock


# ============================================================================
# Timing Stats
# ============================================================================

@dataclass
class TimingStats:
    """Track timing for a domain generation run."""
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
# Configuration
# ============================================================================

# Groq Cloud configuration
MODEL = "qwen/qwen3-32b"

# Generation targets per domain
BLOCKS_PER_DOMAIN = 500

# Concurrency control
MAX_CONCURRENT_REQUESTS = 4

# Batch size - generate multiple blocks per API call
BLOCKS_PER_BATCH = 25

# Output paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_TEXTS_DIR = DATA_DIR / "raw_texts"


# Domain definitions loaded from domains.json (see build/domains.py)


# ============================================================================
# Block Composition Types
# ============================================================================

class BlockType(Enum):
    SINGLE_SENTENCE = "single"      # 20% - 1 sentence
    CONTEXTUAL = "contextual"        # 40% - 3-4 sentences
    LONG_DISTANCE = "long_distance"  # 20% - 5-7 sentences
    ADVERSARIAL = "adversarial"      # 10% - Winograd-style, 1-2 sentences
    NOISY = "noisy"                  # 10% - Fragmented/informal, 1-3 sentences


@dataclass
class BlockConfig:
    block_type: BlockType
    min_sentences: int
    max_sentences: int
    weight: float  # Probability weight


BLOCK_CONFIGS = [
    BlockConfig(BlockType.SINGLE_SENTENCE, 1, 1, 0.20),
    BlockConfig(BlockType.CONTEXTUAL, 3, 4, 0.40),
    BlockConfig(BlockType.LONG_DISTANCE, 5, 7, 0.20),
    BlockConfig(BlockType.ADVERSARIAL, 1, 2, 0.10),
    BlockConfig(BlockType.NOISY, 1, 3, 0.10),
]


def get_block_distribution(total_blocks: int) -> list[BlockConfig]:
    """Generate a list of block configs matching the target distribution."""
    result = []
    for config in BLOCK_CONFIGS:
        count = int(total_blocks * config.weight)
        result.extend([config] * count)

    # Fill any remainder with contextual blocks
    while len(result) < total_blocks:
        result.append(BLOCK_CONFIGS[1])  # CONTEXTUAL

    random.shuffle(result)
    return result


# ============================================================================
# Prompt Templates
# ============================================================================

def get_block_type_description(block_type: BlockType, min_sent: int, max_sent: int) -> str:
    """Get description for a block type."""
    if block_type == BlockType.SINGLE_SENTENCE:
        return f"SINGLE: Write exactly 1 information-dense sentence with at least one named entity."
    elif block_type == BlockType.CONTEXTUAL:
        return f"CONTEXTUAL: Write {min_sent}-{max_sent} sentences with clear entity relationships, multiple entity types, and pronouns referring back to earlier entities."
    elif block_type == BlockType.LONG_DISTANCE:
        return f"LONG: Write {min_sent}-{max_sent} sentences with pronouns referring to entities mentioned several sentences earlier. Maintain topical coherence."
    elif block_type == BlockType.ADVERSARIAL:
        return f"ADVERSARIAL: Write {min_sent}-{max_sent} Winograd-schema style sentences with ambiguous pronouns requiring semantic understanding to resolve."
    elif block_type == BlockType.NOISY:
        return f"NOISY: Write {min_sent}-{max_sent} sentences in informal/fragmented style (like chat logs or notes). May include typos, abbreviations, missing punctuation."
    return ""


def build_batch_prompt(domain: str, block_configs: list[BlockConfig]) -> str:
    """Build a prompt requesting multiple blocks at once."""

    domain_readable = domain.replace("_", " ")

    # Build the numbered list of requested blocks
    block_requests = []
    for i, config in enumerate(block_configs, 1):
        desc = get_block_type_description(config.block_type, config.min_sentences, config.max_sentences)
        block_requests.append(f"{i}. [{config.block_type.value.upper()}] {desc}")

    blocks_list = "\n".join(block_requests)

    prompt = f"""Generate {len(block_configs)} distinct text passages about {domain_readable}.

Each passage must:
- Include specific entities (people, organizations, locations, dates, quantities)
- Use pronouns that refer back to mentioned entities
- Be factually plausible and domain-appropriate
- Be completely independent from other passages

Generate these {len(block_configs)} passages:
{blocks_list}

Output format - use exactly this structure with === separators:
===1===
[First passage text here]
===2===
[Second passage text here]
===3===
[Third passage text here]
...and so on for all {len(block_configs)} passages.

Output ONLY the passages in the format above. No commentary or explanations."""

    return prompt


def clean_block_text(text: str) -> str:
    """Strip thinking tags and type label prefixes from a block."""
    import re
    # Remove <think>...</think> blocks (including partial/unclosed)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.strip()
    # Remove type label prefixes like "CONTEXTUAL:" or "LONG_DISTANCE:"
    text = re.sub(r'^(?:SINGLE|CONTEXTUAL|LONG_DISTANCE|ADVERSARIAL|NOISY)\s*:\s*', '', text)
    return text.strip()


def parse_batch_response(response_text: str, expected_count: int) -> list[str]:
    """Parse a batched response into individual text blocks."""
    import re

    # Strip any leading thinking block before the first separator
    response_text = re.sub(r'^<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)

    blocks = []

    # Split by ===N=== pattern
    pattern = r'===\d+===\s*'
    parts = re.split(pattern, response_text)

    # Filter out empty parts, clean, and strip whitespace
    for part in parts:
        text = clean_block_text(part)
        if text and len(text) >= 20:  # Basic validation
            blocks.append(text)

    return blocks


# ============================================================================
# Generation Logic
# ============================================================================

class TextGenerator:
    def __init__(self):
        self.client = AsyncGroq()  # Uses GROQ_API_KEY env var
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def generate_batch(
        self,
        domain: str,
        domain_index: int,
        batch_index: int,
        block_configs: list[BlockConfig],
        start_block_index: int,
    ) -> tuple[list[RawTextBlock], float]:
        """Generate a batch of text blocks. Returns (blocks, request_time_sec)."""

        prompt = build_batch_prompt(domain, block_configs)

        async with self.semaphore:
            start_time = time.perf_counter()
            try:
                response = await self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=8192,
                    reasoning_effort="none",
                )
                elapsed = time.perf_counter() - start_time

                raw_response = (response.choices[0].message.content or "").strip()
                parsed_texts = parse_batch_response(raw_response, len(block_configs))

                blocks = []
                for i, text in enumerate(parsed_texts):
                    if i < len(block_configs):
                        blocks.append(RawTextBlock(
                            id=f"{domain}-{start_block_index + i:04d}",
                            domain=domain,
                            domain_index=domain_index,
                            block_index=start_block_index + i,
                            block_type=block_configs[i].block_type.value,
                            raw_text=text,
                        ))

                return blocks, elapsed

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                print(f"\nError generating batch {batch_index} for {domain}: {e}")
                return [], elapsed

    async def generate_domain(
        self,
        domain: str,
        domain_index: int,
        num_blocks: int = BLOCKS_PER_DOMAIN,
    ) -> tuple[list[RawTextBlock], TimingStats]:
        """Generate all blocks for a single domain. Returns (blocks, timing_stats)."""

        block_configs = get_block_distribution(num_blocks)
        stats = TimingStats(domain=domain, total_blocks=num_blocks)

        # Split into batches
        batches = []
        for i in range(0, len(block_configs), BLOCKS_PER_BATCH):
            batch_configs = block_configs[i:i + BLOCKS_PER_BATCH]
            batches.append((i // BLOCKS_PER_BATCH, batch_configs, i))

        domain_start = time.perf_counter()

        tasks = [
            self.generate_batch(domain, domain_index, batch_idx, configs, start_idx)
            for batch_idx, configs, start_idx in batches
        ]

        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"  {domain} (batches)",
            leave=False,
        )

        stats.total_time_sec = time.perf_counter() - domain_start

        all_blocks = []
        for batch_blocks, req_time in results:
            stats.request_times.append(req_time)
            stats.successful_blocks += len(batch_blocks)
            all_blocks.extend(batch_blocks)

        stats.failed_blocks = num_blocks - stats.successful_blocks

        return all_blocks, stats


# ============================================================================
# Persistence
# ============================================================================

def save_domain_blocks(domain: str, blocks: list[RawTextBlock]) -> Path:
    """Save blocks for a domain to JSONL file."""
    RAW_TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_TEXTS_DIR / f"{domain}.jsonl"

    with open(output_path, "w") as f:
        for block in blocks:
            f.write(block.model_dump_json() + "\n")

    return output_path


def load_completed_domains() -> set[str]:
    """Check which domains have already been processed."""
    if not RAW_TEXTS_DIR.exists():
        return set()

    completed = set()
    for path in RAW_TEXTS_DIR.glob("*.jsonl"):
        # Only count as complete if file has substantial content
        if path.stat().st_size > 1000:
            completed.add(path.stem)

    return completed


# ============================================================================
# Main Entry Point
# ============================================================================

async def main(selected_domains: list[str] | None = None):
    # Filter to selected domains if specified
    if selected_domains:
        domain_list = [DOMAIN_BY_ID[d] for d in selected_domains if d in DOMAIN_BY_ID]
        invalid = [d for d in selected_domains if d not in DOMAIN_BY_ID]
        if invalid:
            print(f"WARNING: Unknown domains ignored: {invalid}")
    else:
        domain_list = list(DOMAINS)

    print("=" * 60)
    print("Phase 1, Step 1: Raw Text Generation")
    print("=" * 60)
    print(f"Domains: {len(domain_list)}")
    print(f"Blocks per domain: {BLOCKS_PER_DOMAIN}")
    print(f"Total target blocks: {len(domain_list) * BLOCKS_PER_DOMAIN:,}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Output directory: {RAW_TEXTS_DIR}")
    print("=" * 60)

    # Check for already completed domains (resume support)
    completed = load_completed_domains()
    if completed:
        completed_in_scope = completed & {d.id for d in domain_list}
        if completed_in_scope:
            print(f"\nResuming: {len(completed_in_scope)} domains already completed")

    remaining_domains = [d for d in domain_list if d.id not in completed]

    if not remaining_domains:
        print("\nAll domains complete!")
        return

    print(f"Domains to process: {len(remaining_domains)}\n")

    generator = TextGenerator()
    all_stats: list[TimingStats] = []
    overall_start = time.perf_counter()

    for domain in tqdm(remaining_domains, desc="Domains"):
        blocks, stats = await generator.generate_domain(domain.id, domain.index)
        all_stats.append(stats)

        if blocks:
            output_path = save_domain_blocks(domain.id, blocks)
            tqdm.write(f"\n{domain.id}:")
            tqdm.write(stats.summary())
            tqdm.write(f"  Saved -> {output_path.name}")
        else:
            tqdm.write(f"\n  WARNING: No blocks generated for {domain.id}")

    overall_time = time.perf_counter() - overall_start

    # Print overall summary
    print("\n" + "=" * 60)
    print("Generation Complete - Summary")
    print("=" * 60)
    total_successful = sum(s.successful_blocks for s in all_stats)
    total_failed = sum(s.failed_blocks for s in all_stats)
    total_requests = sum(len(s.request_times) for s in all_stats)
    avg_request = sum(sum(s.request_times) for s in all_stats) / total_requests if total_requests else 0

    print(f"Domains processed: {len(all_stats)}")
    print(f"Total blocks: {total_successful} successful, {total_failed} failed")
    print(f"Overall time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print(f"Average request time: {avg_request:.2f}s")
    print(f"Overall rate: {(total_successful / overall_time) * 60:.1f} blocks/min")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raw text blocks across domains")
    parser.add_argument("--domains", nargs="+", help="Specific domain(s) to process (default: all)")
    args = parser.parse_args()
    asyncio.run(main(selected_domains=args.domains))
