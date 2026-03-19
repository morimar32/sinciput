"""
Dataset and collation for multi-task training on sinciput JSONL data.

Reads TrainingExample JSONL files produced by the build pipeline, converts
pre-tokenized WordPiece token strings to IDs via vocabulary lookup (no
re-tokenization), and provides a collate_fn with dynamic padding.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"


class SinciputDataset(Dataset):
    """
    Loads training examples from JSONL files and serves them as unpadded tensors.

    Each example is a dict with pre-tokenized WordPiece tokens from the build
    pipeline. Token strings are converted to IDs via the tokenizer vocabulary
    (convert_tokens_to_ids), not re-tokenized.
    """

    def __init__(self, data_dir: str, domains: list[str] | None = None, max_examples: int | None = None):
        self.data_dir = Path(data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.examples: list[dict] = []

        # Discover JSONL files
        if domains:
            files = [self.data_dir / f"{d}.jsonl" for d in domains]
        else:
            files = sorted(self.data_dir.glob("*.jsonl"))

        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"Training data not found: {f}")
            with open(f) as fh:
                for line in fh:
                    self.examples.append(json.loads(line))
                    if max_examples and len(self.examples) >= max_examples:
                        break
            if max_examples and len(self.examples) >= max_examples:
                break

        # Round-trip sanity check: verify tokenizer vocab matches build pipeline
        if self.examples:
            sample = self.examples[0]
            tokens = sample["tokens"]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            reconstructed = self.tokenizer.convert_ids_to_tokens(ids)
            # Check first few real tokens (skip [CLS])
            for i in range(1, min(5, len(tokens))):
                if tokens[i] != reconstructed[i]:
                    raise ValueError(
                        f"Tokenizer vocab mismatch at position {i}: "
                        f"'{tokens[i]}' -> id {ids[i]} -> '{reconstructed[i]}'. "
                        f"Ensure build and train use the same model: {MODEL_NAME}"
                    )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex["tokens"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "ner_labels": torch.tensor(ex["ner_labels"], dtype=torch.long),
            "parsing_arcs": torch.tensor(ex["parsing_arcs"], dtype=torch.long),
            "domain_class": torch.tensor(ex["domain_class"], dtype=torch.long),
            "coref_clusters": ex["coref_clusters"],  # list of dicts, variable structure
            "seq_len": len(tokens),
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Dynamic padding collation: pads to max length in batch, not 512.

    Returns:
        input_ids:     [B, max_len] padded with pad_token_id (0)
        attention_mask: [B, max_len] 1=real, 0=pad
        ner_labels:    [B, max_len] padded with -100
        parsing_arcs:  [B, max_len] padded with -100
        domain_class:  [B]
        coref_clusters: list of list-of-dicts (raw, variable structure)
        seq_lens:      [B] original lengths
    """
    seq_lens = [item["seq_len"] for item in batch]
    max_len = max(seq_lens)

    # Tokenizer pad_token_id for MiniLM is 0
    pad_token_id = 0
    ignore_index = -100

    input_ids = []
    attention_mask = []
    ner_labels = []
    parsing_arcs = []
    domain_class = []
    coref_clusters = []

    for item in batch:
        length = item["seq_len"]
        pad_len = max_len - length

        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([torch.ones(length, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        )
        ner_labels.append(
            torch.cat([item["ner_labels"], torch.full((pad_len,), ignore_index, dtype=torch.long)])
        )
        parsing_arcs.append(
            torch.cat([item["parsing_arcs"], torch.full((pad_len,), ignore_index, dtype=torch.long)])
        )
        domain_class.append(item["domain_class"])
        coref_clusters.append(item["coref_clusters"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "ner_labels": torch.stack(ner_labels),
        "parsing_arcs": torch.stack(parsing_arcs),
        "domain_class": torch.stack(domain_class),
        "coref_clusters": coref_clusters,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
    }
