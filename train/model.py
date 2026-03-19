"""
Multi-task model: shared MiniLM encoder with 4 task heads.

Tasks:
  1. Domain classification (CLS token → 300 classes)
  2. Named entity recognition (sequence → 17 IOB2 labels)
  3. Dependency parsing (biaffine arc scoring)
  4. Coreference resolution (span-pair scoring with marginalized log-likelihood)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
HIDDEN_SIZE = 384


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BiaffineScorer(nn.Module):
    """Biaffine scoring: x^T U y for arc prediction."""

    def __init__(self, in_dim1: int, in_dim2: int, out_dim: int, bias_x: bool = True):
        super().__init__()
        self.bias_x = bias_x
        self.out_dim = out_dim
        x_dim = in_dim1 + (1 if bias_x else 0)
        # U: [out_dim, x_dim, in_dim2]
        self.U = nn.Parameter(torch.zeros(out_dim, x_dim, in_dim2))
        nn.init.xavier_normal_(self.U)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, in_dim1]  (head representations)
            y: [B, T, in_dim2]  (dep representations)
        Returns:
            scores: [B, T, T, out_dim] or [B, T, T] if out_dim=1
        """
        if self.bias_x:
            ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ones], dim=-1)

        # x: [B, T, x_dim], y: [B, T, in_dim2], U: [out_dim, x_dim, in_dim2]
        # scores[b, i, j, k] = x[b,i,:] @ U[k,:,:] @ y[b,j,:]
        scores = torch.einsum("bix,kxy,bjy->bijk", x, self.U, y)

        if self.out_dim == 1:
            scores = scores.squeeze(-1)  # [B, T, T]
        return scores


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class DomainClassificationHead(nn.Module):
    """[CLS] → domain logits (300 classes)."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_classes: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, cls_embedding: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Args:
            cls_embedding: [B, H]
            labels: [B] domain class indices
        """
        logits = self.classifier(self.dropout(cls_embedding))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class NERHead(nn.Module):
    """Sequence labeling for IOB2 NER (17 labels)."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, num_labels: int = 17, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, sequence_output: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Args:
            sequence_output: [B, T, H]
            labels: [B, T] with -100 for ignored positions
        """
        logits = self.classifier(self.dropout(sequence_output))  # [B, T, num_labels]
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}


class DependencyParsingHead(nn.Module):
    """Biaffine arc scoring for dependency parsing."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, arc_dim: int = 500, dropout: float = 0.33):
        super().__init__()
        self.mlp_arc_head = nn.Sequential(
            nn.Linear(hidden_size, arc_dim), nn.ELU(), nn.Dropout(dropout)
        )
        self.mlp_arc_dep = nn.Sequential(
            nn.Linear(hidden_size, arc_dim), nn.ELU(), nn.Dropout(dropout)
        )
        self.arc_biaffine = BiaffineScorer(arc_dim, arc_dim, 1, bias_x=True)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, sequence_output: torch.Tensor, arcs: torch.Tensor | None = None):
        """
        Args:
            sequence_output: [B, T, H]
            arcs: [B, T] head indices, -100 for ignored
        Returns:
            scores [B, T, T]: scores[b, i, j] = prob token j is head of token i
        """
        arc_head = self.mlp_arc_head(sequence_output)  # [B, T, arc_dim]
        arc_dep = self.mlp_arc_dep(sequence_output)     # [B, T, arc_dim]
        scores = self.arc_biaffine(arc_dep, arc_head)   # [B, T, T] - dep x head

        loss = None
        if arcs is not None:
            B, T, _ = scores.shape
            loss = self.loss_fn(scores.reshape(B * T, T), arcs.reshape(B * T))
        return {"loss": loss, "logits": scores}


class CoreferenceHead(nn.Module):
    """
    Span-pair coreference resolution with marginalized log-likelihood loss.

    Pipeline: enumerate spans → score mentions → prune top-K → pairwise scoring → MLL loss.
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        span_dim: int = 256,
        ffnn_dim: int = 150,
        max_span_width: int = 30,
        max_num_spans: int = 250,
        top_k_ratio: float = 0.4,
        width_emb_dim: int = 20,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.max_span_width = max_span_width
        self.max_num_spans = max_num_spans
        self.top_k_ratio = top_k_ratio

        # Span width embedding
        self.width_embedding = nn.Embedding(max_span_width, width_emb_dim)

        # Head attention over span tokens
        self.head_attn = nn.Linear(hidden_size, 1)

        # Span representation projection: [start; end; head; width_emb] → span_dim
        self.span_proj = nn.Sequential(
            nn.Linear(hidden_size * 3 + width_emb_dim, span_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Mention scorer: span_dim → 1
        self.mention_scorer = nn.Sequential(
            nn.Linear(span_dim, ffnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffnn_dim, 1),
        )

        # Distance embedding for pairwise scoring
        self.dist_embedding = nn.Embedding(10, 20)  # bucketed distances
        self.register_buffer("_dist_bins", torch.tensor([1, 2, 3, 4, 5, 8, 16, 32, 64]))

        # Pairwise scorer: [g_i; g_j; g_i*g_j; dist_emb] → 1
        self.pair_scorer = nn.Sequential(
            nn.Linear(span_dim * 3 + 20, ffnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffnn_dim, 1),
        )

    def _enumerate_spans(self, seq_len: int, device: torch.device):
        """Enumerate (start, end) pairs skipping [CLS] (0) and [SEP] (seq_len-1). Returns tensor [N, 2]."""
        starts = []
        ends = []
        for i in range(1, seq_len - 1):
            for j in range(i, min(i + self.max_span_width, seq_len - 1)):
                starts.append(i)
                ends.append(j)
                if len(starts) >= self.max_num_spans:
                    break
            if len(starts) >= self.max_num_spans:
                break
        if not starts:
            return None, None, None
        start_idx = torch.tensor(starts, device=device, dtype=torch.long)
        end_idx = torch.tensor(ends, device=device, dtype=torch.long)
        widths = end_idx - start_idx  # 0-indexed
        return start_idx, end_idx, widths

    def _get_span_repr(self, hidden: torch.Tensor, start_idx: torch.Tensor,
                        end_idx: torch.Tensor, widths: torch.Tensor):
        """
        Build span representations vectorized: [x_start; x_end; x_hat; phi(width)].

        Uses a single gather instead of looping over span widths.

        Args:
            hidden: [T, H] single example
            start_idx: [N] start indices
            end_idx: [N] end indices
            widths: [N] span widths (end - start)
        Returns:
            [N, span_dim]
        """
        device = hidden.device
        N = start_idx.size(0)
        H = hidden.size(1)
        max_w = widths.max().item() + 1

        start_embs = hidden[start_idx]  # [N, H]
        end_embs = hidden[end_idx]      # [N, H]
        width_embs = self.width_embedding(widths)  # [N, width_emb_dim]

        # Build [N, max_w] position index matrix and mask in one shot
        offsets = torch.arange(max_w, device=device).unsqueeze(0)  # [1, max_w]
        positions = start_idx.unsqueeze(1) + offsets               # [N, max_w]
        span_mask = offsets <= widths.unsqueeze(1)                  # [N, max_w]

        # Clamp out-of-bounds positions (masked out later) and gather all at once
        positions = positions.clamp(0, hidden.size(0) - 1)
        span_tokens = hidden[positions]  # [N, max_w, H]

        # Attention scores: [N, max_w]
        attn_scores = self.head_attn(span_tokens).squeeze(-1)  # [N, max_w]
        attn_scores = attn_scores.masked_fill(~span_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)  # [N, max_w]
        head_embs = (attn_weights.unsqueeze(-1) * span_tokens).sum(1)  # [N, H]

        span_repr = torch.cat([start_embs, end_embs, head_embs, width_embs], dim=-1)
        return self.span_proj(span_repr)  # [N, span_dim]

    def _bucket_distance(self, distances: torch.Tensor) -> torch.Tensor:
        """Bucket span distances into 10 bins."""
        return torch.bucketize(distances, self._dist_bins)

    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lens: torch.Tensor,
        coref_clusters: list[list[dict]] | None = None,
    ):
        """
        Process each example independently (no batched span ops).

        Args:
            sequence_output: [B, T, H]
            attention_mask: [B, T]
            seq_lens: [B]
            coref_clusters: list of cluster lists per example
        Returns:
            {"loss": scalar, "logits": None}
        """
        B = sequence_output.size(0)
        device = sequence_output.device
        total_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            sl = seq_lens[b].item()
            hidden = sequence_output[b, :sl]  # [T, H]

            # Enumerate candidate spans
            start_idx, end_idx, widths = self._enumerate_spans(sl, device)
            if start_idx is None:
                continue
            N = start_idx.size(0)

            # Get span representations and mention scores
            span_repr = self._get_span_repr(hidden, start_idx, end_idx, widths)  # [N, span_dim]
            mention_scores = self.mention_scorer(span_repr).squeeze(-1)  # [N]

            # Prune to top-K
            k = max(1, int(N * self.top_k_ratio))
            k = min(k, N)
            top_indices = torch.topk(mention_scores, k).indices
            top_indices_sorted = top_indices.sort().values

            top_starts = start_idx[top_indices_sorted]  # [K]
            top_ends = end_idx[top_indices_sorted]      # [K]
            top_repr = span_repr[top_indices_sorted]    # [K, span_dim]
            top_ms = mention_scores[top_indices_sorted]  # [K]
            K = top_repr.size(0)

            if K < 2:
                continue

            # Build gold cluster mapping: (start, end) → cluster_id
            gold_span_to_cluster = {}
            clusters = coref_clusters[b] if coref_clusters else []
            for cluster_id, cluster in enumerate(clusters):
                for mention in cluster["mentions"]:
                    gold_span_to_cluster[(mention["start"], mention["end"])] = cluster_id

            # Vectorized pairwise scoring for all (i, j) where j < i
            # Build upper-triangular antecedent pairs
            # For span i, antecedents are spans 0..i-1
            # Use a [K, K] score matrix where scores[i, j] is the antecedent score of j for i

            # Pairwise representations: for all (i, j) pairs
            # repr_i expanded: [K, K, D], repr_j expanded: [K, K, D]
            D = top_repr.size(1)
            ri = top_repr.unsqueeze(1).expand(K, K, D)  # [K, K, D] - each row is span i
            rj = top_repr.unsqueeze(0).expand(K, K, D)  # [K, K, D] - each col is span j

            # Distance features
            dists = (top_starts.unsqueeze(0) - top_starts.unsqueeze(1)).abs()  # [K, K]
            dist_embs = self.dist_embedding(self._bucket_distance(dists))  # [K, K, 20]

            # Pair input: [K, K, 3*D + 20]
            pair_input = torch.cat([ri, rj, ri * rj, dist_embs], dim=-1)
            pair_scores = self.pair_scorer(pair_input).squeeze(-1)  # [K, K]

            # Total antecedent scores: s_m(i) + s_m(j) + s_a(i,j)
            # [K, K] where [i, j] = score that span j is antecedent of span i
            ant_scores = top_ms.unsqueeze(1) + top_ms.unsqueeze(0) + pair_scores  # [K, K]

            # Mask: only j < i are valid antecedents
            causal_mask = torch.triu(torch.ones(K, K, device=device, dtype=torch.bool), diagonal=0)  # upper tri + diag
            ant_scores = ant_scores.masked_fill(causal_mask, float("-inf"))

            # Prepend epsilon (score=0) as column 0: [K, K+1]
            epsilon = torch.zeros(K, 1, device=device)
            all_scores = torch.cat([epsilon, ant_scores], dim=1)  # [K, K+1]

            # Build gold mask: [K, K+1] where gold_mask[i, j+1] = True if span j is gold antecedent of span i
            top_span_tuples = list(zip(top_starts.tolist(), top_ends.tolist()))
            gold_mask = torch.zeros(K, K + 1, device=device, dtype=torch.bool)

            for i in range(1, K):
                span_i_key = top_span_tuples[i]
                cluster_i = gold_span_to_cluster.get(span_i_key)
                if cluster_i is not None:
                    for j in range(i):
                        if gold_span_to_cluster.get(top_span_tuples[j]) == cluster_i:
                            gold_mask[i, j + 1] = True  # +1 for epsilon offset

                if not gold_mask[i].any():
                    gold_mask[i, 0] = True  # epsilon is correct

            # Skip span 0 (no antecedents) — set its gold to epsilon so loss = 0
            gold_mask[0, 0] = True

            # MLL loss: -log sum_{y in GOLD} P(y|i) for each span i
            log_norm = torch.logsumexp(all_scores, dim=1)  # [K]
            # Mask non-gold scores to -inf before logsumexp
            gold_scores = all_scores.masked_fill(~gold_mask, float("-inf"))
            log_gold_sum = torch.logsumexp(gold_scores, dim=1)  # [K]

            example_loss = -(log_gold_sum - log_norm).mean()
            total_loss = total_loss + example_loss

        total_loss = total_loss / max(B, 1)
        return {"loss": total_loss, "logits": None}


# ---------------------------------------------------------------------------
# Top-level multi-task model
# ---------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    """
    Shared MiniLM encoder + 4 task-specific heads.

    Forward pass takes a `task` parameter to select which head fires.
    Single-task-per-forward-pass design for task-stratified batching.
    """

    TASKS = ("domain", "ner", "parsing", "coref")

    def __init__(self, freeze_layers: int = 3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        # Freeze layers 0 through freeze_layers-1
        if freeze_layers > 0:
            self.encoder.encoder.layer[:freeze_layers].requires_grad_(False)

        # Task heads
        self.domain_head = DomainClassificationHead()
        self.ner_head = NERHead()
        self.parsing_head = DependencyParsingHead()
        self.coref_head = CoreferenceHead()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        ner_labels: torch.Tensor | None = None,
        parsing_arcs: torch.Tensor | None = None,
        domain_class: torch.Tensor | None = None,
        coref_clusters: list | None = None,
        seq_lens: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            input_ids: [B, T]
            attention_mask: [B, T]
            task: one of "domain", "ner", "parsing", "coref"
            **task-specific label tensors
        Returns:
            {"loss": tensor, "logits": tensor or None}
        """
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoder_output.last_hidden_state  # [B, T, H]

        if task == "domain":
            cls_emb = hidden[:, 0, :]  # [B, H]
            return self.domain_head(cls_emb, domain_class)

        elif task == "ner":
            return self.ner_head(hidden, ner_labels)

        elif task == "parsing":
            return self.parsing_head(hidden, parsing_arcs)

        elif task == "coref":
            return self.coref_head(hidden, attention_mask, seq_lens, coref_clusters)

        else:
            raise ValueError(f"Unknown task: {task}. Expected one of {self.TASKS}")
