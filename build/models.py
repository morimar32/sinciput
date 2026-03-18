"""
Pydantic models for the data generation pipeline.

Pipeline stages:
1. generate_texts.py  → RawTextBlock
2. extract_entities.py → ExtractedBlock (adds entities + coref)
3. tokenize.py        → TokenizedBlock (adds tokens)
4. build_labels.py    → TrainingExample (final format with all labels)
"""

from pydantic import BaseModel, Field


# ============================================================================
# Stage 1: Raw Text Generation
# ============================================================================

class RawTextBlock(BaseModel):
    """Output from generate_texts.py"""
    id: str
    domain: str
    domain_index: int
    block_index: int
    block_type: str  # single, contextual, long_distance, adversarial, noisy
    raw_text: str


# ============================================================================
# Stage 2: Entity & Coreference Extraction (LLM)
# ============================================================================

class EntityMention(BaseModel):
    """A single entity mention extracted by the LLM."""
    text: str = Field(description="Exact text span as it appears in the passage")
    entity_type: str = Field(description="Entity type: PERSON, ORG, LOC, DATE, QUANTITY, EVENT, PRODUCT, OTHER")


class CoreferenceLink(BaseModel):
    """A coreference link between an antecedent and a referring expression."""
    antecedent: str = Field(description="The full noun phrase being referred to")
    referent: str = Field(description="The pronoun or shorter reference (e.g., 'he', 'it', 'the company')")


class ExtractionResult(BaseModel):
    """Structured output schema for LLM extraction."""
    entities: list[EntityMention] = Field(
        default_factory=list,
        description="All named entities found in the text"
    )
    coreferences: list[CoreferenceLink] = Field(
        default_factory=list,
        description="Pronoun/reference to antecedent links"
    )


class ExtractedBlock(RawTextBlock):
    """RawTextBlock enriched with extracted entities and coreferences."""
    entities: list[EntityMention] = Field(default_factory=list)
    coreferences: list[CoreferenceLink] = Field(default_factory=list)


# ============================================================================
# Stage 3: Tokenization (Deterministic)
# ============================================================================

class TokenizedBlock(ExtractedBlock):
    """ExtractedBlock with WordPiece tokenization added."""
    tokens: list[str] = Field(
        description="WordPiece tokens including [CLS] and [SEP]"
    )
    # Maps character offsets to token indices for later alignment
    char_to_token: list[int | None] = Field(
        default_factory=list,
        description="For each character position, the token index it belongs to (None for whitespace)"
    )


# ============================================================================
# Stage 4: Index Mapping (NER labels + Coref clusters)
# ============================================================================

class SpanBounds(BaseModel):
    """Token-level span boundaries for coreference."""
    start: int = Field(description="Start token index (inclusive)")
    end: int = Field(description="End token index (inclusive)")
    text: str = Field(description="Original text for debugging")


class CoreferenceCluster(BaseModel):
    """A cluster of coreferent mentions."""
    mentions: list[SpanBounds]


class LabeledBlock(BaseModel):
    """Output of Step 4: TokenizedBlock with NER labels and coref clusters mapped to token indices."""
    id: str
    raw_text: str
    domain: str
    domain_index: int
    block_type: str

    # Tokenization
    tokens: list[str]

    # Task 1: Domain Classification (targets [CLS] token)
    domain_class: int

    # Task 2: NER (IOB2 labels, -100 for special/subword tokens)
    ner_labels: list[int]

    # Task 4: Coreference Resolution (span boundaries)
    coref_clusters: list[CoreferenceCluster] = Field(default_factory=list)


# ============================================================================
# Stage 5: Final Training Example (adds dependency parsing)
# ============================================================================

class TrainingExample(BaseModel):
    """
    Final training example matching the JSON schema in PROJ.md.

    All indices are token-level (post-WordPiece).
    Special tokens ([CLS], [SEP]) and subword continuations (##) use -100.
    """
    id: str
    raw_text: str
    domain: str
    block_type: str

    # Tokenization
    tokens: list[str]

    # Task 1: Domain Classification (targets [CLS] token)
    domain_class: int

    # Task 2: NER (IOB2 labels, -100 for special/subword tokens)
    ner_labels: list[int]

    # Task 3: Dependency Parsing (head indices, -100 for special/subword tokens)
    parsing_arcs: list[int]

    # Task 4: Coreference Resolution (span boundaries)
    coref_clusters: list[CoreferenceCluster] = Field(default_factory=list)


# ============================================================================
# Label Mappings
# ============================================================================

# NER label vocabulary (IOB2 format)
# Index 0 = O (outside), then B-X, I-X pairs for each entity type
NER_LABELS = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-DATE", "I-DATE",
    "B-QUANTITY", "I-QUANTITY",
    "B-EVENT", "I-EVENT",
    "B-PRODUCT", "I-PRODUCT",
    "B-OTHER", "I-OTHER",
]

NER_LABEL_TO_ID = {label: i for i, label in enumerate(NER_LABELS)}
NER_ID_TO_LABEL = {i: label for i, label in enumerate(NER_LABELS)}

# Entity type to NER label prefix mapping
ENTITY_TYPE_TO_NER = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "DATE": "DATE",
    "TIME": "DATE",
    "QUANTITY": "QUANTITY",
    "NUMBER": "QUANTITY",
    "MONEY": "QUANTITY",
    "PERCENT": "QUANTITY",
    "EVENT": "EVENT",
    "PRODUCT": "PRODUCT",
    "OTHER": "OTHER",
}

IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss ignore index
