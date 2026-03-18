#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# Forward --domains flag if provided
ARGS="$@"

echo "============================================================"
echo "  sinciput - Full Data Build Pipeline"
echo "============================================================"
echo ""

echo "[1/5] Generating raw text blocks..."
python build/generate_texts.py $ARGS
echo ""

echo "[2/5] Extracting entities & coreferences..."
python build/extract_entities.py $ARGS
echo ""

echo "[3/5] Tokenizing with WordPiece..."
python build/tokenize_blocks.py $ARGS
echo ""

echo "[4/5] Mapping NER & coreference labels..."
python build/build_labels.py $ARGS
echo ""

echo "[5/5] Parsing dependencies (spaCy)..."
python build/parse_deps.py $ARGS
echo ""

echo "============================================================"
echo "  Build pipeline complete!"
echo "============================================================"
