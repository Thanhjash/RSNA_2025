#!/bin/bash
# Run Phase 0 training notebook in Docker

set -e

NOTEBOOK="notebooks/phase0_pretrain_kaggle.ipynb"
OUTPUT_DIR="$(pwd)/notebook_output"

echo "🚀 Running Phase 0 notebook in Docker..."
echo "Notebook: $NOTEBOOK"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run in Docker with GPU
docker run --rm --gpus all \
  -v "$(pwd)":/workspace/rsna \
  -w /workspace/rsna \
  rsna-minkowski:final \
  jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=36000 \
  --output-dir="$OUTPUT_DIR" \
  "$NOTEBOOK"

echo ""
echo "✅ Notebook execution complete!"
echo "Output: $OUTPUT_DIR/$(basename $NOTEBOOK)"
