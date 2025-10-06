#!/bin/bash
# Build MinkowskiEngine wheel for Kaggle
# Usage: bash build_minkowski_wheel.sh

set -e

IMAGE_NAME="minkowski-wheel-builder:cuda121"
OUTPUT_DIR="$(pwd)/wheels"

echo "🔨 Building MinkowskiEngine wheel..."

# Build Docker image
docker build -f source/MinkowskiEngine/docker/Dockerfile.wheel -t "$IMAGE_NAME" .

# Extract wheel
echo "📦 Extracting wheel..."
mkdir -p "$OUTPUT_DIR"
CONTAINER_ID=$(docker create "$IMAGE_NAME")
docker cp "$CONTAINER_ID:/workspace/MinkowskiEngine/dist/" "$OUTPUT_DIR/"
docker rm "$CONTAINER_ID"

# Show result
echo ""
echo "✅ Done! Wheel: $OUTPUT_DIR/dist/*.whl"
echo ""
echo "Upload to Kaggle dataset: minkowski-engine-wheel-cuda121-torch240"
ls -lh "$OUTPUT_DIR"/dist/*.whl
