#!/bin/bash
# Build the MIME usd-gl Docker image with MADDENING bundled.
#
# Usage (from MIME repo root):
#   ./docker/build.sh
#
# Requires: ../MADDENING directory exists alongside MIME.
# The script copies MADDENING source into a temporary build context.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MADDENING_DIR="$(cd "$MIME_DIR/../MADDENING" && pwd)"

if [ ! -d "$MADDENING_DIR/src/maddening" ]; then
    echo "ERROR: MADDENING not found at $MADDENING_DIR"
    echo "Expected directory structure:"
    echo "  MSF/"
    echo "    MADDENING/"
    echo "    MIME/"
    exit 1
fi

# Create temporary build context with both repos
BUILD_CTX=$(mktemp -d)
trap "rm -rf $BUILD_CTX" EXIT

echo "Building MIME usd-gl Docker image..."
echo "  MIME:      $MIME_DIR"
echo "  MADDENING: $MADDENING_DIR"
echo "  Context:   $BUILD_CTX"

# Copy MIME source
cp "$MIME_DIR/pyproject.toml" "$BUILD_CTX/"
cp "$MIME_DIR/README.md" "$BUILD_CTX/"
cp -r "$MIME_DIR/src" "$BUILD_CTX/src"
cp -r "$MIME_DIR/experiments" "$BUILD_CTX/experiments"
cp "$MIME_DIR/docker/Dockerfile.usd-gl" "$BUILD_CTX/Dockerfile"

# Copy MADDENING source
mkdir -p "$BUILD_CTX/maddening"
cp "$MADDENING_DIR/pyproject.toml" "$BUILD_CTX/maddening/"
cp "$MADDENING_DIR/README.md" "$BUILD_CTX/maddening/"
cp -r "$MADDENING_DIR/src" "$BUILD_CTX/maddening/src"

echo "Build context prepared ($(du -sh $BUILD_CTX | cut -f1))"

docker build \
    -t ghcr.io/microrobotics-simulation-framework/mime:usd-gl \
    -f "$BUILD_CTX/Dockerfile" \
    "$BUILD_CTX"

echo ""
echo "Image built: ghcr.io/microrobotics-simulation-framework/mime:usd-gl"
echo "Push with:   docker push ghcr.io/microrobotics-simulation-framework/mime:usd-gl"
