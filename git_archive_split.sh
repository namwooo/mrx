#!/bin/bash

TARGET_COMMIT=$1
PART_SIZE="10m"

if [ -z "$TARGET_COMMIT" ]; then
    echo "❌ Error: Please specify a target (e.g., HEAD, main, or a commit hash)."
    echo "Usage: $0 <commit-hash/branch> [output-name]"
    exit 1
fi

OUTPUT_DIR="archives/${TARGET_COMMIT}"
OUTPUT_PATH="${OUTPUT_DIR}/${TARGET_COMMIT}"
mkdir -p "$OUTPUT_DIR"

echo "📦 Archiving source code at commit [$TARGET_COMMIT]..."

# Archive the specific commit, compress with gzip, and split into 10MB parts
# -d uses numeric suffixes (00, 01, 02...) instead of alphabetical ones (aa, ab...)
git archive --format=tar.gz "$TARGET_COMMIT" | split -b "$PART_SIZE" -d - "${OUTPUT_PATH}.tar.gz."

echo "🎉 Split archiving completed successfully!"
echo "👉 Generated files:"
ls -lh ${OUTPUT_PATH}.tar.gz.*