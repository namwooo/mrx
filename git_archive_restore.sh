#!/bin/bash

# Usage: ./git-archive-restore.sh [commit-hash_or_branch] [restore_directory(optional)]
TARGET_COMMIT=$1
RESTORE_DIR=${2:-"."} # Default directory is current directory (.)

# Check if target commit is provided
if [ -z "$TARGET_COMMIT" ]; then
    echo "❌ Error: Please specify the target commit to restore."
    echo "Usage: $0 <commit-hash/branch> [restore-directory]"
    exit 1
fi

# Define path where archive files are stored
ARCHIVE_DIR="archives/${TARGET_COMMIT}"
PREFIX_PATH="${ARCHIVE_DIR}/${TARGET_COMMIT}.tar.gz."

# Check if the split archive files actually exist
if [ ! -f "${PREFIX_PATH}000" ]; then
    echo "❌ Error: Cannot find archive files at [${PREFIX_PATH}000]"
    echo "Please check if 'archives/${TARGET_COMMIT}' directory exists."
    exit 1
fi

# Create target restore directory if it doesn't exist
mkdir -p "$RESTORE_DIR"

echo "🔄 Restoring source code from [$ARCHIVE_DIR]..."
echo "📂 Extracting to: $RESTORE_DIR"

# Concatenate split files and extract tar archive directly into target directory
cat "${PREFIX_PATH}"* | tar zxvf - -C "$RESTORE_DIR"

echo "🎉 Restore completed successfully!"
