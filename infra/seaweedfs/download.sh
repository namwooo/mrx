#!/usr/bin/env bash
# Download SeaweedFS release files into dist/ (run on a machine with internet access),
# then carry dist/ into the closed network together with wheels_pool.
#
# Usage:
#   ./download.sh                          # default platforms below
#   ./download.sh linux_amd64 darwin_amd64 # pick platforms
#
# Platforms: linux_amd64, linux_arm64, darwin_arm64 (Apple Silicon Mac), darwin_amd64 (Intel Mac)
set -euo pipefail

VERSION="4.48"   # change here to upgrade; this file records which version went into the closed network
BASE_URL="https://github.com/seaweedfs/seaweedfs/releases/download/${VERSION}"
DEFAULT_PLATFORMS=(linux_amd64 darwin_arm64)

DIST_DIR="$(cd "$(dirname "$0")" && pwd)/dist/${VERSION}"
mkdir -p "$DIST_DIR"

md5_of() {
    # Linux has md5sum, macOS has md5
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | awk '{print $1}'
    else
        md5 -q "$1"
    fi
}

platforms=("$@")
[ ${#platforms[@]} -eq 0 ] && platforms=("${DEFAULT_PLATFORMS[@]}")

for platform in "${platforms[@]}"; do
    file="${platform}.tar.gz"
    target="${DIST_DIR}/${file}"

    curl -fsSL -o "${target}.md5" "${BASE_URL}/${file}.md5"
    expected="$(awk '{print $1}' "${target}.md5")"

    if [ -f "$target" ] && [ "$(md5_of "$target")" = "$expected" ]; then
        echo "ok      ${file} (already downloaded)"
        continue
    fi

    echo "download ${file} ..."
    curl -fL --progress-bar -o "${target}.part" "${BASE_URL}/${file}"

    actual="$(md5_of "${target}.part")"
    if [ "$actual" != "$expected" ]; then
        rm -f "${target}.part"
        echo "error   ${file}: md5 mismatch (expected ${expected}, got ${actual})" >&2
        exit 1
    fi
    mv "${target}.part" "$target"
    echo "ok      ${file} (md5 ${actual})"
done

echo
echo "files in ${DIST_DIR}:"
ls -lh "$DIST_DIR"