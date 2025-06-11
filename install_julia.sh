#!/usr/bin/env bash
# install_julia.sh - Script to install Julia 1.10.2 in /opt/julia
set -euo pipefail

JULIA_VERSION="1.10.2"
JULIA_DIR="/opt/julia"
JULIA_SYMLINK="/usr/local/bin/julia"

if command -v julia >/dev/null 2>&1; then
    echo "Julia is already installed: $(julia --version)"
    exit 0
fi

ARCH="$(uname -m)"
OS="linux"
TAR_NAME="julia-${JULIA_VERSION}-${ARCH}-${OS}gnu.tar.gz"
DOWNLOAD_URL="https://julialang-s3.julialang.org/bin/${OS}/${ARCH}/${JULIA_VERSION%.*}/${TAR_NAME}"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cd "$TMP_DIR"

# Download and extract Julia
wget -q "$DOWNLOAD_URL" -O julia.tar.gz
mkdir -p "$JULIA_DIR"
tar -xzf julia.tar.gz -C "$TMP_DIR"

# Move to /opt and create symlink
mv "julia-${JULIA_VERSION}" "$JULIA_DIR"
ln -sf "$JULIA_DIR/bin/julia" "$JULIA_SYMLINK"

julia --version

echo "Julia ${JULIA_VERSION} installed successfully."
