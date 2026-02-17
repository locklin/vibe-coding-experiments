#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$SCRIPT_DIR/deps"

echo "=== Installing system dependencies ==="
apt-get update -qq
apt-get install -y -qq \
  libarmadillo-dev liblapack-dev libopenblas-dev libensmallen-dev \
  build-essential g++ cmake wget curl pkg-config \
  2>&1 | tail -5

echo "=== Installing mlpack headers ==="
mkdir -p "$DEPS_DIR"
if [ ! -d "$DEPS_DIR/mlpack-4.4.0" ]; then
  cd "$DEPS_DIR"
  wget -q https://www.mlpack.org/files/mlpack-4.4.0.tar.gz
  tar xzf mlpack-4.4.0.tar.gz
  rm -f mlpack-4.4.0.tar.gz
  echo "mlpack 4.4.0 headers installed to $DEPS_DIR/mlpack-4.4.0"
else
  echo "mlpack headers already present"
fi

echo "=== Installing J ==="
if [ ! -f /usr/local/bin/jconsole ]; then
  cd /tmp
  J_URL="https://www.jsoftware.com/download/j9.6/install/j9.6-linux-x86_64.tar.gz"
  wget -q "$J_URL" -O j.tar.gz || {
    echo "Trying alternative J URL..."
    J_URL="https://www.jsoftware.com/download/j9.5/install/j9.5-linux-x86_64.tar.gz"
    wget -q "$J_URL" -O j.tar.gz
  }
  tar xzf j.tar.gz
  J_DIR=$(ls -d j9* 2>/dev/null | head -1)
  if [ -n "$J_DIR" ]; then
    cp -r "$J_DIR"/* /usr/local/ 2>/dev/null || true
    # Find jconsole and ensure it's accessible
    find "$J_DIR" -name jconsole -type f -exec cp {} /usr/local/bin/ \; 2>/dev/null || true
    find "$J_DIR" -name libj.so -type f -exec cp {} /usr/local/lib/ \; 2>/dev/null || true
    # Copy the full J tree for profile/stdlib access
    mkdir -p /usr/local/share/j
    cp -r "$J_DIR"/* /usr/local/share/j/ 2>/dev/null || true
  fi
  rm -rf j.tar.gz "$J_DIR"
  ldconfig 2>/dev/null || true
  echo "J installed"
else
  echo "J already installed"
fi

# Verify
echo ""
echo "=== Verification ==="
echo -n "g++: "; g++ --version | head -1
echo -n "armadillo: "; pkg-config --modversion armadillo 2>/dev/null || echo "installed (no pkg-config)"
echo -n "mlpack headers: "; ls "$DEPS_DIR"/mlpack-*/src/mlpack/mlpack.hpp 2>/dev/null && echo "OK" || echo "MISSING"
echo -n "jconsole: "; which jconsole 2>/dev/null && echo "OK" || echo "MISSING"
echo ""
echo "=== Done ==="
