#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_gr1_assets.sh — Fourier GR1T1 MuJoCo mesh letöltő
#
# Sparse-clone-nal csak a szükséges STL fájlokat tölti le az FFTAI repo-ból.
# Nem kell az egész 18MB-os mesh könyvtár a git repo-ban.
#
# Használat:
#   bash setup_gr1_assets.sh
#
# Forrás: FFTAI/Wiki-GRx-Models (GPL-3.0)
#         https://github.com/FFTAI/Wiki-GRx-Models
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DEST="mujoco_menagerie/fourier_gr1t1/assets"
REPO="https://github.com/FFTAI/Wiki-GRx-Models.git"
MESH_PATH="GRX/GR1/gr1t1/meshes/gr1t1"
TMP_DIR="$(mktemp -d)"

echo "→ GR1T1 mesh letöltés: FFTAI/Wiki-GRx-Models"

# Ha már megvan, kihagyjuk
if [ -d "$DEST" ] && [ "$(ls -A "$DEST"/*.STL 2>/dev/null | wc -l)" -ge 35 ]; then
    echo "✅ Már megvan ($DEST) — kihagyva."
    exit 0
fi

mkdir -p "$DEST"

echo "→ Sparse clone: csak $MESH_PATH ..."
cd "$TMP_DIR"
git init -q
git remote add origin "$REPO"
git sparse-checkout init --cone
git sparse-checkout set "$MESH_PATH"
git fetch --depth=1 origin main -q
git checkout FETCH_HEAD -q 2>/dev/null || git checkout main -q

cd - > /dev/null
cp "$TMP_DIR/$MESH_PATH"/*.STL "$DEST/"
rm -rf "$TMP_DIR"

COUNT=$(ls "$DEST"/*.STL 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Kész: $COUNT STL fájl → $DEST"
echo "   A scene betölthető: src/envs/assets/scene_manip_sandbox_gr1_v1.xml"
