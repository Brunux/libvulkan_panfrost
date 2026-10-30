#!/usr/bin/env bash
set -euo pipefail

# Reproducible build script for Mesa PanVK Valhall v9 (Job Manager over Linux DRM Panfrost)
# Target Hardware: MediaTek MT8192 (ARM Mali-G57 MC5, GPU Product ID 0x9093)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESA_COMMIT="5a07217f034b3e50d8c7c7794f97a2df1742613b"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/build-work}"
MESA_DIR="${WORK_DIR}/mesa"
BUILD_DIR="${MESA_DIR}/build-panvk-v9"
STAGE_DIR="${WORK_DIR}/pkg-stage"
DIST_DIR="${REPO_ROOT}/dist"

echo "==> Preparing Mesa source at commit ${MESA_COMMIT}..."
mkdir -p "${WORK_DIR}" "${DIST_DIR}"
if [ ! -d "${MESA_DIR}/.git" ]; then
    git clone https://gitlab.freedesktop.org/mesa/mesa.git "${MESA_DIR}"
fi

cd "${MESA_DIR}"
git fetch origin "${MESA_COMMIT}" || true
git checkout -f "${MESA_COMMIT}"
git clean -fd

echo "==> Applying panvk-v9-valhall-jm-drm-mt8192 patch..."
git apply "${REPO_ROOT}/patches/0001-panvk-v9-valhall-jm-drm-mt8192.patch"

echo "==> Configuring Meson for PanVK (Valhall v9 JM over DRM Panfrost)..."
meson setup "${BUILD_DIR}" \
    -Dprefix=/usr/local \
    -Dbuildtype=release \
    -Dgallium-drivers= \
    -Dvulkan-drivers=panfrost \
    -Dpanfrost-kmds=panfrost \
    -Dplatforms=wayland,x11 \
    -Dopengl=false \
    -Dglx=disabled

echo "==> Building libvulkan_panfrost.so..."
ninja -C "${BUILD_DIR}"

echo "==> Staging Debian package..."
rm -rf "${STAGE_DIR}"
DESTDIR="${STAGE_DIR}" ninja -C "${BUILD_DIR}" install

mkdir -p "${STAGE_DIR}/DEBIAN" "${STAGE_DIR}/usr/local/bin"
cp "${REPO_ROOT}/bin/llama-vulkan" "${STAGE_DIR}/usr/local/bin/llama-vulkan"
chmod 755 "${STAGE_DIR}/usr/local/bin/llama-vulkan"

cat > "${STAGE_DIR}/DEBIAN/control" << 'EOF'
Package: mesa-vulkan-panfrost-v9
Version: 26.1.0-mt8192-lazyos1
Section: libs
Priority: optional
Architecture: arm64
Depends: libc6, libdrm2, libexpat1, libgcc-s1, libstdc++6, libvulkan1, zlib1g, libzstd1
Maintainer: Brunux <dev@lazyos.local>
Description: Mesa PanVK Vulkan Compute Driver for ARM Mali-G57 MC5 (Valhall v9 JM DRM)
 Custom build of Mesa's PanVK Vulkan driver with Valhall v9 Job Manager (JM)
 support over mainline Linux DRM Panfrost (/dev/dri/renderD128) for MediaTek
 MT8192 (Mali-G57 MC5, Product ID 0x9093).
 Includes Valhall v9 native FP16, 5-core compute task partitioning, workgroup
 merging, adaptive compute job chaining, and device-shared TLS/WLS scratch
 buffers optimized for llama.cpp / ggml-vulkan inference on 4 GB Chromebooks.
 Installs into /usr/local to preserve system OpenGL ES (panfrost_dri.so).
EOF

DEB_OUT="${DIST_DIR}/mesa-vulkan-panfrost-v9_26.1.0-mt8192-lazyos1_arm64.deb"
dpkg-deb --build --root-owner-group "${STAGE_DIR}" "${DEB_OUT}"
echo "==> Done! Package written to: ${DEB_OUT}"
