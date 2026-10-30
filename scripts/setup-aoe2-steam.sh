#!/usr/bin/env bash
set -euo pipefail

# Automated Setup for Age of Empires II (2013 HD & 2019 DE) from Steam
# Target: MediaTek MT8192 (ARM Mali-G57 MC5, 4 GB RAM, LazyOS arm64 + PanVK v9)

GAMES_DIR="${HOME}/Games"
TOOLS_DIR="${GAMES_DIR}/tools"
WINE_VER="9.22"
DXVK_VER="2.4.1"
DEPOT_VER="2.7.4"
MESA_BUILD_DIR="/home/brunux/Dev/mt8192-vulkan/work/mesa/build-test"
STAGE_DIR="/home/brunux/Dev/mt8192-vulkan/work/pkg-stage"
DEB_OUT="/home/brunux/Dev/mt8192-vulkan/dist/mesa-vulkan-panfrost-v9_26.1.0-mt8192-lazyos1_arm64.deb"

github_latest_asset_url() {
    local repo="$1"
    local pattern="$2"
    python3 - "${repo}" "${pattern}" << 'PYEOF'
import json, re, sys, urllib.request
repo, pattern = sys.argv[1], sys.argv[2]
req = urllib.request.Request(
    f"https://api.github.com/repos/{repo}/releases",
    headers={"User-Agent": "mt8192-vulkan-setup"}
)
with urllib.request.urlopen(req, timeout=30) as resp:
    releases = json.loads(resp.read().decode("utf-8"))
for rel in releases:
    for asset in rel.get("assets", []):
        url = asset.get("browser_download_url", "")
        if re.search(pattern, asset.get("name", "")):
            print(url)
            sys.exit(0)
sys.exit(1)
PYEOF
}

echo "==> [1/5] Checking box64 and X11/XCB Vulkan WSI headers..."
if ! command -v box64 >/dev/null 2>&1 || ! command -v 7z >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y box64 unzip p7zip-full curl xz-utils \
        libx11-xcb-dev libxcb-dri3-dev libxcb-present-dev
fi

if ! vulkaninfo --summary 2>/dev/null | grep -q "VK_KHR_xcb_surface"; then
    if [ -d "${MESA_BUILD_DIR}" ]; then
        echo "==> [2/5] Rebuilding libvulkan_panfrost.so with Wayland + X11 WSI (for Wine/DXVK)..."
        meson configure "${MESA_BUILD_DIR}" -Dplatforms=wayland,x11
        ninja -C "${MESA_BUILD_DIR}"
        DESTDIR="${STAGE_DIR}" ninja -C "${MESA_BUILD_DIR}" install
        dpkg-deb --build --root-owner-group "${STAGE_DIR}" "${DEB_OUT}"
        sudo dpkg -i "${DEB_OUT}"
    fi
else
    echo "==> [2/5] libvulkan_panfrost.so already has Wayland + X11 WSI enabled. Skipping rebuild."
fi

mkdir -p "${TOOLS_DIR}/depotdownloader" "${TOOLS_DIR}/gbe" "${GAMES_DIR}/AoE2HD" "${GAMES_DIR}/AoE2DE"

echo "==> [3/5] Checking native arm64 Steam DepotDownloader v${DEPOT_VER} & lightweight SteamAPI..."
if [ ! -x "${TOOLS_DIR}/depotdownloader/DepotDownloader" ]; then
    curl -fL "https://github.com/SteamRE/DepotDownloader/releases/download/DepotDownloader_${DEPOT_VER}/DepotDownloader-linux-arm64.zip" \
        -o /tmp/DepotDownloader-linux-arm64.zip
    unzip -o /tmp/DepotDownloader-linux-arm64.zip -d "${TOOLS_DIR}/depotdownloader"
    chmod +x "${TOOLS_DIR}/depotdownloader/DepotDownloader"
    rm -f /tmp/DepotDownloader-linux-arm64.zip
fi

if [ -z "$(find "${TOOLS_DIR}/gbe" -name "steam_api.dll" 2>/dev/null)" ]; then
    echo "    Fetching latest Goldberg SteamAPI release URL from GitHub..."
    GBE_URL="$(github_latest_asset_url "Detanup01/gbe_fork" "emu-win-release.*\\.7z$")"
    echo "    Downloading: ${GBE_URL}"
    curl -fL "${GBE_URL}" -o /tmp/gbe.7z
    7z x -y /tmp/gbe.7z -o"${TOOLS_DIR}/gbe" >/dev/null
    rm -f /tmp/gbe.7z
fi

echo "==> [4/5] Checking Wine ${WINE_VER} (amd64-wow64) and DXVK v${DXVK_VER}..."
if [ ! -d "${TOOLS_DIR}/wine-${WINE_VER}-staging-tkg-amd64-wow64" ]; then
    WINE_URL="https://github.com/Kron4ek/Wine-Builds/releases/download/${WINE_VER}/wine-${WINE_VER}-staging-tkg-amd64-wow64.tar.xz"
    echo "    Downloading: ${WINE_URL}"
    curl -fL "${WINE_URL}" -o /tmp/wine-wow64.tar.xz
    tar -xJf /tmp/wine-wow64.tar.xz -C "${TOOLS_DIR}"
    rm -f /tmp/wine-wow64.tar.xz
fi
ln -sfn "${TOOLS_DIR}/wine-${WINE_VER}-staging-tkg-amd64-wow64" "${TOOLS_DIR}/wine"
rm -rf "${TOOLS_DIR}"/wine-11.* 2>/dev/null || true

if [ ! -e "${TOOLS_DIR}/wine/bin/wine64" ]; then
    ln -sf wine "${TOOLS_DIR}/wine/bin/wine64"
fi

if [ ! -d "${TOOLS_DIR}/dxvk-${DXVK_VER}" ]; then
    curl -fL "https://github.com/doitsujin/dxvk/releases/download/v${DXVK_VER}/dxvk-${DXVK_VER}.tar.gz" \
        -o /tmp/dxvk.tar.gz
    tar -xzf /tmp/dxvk.tar.gz -C "${TOOLS_DIR}"
    rm -f /tmp/dxvk.tar.gz
fi

echo "==> Initializing Wine prefix (~/.wine) and installing DXVK (d3d9/d3d11 -> PanVK)..."
export PATH="${TOOLS_DIR}/wine/bin:${PATH}"
export WINEDEBUG=-all
box64 "${TOOLS_DIR}/wine/bin/wine" wineboot -u

cp -f "${TOOLS_DIR}/dxvk-${DXVK_VER}/x64/"*.dll "${HOME}/.wine/drive_c/windows/system32/"
if [ -d "${HOME}/.wine/drive_c/windows/syswow64" ]; then
    cp -f "${TOOLS_DIR}/dxvk-${DXVK_VER}/x32/"*.dll "${HOME}/.wine/drive_c/windows/syswow64/"
fi

for dll in d3d9 d3d10core d3d11 dxgi; do
    box64 "${TOOLS_DIR}/wine/bin/wine" reg.exe add 'HKCU\Software\Wine\DllOverrides' /v "${dll}" /d native /f
done

echo "==> [5/5] Creating download and launch scripts in ${GAMES_DIR}..."

cat > "${GAMES_DIR}/download-aoe2.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
TOOLS_DIR="${HOME}/Games/tools"

echo "Which Steam edition do you want to download?"
echo "  1) Age of Empires II (2013 HD Edition) [AppID 221380] (~2.2 GB - Recommended for 4 GB RAM)"
echo "  2) Age of Empires II: Definitive Edition [AppID 813780] (~18 GB)"
read -rp "Select [1 or 2]: " choice
read -rp "Enter your Steam username: " steam_user

if [ "${choice}" = "1" ]; then
    GAME_DIR="${HOME}/Games/AoE2HD"
    "${TOOLS_DIR}/depotdownloader/DepotDownloader" \
        -app 221380 -os windows -osarch 32 \
        -username "${steam_user}" \
        -dir "${GAME_DIR}"
    echo "221380" > "${GAME_DIR}/steam_appid.txt"
    if [ -f "${GAME_DIR}/steam_api.dll" ] && [ ! -f "${GAME_DIR}/steam_api.dll.valve" ]; then
        cp "${GAME_DIR}/steam_api.dll" "${GAME_DIR}/steam_api.dll.valve"
    fi
    GBE_DLL32="$(find "${TOOLS_DIR}/gbe" -path "*regular*x32/steam_api.dll" | head -n 1)"
    if [ -n "${GBE_DLL32}" ]; then
        cp -f "${GBE_DLL32}" "${GAME_DIR}/steam_api.dll"
    fi
    echo "==> Download complete! Run: ~/Games/play-aoe2-2013.sh"
elif [ "${choice}" = "2" ]; then
    GAME_DIR="${HOME}/Games/AoE2DE"
    "${TOOLS_DIR}/depotdownloader/DepotDownloader" \
        -app 813780 -os windows -osarch 64 \
        -username "${steam_user}" \
        -dir "${GAME_DIR}"
    echo "813780" > "${GAME_DIR}/steam_appid.txt"
    if [ -f "${GAME_DIR}/steam_api64.dll" ] && [ ! -f "${GAME_DIR}/steam_api64.dll.valve" ]; then
        cp "${GAME_DIR}/steam_api64.dll" "${GAME_DIR}/steam_api64.dll.valve"
    fi
    GBE_DLL64="$(find "${TOOLS_DIR}/gbe" -path "*regular*x64/steam_api64.dll" | head -n 1)"
    if [ -n "${GBE_DLL64}" ]; then
        cp -f "${GBE_DLL64}" "${GAME_DIR}/steam_api64.dll"
    fi
    echo "==> Download complete! Run: ~/Games/play-aoe2-de.sh"
else
    echo "Invalid selection."
    exit 1
fi
EOF
chmod +x "${GAMES_DIR}/download-aoe2.sh"

cat > "${GAMES_DIR}/play-aoe2-2013.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/Games/tools/wine/bin:${PATH}"
export VK_ICD_FILENAMES="/usr/local/share/vulkan/icd.d/panfrost_icd.aarch64.json"
export DXVK_HUD="devinfo,fps"
export BOX64_DYNAREC=1
export BOX64_LOG=0
export WINEDEBUG=-all

cd "${HOME}/Games/AoE2HD"
# Bypass Launcher.exe and run AoK HD.exe directly with NoStartup
exec choom -n 1000 -- box64 "${HOME}/Games/tools/wine/bin/wine" "AoK HD.exe" NoStartup "$@"
EOF
chmod +x "${GAMES_DIR}/play-aoe2-2013.sh"

cat > "${GAMES_DIR}/play-aoe2-de.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/Games/tools/wine/bin:${PATH}"
export VK_ICD_FILENAMES="/usr/local/share/vulkan/icd.d/panfrost_icd.aarch64.json"
export DXVK_HUD="devinfo,fps,memory"
export BOX64_DYNAREC=1
export BOX64_LOG=0
export WINEDEBUG=-all

cd "${HOME}/Games/AoE2DE"
exec choom -n 1000 -- box64 "${HOME}/Games/tools/wine/bin/wine" "AoE2DE_s.exe" -SKIPINTRO "$@"
EOF
chmod +x "${GAMES_DIR}/play-aoe2-de.sh"

echo ""
echo "================================================================="
echo " Setup Complete!"
echo " 1. Download your game from Steam:  ~/Games/download-aoe2.sh"
echo " 2. Launch AoE II (2013 HD):        ~/Games/play-aoe2-2013.sh"
echo " 3. Launch AoE II (2019 DE):        ~/Games/play-aoe2-de.sh"
echo "================================================================="
