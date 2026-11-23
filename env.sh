# Usage: source /path/to/Bipedal_AMP/env.sh
BIPEDAL_AMP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
source "$BIPEDAL_AMP_ROOT/.venv/bin/activate" || return 1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
export ISAACLAB_PATH="${ISAACLAB_PATH:-$BIPEDAL_AMP_ROOT/../IsaacLab}"
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi
export MUJOCO_GL=egl
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BIPEDAL_AMP_ROOT/../.cache/uv}"
export UV_HTTP_TIMEOUT=600
export UV_LINK_MODE=hardlink
