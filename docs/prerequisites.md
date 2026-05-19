# Prerequisites

Use this page before running a release package or building from source. The
repository does not include model weights or OpenVINO IR. The release server can
download the public IR on first start.

## Quick Check

Source development requires:

- Python `>=3.12`, matching `pyproject.toml`.
- `uv` for dependency management and script execution. Official installation
  docs: <https://docs.astral.sh/uv/getting-started/installation/>.
- A C++ compiler and CMake when building the native runtime from source.
  Windows source builds use Visual Studio 2022 Build Tools / MSVC.
- Network access to GitHub Releases and Hugging Face, unless IR has already
  been downloaded locally.

Release users need:

- Linux x86_64 or Windows x64.
- The matching release package: `.tar.zst` on Linux, `.zip` on Windows.
- If using `--device GPU`, OpenVINO must be able to see an Intel GPU.

## Install uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

If direct install scripts are blocked, use one of the official alternatives
such as `pipx install uv`, WinGet, Scoop, or standalone binaries.

## Python and OpenVINO

Source installs are managed by `uv`:

```bash
uv sync --extra native --extra server --extra export
```

`openvino`, `openvino-genai`, and `openvino-tokenizers` are installed through
project dependencies. You normally do not install OpenVINO separately. The
current lock file resolves `openvino` to `2026.1.0`, but the effective version
comes from `uv.lock` and platform wheel availability. OpenVINO's official
installation entry point is <https://docs.openvino.ai/install>.

Release packages bundle the runtime dependencies needed by the app. End users
still need the system GPU/NPU drivers for their machine.

## Windows Source Toolchain

Windows source builds require Visual Studio 2022 Build Tools with the C++
toolchain and Windows SDK, plus CMake. Check:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
uv --version
cmake --version
where.exe cl
```

Build the native runtime:

```powershell
uv run python scripts\build_native_codegen.py --backend cmake --config Release
```

The expected output is `native/build/qwen3_tts_ov_genai.dll`. Use the current
source if MSVC reports code-page issues such as `C4819` or `C2001`; the CMake
targets compile with `/utf-8`. The PowerShell UTF-8 environment variables avoid
Python progress/output failures on some non-English Windows consoles.

## Device Selection

The server passes `--device` to OpenVINO:

| Value | Use | Notes |
| --- | --- | --- |
| `GPU` | Recommended production path | Default in release examples. |
| `CPU` | Smoke/fallback | Useful for startup/API checks; realtime performance is not promised. |
| `AUTO` | OpenVINO automatic selection | Experimental for this project; OpenVINO chooses the device. |
| `GPU.0`, etc. | Multi-GPU selection | Use when OpenVINO reports several GPU devices. |

OpenVINO supports CPU, GPU, and NPU devices, and also provides AUTO device
selection. This project's Windows GPU+NPU path keeps `--device GPU` and uses
`--npu-offload ...`; using `NPU` as the main device is not the recommended
production path.

Check devices visible to OpenVINO:

```bash
uv run python - <<'PY'
import openvino as ov
core = ov.Core()
print(core.available_devices)
for name in core.available_devices:
    print(name, core.get_property(name, "FULL_DEVICE_NAME"))
PY
```

For a release package without `uv`, use a local Python environment for this
probe, or start the server and inspect `/health`.

PowerShell alternative:

```powershell
uv run python -c "import openvino as ov; core=ov.Core(); print(core.available_devices); [print(d, core.get_property(d, 'FULL_DEVICE_NAME')) for d in core.available_devices]"
```

## Intel GPU Requirements

This project targets Intel GPUs visible through OpenVINO, including Intel Arc
discrete GPUs, Intel Core Ultra/Arc integrated GPUs, and Xe-family integrated
GPUs. The source of truth is whether `ov.Core().available_devices` contains
`GPU`.

On Linux, OpenVINO GPU inference requires Intel OpenCL/Level Zero runtime
packages. OpenVINO's GPU setup page is:
<https://docs.openvino.ai/nightly/get-started/install-openvino/configurations/configurations-intel-gpu.html>.
Common Ubuntu 22.04/24.04 package names are:

```bash
sudo apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
sudo usermod -a -G render "$LOGNAME"
```

Log out and back in after changing groups. For discrete GPUs, also verify the
kernel driver, `/dev/dri/renderD*`, BIOS/passthrough settings, and container
device mapping if applicable.

On Windows, install the Intel Graphics Driver for your machine. Intel's Arc/Core
Ultra driver page is:
<https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html>.
OEM laptops may require vendor-specific driver guidance.

## Optional NPU Path

NPU is an optional heterogeneous offload path for Windows and supported Intel
platforms:

```powershell
.\qwen3-tts-ov-server.exe --device GPU --npu-offload auto
```

`--npu-offload decoder` requires OpenVINO to see `NPU` and requires the
streaming decoder to compile on NPU. It fails hard if that is not true. Use
`auto` when fallback is desired. See [Windows GPU+NPU path](windows_gpu_npu_zh.md).

## First Compile Cache

On first start, first device change, or first use of a new IR, OpenVINO compiles
graphs and writes a user cache. This can look slow on cold GPU/NPU caches. Later
runs are much faster when the cache hits.

Recommended checks:

- Open `http://127.0.0.1:17860/health` and inspect warmup state.
- Use [OpenVINO compile cache](cache_zh.md) for offline warmup.
- Do not commit OpenVINO caches, `outputs/`, `models/`, or `openvino/`.

## Network and Model Downloads

The release server downloads public IR from Hugging Face by default. For offline
deployment, download `openvino_realtime/` as described in [Release usage](release_zh.md),
then start with:

```bash
./qwen3-tts-ov-server --device GPU --model-root /path/to/openvino_realtime
```

For proxy and cache settings, see Hugging Face Hub environment variables:
<https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables>.

