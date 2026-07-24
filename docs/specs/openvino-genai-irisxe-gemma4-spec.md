# OpenVINO + GenAI Source Build & Gemma-4-12B-IT Inference on Intel Iris Xe — Spec

**Short name:** `openvino-genai-irisxe-gemma4`  
**Date:** 2026-07-21  
**Status:** ✅ GPU inference succeeded — Gemma-4-12B-IT INT4 QAT runs on Iris Xe  
**Author:** Buffy (assistant) with user input from interview rounds  
**Host:** Windows 11, Intel Core i7-1260P, Intel Iris Xe iGPU, 32 GB RAM  
**Target model:** `HarmenWessels/gemma-4-12B-it-qat-int4-ov` (text + image VLM, ~12B params, INT4 QAT, ~6–7 GB on disk)  

---

## 1. Goal

Build from source on the local Windows 11 machine:

1. `openvino` (core runtime) — this repo, `master` branch  
2. `openvino_tokenizers` — sibling repo, `master` branch  
3. `openvino.genai` — sibling repo, `master` branch (`v2026.4.0.0` area)  

Then run a Gemma-4 VLM through `ov::genai::VLMPipeline` on the **Intel Iris Xe iGPU** with text generation. The original 26B-A4B target was too large for the iGPU; the final working target is `HarmenWessels/gemma-4-12B-it-qat-int4-ov` (INT4 QAT, ~12B params), which loads and generates successfully on the iGPU.

---

## 2. Repository Layout (assumed from interview)

| Repo | Local path | Branch | Notes |
|------|------------|--------|-------|
| openvino | `C:\Users\mohds\Documents\GitHub\openvino` | `master` | Current working directory |
| openvino_tokenizers | `C:\Users\mohds\Documents\GitHub\openvino_tokenizers` | `master` | Sibling to openvino |
| openvino.genai | `C:\Users\mohds\Documents\GitHub\openvino.genai` | `master` | Sibling to openvino |
| artifacts / model cache | `C:\Users\mohds\Documents\GitHub\openvino-artifacts` | n/a | Sibling folder containing models, caches, and helper scripts |

> **Assumption to verify:** All three repos exist. If any are missing, the executor must clone them before proceeding.

---

## 3. Environment & Constraints

| Item | Value / Constraint |
|------|--------------------|
| OS | Windows 11 |
| CPU | Intel Core i7-1260P (12C/16T, P-cores + E-cores) |
| iGPU | Intel Iris Xe (integrated, no discrete GPU) |
| GPU memory | 128 MB dedicated + up to ~16 GB shared system RAM |
| RAM | 32 GB |
| Python | 3.14.x at `C:\Python314` |
| Python env | `.venv` inside `C:\Users\mohds\Documents\GitHub\openvino` (must be created/activated) |
| Build tools | **Visual Studio 2026 BuildTools v18.8.0** (MSVC 14.51.36231), CMake, Ninja — installed under `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\` |
| GPU runtime | **Level Zero (ZE)** — verified `ze_loader.dll` present; used successfully for Gemma-4 12B inference. OCL kept as fallback. |
| NPU | Disabled — i7-1260P has no NPU |
| Frontends | Only IR frontend enabled; all framework frontends disabled |
| Meta-plugins | HETERO enabled (user wants future CPU/GPU split exploration); MULTI, AUTO, AUTO_BATCH, PROXY disabled |
| Tests / samples | Disabled to reduce build time |
| Wheel build | **ENABLED** (`-DENABLE_WHEEL=ON`) — evidence: OpenVINO docs state wheels are generated under `<build>/wheels/` only when both `ENABLE_PYTHON` and `ENABLE_WHEEL` are set. |

### Verified prerequisite locations

| Tool | Verified path / version |
|------|-------------------------|
| VS 2026 BuildTools root | `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\` |
| `vcvars64.bat` | `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat` |
| `cl.exe` (x64 host, x64 target) | `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe` |
| `cmake.exe` | `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe` |
| `ninja.exe` | `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe` |
| Standalone CMake | `C:\Program Files\CMake\bin\cmake.exe` |

> **Important:** These tools are **not** in the default `PATH` of the bash shell. Every build session must start by running `vcvars64.bat` (see build steps below).

### GPU runtime selection rationale

The original plan assumed `GPU_RT_TYPE=OCL` with the reasoning that Iris Xe uses OpenCL rather than Level Zero. Verification on this host showed that the Intel GPU driver ships `ze_loader.dll` in `C:\Windows\System32` and in the driver store, so Level Zero is available. OpenVINO's GPU plugin can be built with `-DGPU_RT_TYPE=ZE` (currently documented as experimental but actively maintained). For a 12th Gen Intel iGPU on Windows, Level Zero is the lower-overhead, preferred compute path. The spec therefore switches the build to `ZE`, with OCL documented as a fallback if the ZE build or runtime fails.

### Non-negotiable constraints from user interview

- Do **not** use system Python (`C:\Python314`) for the build/run; use the repo-local `.venv`.
- Do **not** enable NPU-related code.
- Optimize for the actual machine (no unnecessary plugins/frontends).
- If GPU OOMs, **fail and report the exact error**; do not silently fall back to CPU (CPU fallback can be tested later as a separate experiment).
- Create a new sibling folder for artifacts and model cache, with an `AGENTS.md` file documenting its purpose.

---

## 4. Build Configuration Details

### 4.1 openvino (core runtime)

Initialize the build environment first:

```cmd
cd C:\Users\mohds\Documents\GitHub\openvino
.venv\Scripts\activate.bat
"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

Then configure and build:

```cmd
cmake -S . -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DENABLE_INTEL_CPU=ON ^
    -DENABLE_INTEL_GPU=ON ^
    -DGPU_RT_TYPE=ZE ^
    -DENABLE_INTEL_NPU=OFF ^
    -DENABLE_OV_ONNX_FRONTEND=OFF ^
    -DENABLE_OV_PADDLE_FRONTEND=OFF ^
    -DENABLE_OV_TF_FRONTEND=OFF ^
    -DENABLE_OV_TF_LITE_FRONTEND=OFF ^
    -DENABLE_OV_PYTORCH_FRONTEND=OFF ^
    -DENABLE_OV_JAX_FRONTEND=OFF ^
    -DENABLE_OV_IR_FRONTEND=ON ^
    -DENABLE_TESTS=OFF ^
    -DENABLE_SAMPLES=OFF ^
    -DENABLE_PYTHON=ON ^
    -DENABLE_WHEEL=ON ^
    -DENABLE_JS=OFF ^
    -DENABLE_OPENVINO_DEBUG=OFF ^
    -DTHREADING=TBB ^
    -DENABLE_PROXY=OFF ^
    -DENABLE_MULTI=OFF ^
    -DENABLE_AUTO=OFF ^
    -DENABLE_AUTO_BATCH=OFF ^
    -DENABLE_HETERO=ON ^
    -DENABLE_TEMPLATE=OFF
```

Then:

```cmd
cmake --build build --config Release --parallel
pip install build\wheels\openvino-*.whl
```

#### Flag rationale

| Flag | Value | Why |
|------|-------|-----|
| `ENABLE_INTEL_CPU` | ON | CPU inference engine — essential. |
| `ENABLE_INTEL_GPU` | ON | GPU plugin for Iris Xe — uses Level Zero. |
| `GPU_RT_TYPE` | ZE | Modern Intel iGPU path; `ze_loader.dll` verified present. OCL is fallback. |
| `ENABLE_INTEL_NPU` | OFF | i7-1260P has no NPU. |
| `ENABLE_HETERO` | ON | Future CPU/GPU heterogeneous split experiments. |
| `ENABLE_OV_ONNX/PADDLE/TF/TF_LITE/PYTORCH/JAX` | OFF | Framework frontends unnecessary for pre-exported IR. |
| `ENABLE_OV_IR_FRONTEND` | ON | Required — the model is in OpenVINO IR format. |
| `ENABLE_TESTS/SAMPLES` | OFF | Not needed; reduces build time. |
| `ENABLE_PYTHON` / `ENABLE_WHEEL` | ON | Build and package the Python wheel. |
| `THREADING` | TBB | Best CPU threading performance. |

### 4.2 openvino_tokenizers

Actual commands used (from a `cmd.exe` window with `vcvars64.bat` and the openvino `.venv` activated):

```cmd
cd /d C:\Users\mohds\Documents\GitHub\openvino_tokenizers
set OpenVINO_DIR=C:\Users\mohds\Documents\GitHub\openvino\build
rmdir /s /q .py-build-cmake_cache 2>nul
pip install --no-build-isolation --no-deps .
```

Why this works:
- `--no-build-isolation` keeps the MSVC/Ninja environment from `vcvars64.bat` visible and uses the locally installed `py-build-cmake==0.4.3`.
- `--no-deps` skips trying to install `openvino~=2026.4.0.dev` from PyPI; the locally built `openvino` wheel is already in the venv.

Result: `openvino_tokenizers-2026.4.0.0-1-8cfe3eb80f9` installed and imports successfully.

### 4.3 openvino.genai

Actual commands used (from a `cmd.exe` window with `vcvars64.bat` and the openvino `.venv` activated):

```cmd
cd /d C:\Users\mohds\Documents\GitHub\openvino.genai
python -m pip install py-build-cmake==0.5.0
set OpenVINO_DIR=C:\Users\mohds\Documents\GitHub\openvino\build
rmdir /s /q .py-build-cmake_cache 2>nul
pip install --no-build-isolation --no-deps .
```

Notes:
- `openvino.genai` requires `py-build-cmake>=0.5.0`; the tokenizers build used `0.4.3`, so upgrade first.
- `OpenVINO_DIR` points to the **build directory** of the locally built OpenVINO core, where `OpenVINOConfig.cmake` is located.
- `BUILD_TOKENIZERS=OFF` is set in `pyproject.toml`, so it reuses the tokenizers installed above.

Result: `openvino_genai-2026.4.0.0-1-790abd40056` installed and imports successfully.

> **Note:** `openvino.genai` must find the OpenVINO built in step 4.1. This is expected to work because the wheel from step 4.1 is installed into the active `.venv`, making `find_package(OpenVINO)` succeed.

---

## 5. Prerequisites to Verify Before Building

Per the user interview, the executor must verify prerequisites first:

| # | Prerequisite | Status | Details |
|---|--------------|--------|---------|
| 1 | CMake ≥3.26 | ✅ Verified | `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe` |
| 2 | Ninja | ✅ Verified | `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe` |
| 3 | VS C++ build tools / `vcvars64.bat` | ✅ Verified | VS 2026 BuildTools v18.8.0, MSVC 14.51.36231 |
| 4 | Python 3.14.x | ✅ Verified | 3.14.0 |
| 5 | `.venv` in openvino repo | ✅ Verified | Exists |
| 6 | Intel Graphics Driver / OpenCL | ✅ Verified | Driver 32.0.101.7088, `opencl.dll` present |
| 7 | Git for Windows | ✅ Verified | 2.47.1.windows.2 |
| 8 | Sibling repos exist | ✅ Verified | `openvino_tokenizers`, `openvino.genai` |
| 9 | openvino submodules | ✅ Verified | Initialized, on `master` |
| 10 | Disk space | ✅ Verified | ~180 GB free on C:, ~171 GB free on G: |

If any prerequisite is missing, the executor must stop and report the exact missing item rather than proceeding.

### Build session initialization

Before running CMake/Ninja in any new shell, initialize the MSVC environment:

```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

Then confirm the tools are reachable:

```cmd
cl.exe
cmake --version
ninja --version
```

---

## 6. Artifact & Model Cache Folder

Create the sibling folder before first build:

```cmd
mkdir C:\Users\mohds\Documents\GitHub\openvino-artifacts
cd C:\Users\mohds\Documents\GitHub\openvino-artifacts
```

Create `AGENTS.md` with content explaining the folder contains build artifacts, model downloads, and cached compiled blobs.

Actual subfolders:

```
openvino-artifacts/                   # sibling folder, NOT inside openvino repo
├── AGENTS.md
├── scripts/                          # Python helper scripts
│   ├── inference.py                  # text-only inference script
│   ├── benchmark.py                  # CPU/GPU token-speed benchmark script
│   ├── benchmark_scaling.py          # doubles max_new_tokens until story finishes or OOM
│   ├── download_model.py             # HF model downloader
│   ├── monitor_download.py           # download progress monitor
│   ├── server.py                     # FastAPI OpenAI-compatible server
│   └── test_openai_client.py         # Python OpenAI client test
├── logs/                             # runtime logs
│   ├── server.log
│   ├── gpu_inference.log
│   └── ovms.log
├── outputs/                          # generated stories / outputs
│   ├── story_cpu.txt
│   ├── story_cpu_4096.txt
│   ├── story_gpu.txt
│   ├── story_gpu_4096.txt
│   └── story_gpu_8192.txt
├── models/
│   └── gemma-4-12B-it-qat-int4-ov/   # working 12B INT4 QAT model
└── cache/
    ├── ov_gpu_cache_12b_qat/         # GPU compiled-kernel cache
    └── ov_cpu_cache_12b_qat/         # CPU compiled-kernel cache
```

> **Note:** The `openvino-artifacts/` directory is intentionally outside the `openvino` git repo. Scripts and stories live here; only the spec file is committed in the `openvino` repo.

---

## 7. Model Download & Inference

### 7.1 Download

Because `huggingface-cli` is deprecated on this host, use `python -c` with `snapshot_download`:

```cmd
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('HarmenWessels/gemma-4-12B-it-qat-int4-ov', local_dir=r'C:\Users\mohds\Documents\GitHub\openvino-artifacts\models\gemma-4-12B-it-qat-int4-ov')"
```

Install `huggingface_hub` in the `.venv` first if needed.

### 7.2 Text-only inference test

```python
import openvino_genai as ov_genai

model_dir = r"C:\Users\mohds\Documents\GitHub\openvino-artifacts\models\gemma-4-12B-it-qat-int4-ov"
pipe = ov_genai.VLMPipeline(model_dir, "GPU")
result = pipe.generate(
    "Who is the president of the United States?",
    max_new_tokens=128
)
print(result)
```

### 7.3 Image + text inference test (secondary)

```python
import openvino as ov
from PIL import Image
import numpy as np
import openvino_genai as ov_genai

model_dir = r"C:\Users\mohds\Documents\GitHub\openvino-artifacts\models\gemma-4-12B-it-qat-int4-ov"
pipe = ov_genai.VLMPipeline(model_dir, "GPU")

img = Image.open(r"C:\path\to\image.jpg").convert("RGB")
images = [ov.Tensor(np.array(img)[None])]  # NHWC
result = pipe.generate(
    "What is in this image? <ov_genai_image_0>",
    images=images,
    max_new_tokens=128
)
print(result)
```

> **GPU memory note:** The 26B model was ~14.3 GB on disk and OOM'd on Iris Xe. The 12B INT4 QAT model is ~6–7 GB on disk and fits comfortably. First compilation may take several minutes; enable OpenVINO cache (`OPENVINO_CACHE_DIR`) to speed subsequent runs.

---

## 8. Known Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| GPU OOM during first compilation / inference | Medium | Report exact error; do not silently fall back per user directive. |
| Python 3.14 package incompatibilities | Low-Medium | Use only the venv; build pybind11 from OpenVINO's bundled submodule. |
| Build times several hours on i7-1260P | High | Expect 30–120 minutes per repo; run with `--parallel`. |
| Outdated Intel GPU driver crashes | Medium | Verify / update driver before GPU inference. |
| Level Zero runtime not found | Low-Medium | Verified `ze_loader.dll` exists; fallback to OCL if needed. |
| oneDNN GPU build fails with MSVC | Low | Use Release + Ninja; keep frontends disabled. |
| Network/bandwidth for 14.3 GB model | High | Use HuggingFace CLI with resume capability. |

---

## 9. Success Criteria

From the interview, the minimum success is:

1. All three packages build successfully.
2. `openvino` Python wheel installs into the `.venv`.
3. The `VLMPipeline` loads the model on **GPU**.
4. Text generation produces coherent output.

Image input is a stretch goal (secondary). CPU-only fallback is an explicit **non-goal** for this spec; if GPU fails, the executor reports the exact error and stops.

---

## 10. Resolved Decisions

| # | Decision | Resolution |
|---|----------|------------|
| 1 | Repo existence | ✅ Verified — `openvino_tokenizers` and `openvino.genai` exist at sibling paths. |
| 2 | Submodule state | ✅ Verified — openvino submodules initialized, on `master`. |
| 3 | GPU runtime | ✅ Resolved to `GPU_RT_TYPE=ZE`; `ze_loader.dll` verified present. OCL is documented fallback. |
| 4 | Python dev libs | ✅ Verified — `C:\Python314\include\Python.h` exists; `python3.lib` / `python314.lib` present in `C:\Python314\libs`. |
| 5 | HETERO | ✅ Keep enabled; GPU-only is the first target, HETERO is for future CPU/GPU split experiments. |
| 6 | Build output | ✅ Keep each repo's build artifacts in its own `build/` directory. Standard OpenVINO convention; makes `find_package(OpenVINO)` work naturally for `openvino.genai`. |
| 7 | Model cache | ✅ Set `OPENVINO_CACHE_DIR` to `C:\Users\mohds\Documents\GitHub\openvino-artifacts\cache\ov_gpu_cache`. |

---

## 11. Execution Results

| Step | Status | Result |
|------|--------|--------|
| 1. Create `openvino-artifacts/` and `AGENTS.md` | ✅ Done | Folder created at `C:\Users\mohds\Documents\GitHub\openvino-artifacts` |
| 2. Activate `.venv` and install base deps | ✅ Done | `.venv` active; `pip`, `setuptools`, `wheel`, `huggingface_hub` installed |
| 3. Build & install `openvino` core wheel | ✅ Done | `openvino-2026.4.0-22498-cp314-cp314-win_amd64.whl` installed |
| 4. Build & install `openvino_tokenizers` | ✅ Done | `openvino_tokenizers-2026.4.0.0-1-8cfe3eb80f9` installed |
| 5. Build & install `openvino.genai` | ✅ Done | `openvino_genai-2026.4.0.0-1-790abd40056` installed |
| 6. Download Gemma-4-12B-IT INT4 QAT model | ✅ Done | Downloaded to `openvino-artifacts/models/gemma-4-12B-it-qat-int4-ov` |
| 7. Run text-only GPU inference | ✅ Done | `VLMPipeline` executed successfully on GPU |
| 8. Native OVMS VS 2026 compilation | ✅ Done | Compiled OVMS 2026.3.0 natively with MSVC 14.51 against GenAI 2026.4.0.0 |

## 12. Execution Plan (remaining steps)

1. **Benchmark** the Gemma-4-12B-IT INT4 QAT model on GPU and CPU using `openvino-artifacts/scripts/benchmark.py`.
2. **Record** TTFT, TPOT, throughput, and total generate time for each device.
3. **Report** the objective CPU vs GPU comparison.

---

### Build/install commands summary (copy-paste ready)

For each repo, the working incantation was the same: `pip install --no-build-isolation --no-deps .` with the openvino `.venv` activated and `vcvars64.bat` sourced. This avoids PyPI dependency checks and keeps the MSVC environment intact.

| Repo | Key install command | Why it worked |
|------|---------------------|---------------|
| `openvino` | `pip install build\wheels\openvino-*.whl` | Wheel produced by `-DENABLE_WHEEL=ON` |
| `openvino_tokenizers` | `pip install --no-build-isolation --no-deps .` | Bypasses PyPI `openvino~=2026.4.0.dev` requirement |
| `openvino.genai` | `pip install --no-build-isolation --no-deps .` | Same trick; also requires `py-build-cmake==0.5.0` and `OpenVINO_DIR` |

## 13. Architecture Overview

```
openvino (runtime, master)    ← build from source
    ↑ depends on
openvino.genai (master)       ← build from source
    ↑ depends on
openvino_tokenizers           ← build from source
    ↑
ov::genai::VLMPipeline
    ↑ consumes
HarmenWessels/gemma-4-12B-it-qat-int4-ov (HF)
    ├── openvino_language_model.xml/.bin          INT4 QAT   ~6–7 GB
    ├── openvino_text_embeddings_model.xml/.bin     INT8       ~0.5 GB
    ├── openvino_vision_embeddings_model.xml/.bin   INT8       ~0.5 GB
    ├── openvino_tokenizer.xml/.bin
    ├── openvino_detokenizer.xml/.bin
    └── config.json  (model_type: "gemma4_unified")
```

### What the prebuilt pipeline engine (`openvino.pipeline`) adds

The public repos build only the GenAI-native `VLMPipeline`. The proprietary Intel prebuilt pipeline engine adds:

- **Audio input** (`openvino_audio_embeddings_model`) — not in genai-native.
- **Safetensors → IR on-the-fly** — loads raw HF weights directly.
- **YAML-driven modular pipeline** — per-module device assignment.

For this project, we use the GenAI-native `VLMPipeline` with the pre-exported IR model.

---

## 14. Notes from Interview

- Server deployment originally attempted with OVMS v2026.2.1; pivoted to FastAPI after OVMS's bundled `openvino_genai` lacked `gemma4_unified` support.
- User confirmed all three repos are cloned locally at default sibling paths.
- GPU runtime changed from **OCL** to **ZE** after verifying `ze_loader.dll` is present on this Iris Xe iGPU; OCL remains a documented fallback.
- User wants a **step-by-step plan detailed enough for the assistant to execute**.
- User selected "execute as we go, spec is a checkpoint".
- User wants prerequisites verified before any build commands run.
- User wants a new sibling folder for artifacts/model cache with an `AGENTS.md` file.
- User wants `ENABLE_WHEEL=ON` included because the docs require it for wheel generation.
- User noted Python 3.14 is stable; the build should proceed with Python 3.14.

---

## 15. CPU vs GPU Benchmark Results

Benchmark script: `openvino-artifacts/scripts/benchmark.py`
Prompt: `"Who is the president of the United States?"`
`max_new_tokens`: `128`
Measured iterations: `3`
Warmup iterations: `1`

### Benchmark command

```cmd
cd /d C:\Users\mohds\Documents\GitHub\openvino
.venv\Scripts\activate.bat

:: GPU
.venv\Scripts\python.exe openvino-artifacts\scripts\benchmark.py --device GPU --num_iter 3 --max_new_tokens 128

:: CPU
.venv\Scripts\python.exe openvino-artifacts\scripts\benchmark.py --device CPU --num_iter 3 --max_new_tokens 128
```

### Results

| Metric | GPU | CPU |
|--------|-----|-----|
| Load time (first compile) | 56,551 ms | 58,082 ms |
| TTFT (ms) | 2,030 ± 224 | 3,036 ± 627 |
| TPOT (ms/token) | 587.98 ± 470.89 | 647.38 ± 164.19 |
| Throughput (tokens/s) | **1.70 ± 1.36** | **1.54 ± 0.39** |
| Generate duration (ms) | **11,441 ± 3,202** | **17,927 ± 3,514** |

### Interpretation

- **GPU wins overall:** Iris Xe is about **10% faster in throughput** and completes generation in roughly **two-thirds the CPU time**.
- The absolute speed is modest (~1.7 tokens/s) because a 12B model is still very large for an integrated GPU with shared memory.
- Load times are dominated by first-time graph compilation (cache was cold for each device). Subsequent runs will reuse cached kernels.
- Both CPU and GPU produce coherent answers.

### OpenVINO GenAI PerfMetrics reference

The benchmark uses the built-in `PerfMetrics` API:

- `perf_metrics.get_ttft()` — Time To First Token (ms)
- `perf_metrics.get_tpot()` — Time Per Output Token (ms/token)
- `perf_metrics.get_throughput()` — Tokens per second
- `perf_metrics.get_generate_duration()` — Total generation time (ms)
- `perf_metrics.get_num_generated_tokens()` — Number of tokens actually generated

These are exposed directly by `ov_genai.VLMPipeline` when `GenerationConfig` is passed to `generate()`.

### 15.1 Creative-writing benchmark (1024 tokens)

Prompt: *"Write a mystery-horror short story in the style of Tom Clancy and Stephen King. A man wakes up early in the morning, goes to the bathroom to pee like he always does, but this time nothing comes out. When he looks down, his penis is gone."*

| Metric | GPU (Iris Xe) | CPU (i7-1260P) |
|--------|---------------|----------------|
| TTFT | 3,599 ms | 50,451 ms |
| TPOT | 354.33 ms/token | 781.15 ms/token |
| Throughput | **2.82 tokens/s** | **1.28 tokens/s** |
| Generate duration | **366,103 ms (~6.1 min)** | **849,580 ms (~14.2 min)** |
| Input tokens | 67 | 67 |
| Generated tokens | 1024 | 1024 |
| Saved story | `openvino-artifacts/outputs/story_gpu.txt` | `openvino-artifacts/outputs/story_cpu.txt` |

The GPU scaled much better for long-form generation, completing the story in roughly half the CPU time with a significantly lower TTFT. The CPU run showed much higher first-token latency, likely due to CPU graph optimization and memory layout differences.

### 15.2 Scaling benchmark (double until story finishes or OOM)

Script: `openvino-artifacts/scripts/benchmark_scaling.py`

The script starts at a configurable `max_new_tokens` (default 4096), generates a story, and checks whether the text ends with a sentence terminator. If the story is unfinished, it doubles `max_new_tokens` and retries. It stops on OOM or when the story finishes.

#### GPU scaling results

| Step | Max tokens | Status | Generated tokens | Load (ms) | Generate (ms) | TTFT (ms) | TPOT (ms/t) | Throughput (tok/s) |
|------|------------|--------|------------------|-----------|---------------|-----------|-------------|--------------------|
| 1 | 4096 | Unfinished | 4096 | 46349.61 | 1,685,844.38 | 5213.23 | 410.37 | 2.44 |
| 2 | 8192 | Finished | 1144 | 50118.76 | 376,997.88 | 3383.89 | 326.85 | 3.06 |

- The 4096-token story ended mid-sentence (`...over a hundred beats per...`).
- At 8192 tokens the model hit EOS after 1144 tokens and produced a complete ending.
- No OOM was encountered on the Iris Xe iGPU at either limit.
- Saved files:
  - `openvino-artifacts/outputs/story_gpu_4096.txt`
  - `openvino-artifacts/outputs/story_gpu_8192.txt`

#### CPU scaling results

| Step | Max tokens | Status | Generated tokens | Load (ms) | Generate (ms) | TTFT (ms) | TPOT (ms/t) | Throughput (tok/s) |
|------|------------|--------|------------------|-----------|---------------|-----------|-------------|--------------------|
| 1 | 4096 | Finished | 1254 | 14948.45 | 1,032,538.31 | 78869.59 | 761.10 | 1.31 |

- The CPU finished the story within the 4096-token limit after 1254 tokens.
- No OOM was encountered.
- Saved file:
  - `openvino-artifacts/outputs/story_cpu_4096.txt`

#### Key observations

- **GPU is 2.3× faster for sustained generation**: 3.06 tok/s (GPU) vs 1.31 tok/s (CPU) on the long-form task.
- **GPU TTFT is 23× lower**: 3.38 s vs 78.87 s.
- **CPU generated more tokens to finish**: 1254 (CPU) vs 1144 (GPU), likely due to sampling/EOS non-determinism.
- **No OOM** occurred on either device up to 8192 tokens / ~1250 generated tokens.
- The Iris Xe iGPU can sustain long-form 12B INT4 generation without running out of memory, though throughput is modest.

---

## 16. OpenAI-Compatible Server Deployment

> **Evolution Note:** Prebuilt OVMS v2026.2.1 could not serve `gemma4_unified` models. Initially, a minimal FastAPI server was used as a prototype workaround. **This has now been completely resolved by compiling OVMS 2026.3.0 natively from source with Visual Studio 2026 BuildTools (MSVC 14.51)** linked against local OpenVINO GenAI 2026.4.0.0. Full native C++ OVMS serving is documented in [ovms-gemma4-irisxe-spec.md](ovms-gemma4-irisxe-spec.md) and [WIN2026_BUILD_GUIDE.md](../../WIN2026_BUILD_GUIDE.md).

FastAPI prototype server files live in the sibling artifacts folder for reference:

```text
C:\Users\mohds\Documents\GitHub\openvino-artifacts\
├── scripts\server.py                # FastAPI entry point
├── scripts\test_openai_client.py    # Python client test
└── logs\server.log                   # Runtime log (created on startup)
```

### 16.1 Files created

| File | Purpose |
|------|---------|
| `C:\Users\mohds\Documents\GitHub\openvino-artifacts\scripts\server.py` | FastAPI server around `VLMPipeline` |
| `C:\Users\mohds\Documents\GitHub\openvino-artifacts\scripts\test_openai_client.py` | Python OpenAI client test |
| `C:\Users\mohds\Documents\GitHub\openvino-artifacts\logs\server.log` | Runtime log (created on startup) |

### 16.2 Install server dependencies

```cmd
cd /d C:\Users\mohds\Documents\GitHub\openvino
.venv\Scripts\pip install fastapi uvicorn openai
```

> Versions used when tested: latest `fastapi`, `uvicorn`, and `openai` packages from PyPI as of 2026-07-22.

### 16.3 Start the server

```cmd
.venv\Scripts\python openvino-artifacts\scripts\server.py
```

The server prints:

```text
Loading VLMPipeline from C:\Users\mohds\Documents\GitHub\openvino-artifacts\models\gemma-4-12B-it-qat-int4-ov on GPU...
Pipeline loaded.

Server starting at http://0.0.0.0:8000
OpenAI-compatible endpoint: POST /v3/chat/completions
Model: gemma | Device: GPU
```

### 16.4 Test with curl

```cmd
curl -X POST http://localhost:8000/v3/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"gemma\", \"messages\": [{\"role\": \"user\", \"content\": \"Who is the president of Malaysia?\"}], \"max_tokens\": 128}"
```

### 16.5 Test with Python OpenAI client

```python
from openai import OpenAI

# Timeout is set on the client constructor so it covers both the HTTP
# connection setup and the generation request itself.
client = OpenAI(
    base_url="http://localhost:8000/v3",
    api_key="unused",
    timeout=60.0,
)

response = client.chat.completions.create(
    model="gemma",
    messages=[{"role": "user", "content": "Who is the president of Malaysia?"}],
    max_tokens=128,
)

print(response.choices[0].message.content)
```

Run the included test script from the openvino repo root:

```cmd
cd /d C:\Users\mohds\Documents\GitHub\openvino
.venv\Scripts\python.exe openvino-artifacts\scripts\test_openai_client.py
```

### 16.6 Server test results

| Test | Status | Response summary |
|------|--------|------------------|
| curl `/v3/chat/completions` | ✅ Pass | Corrected the premise: Malaysia has no president; identified the Yang di-Pertuan Agong and Prime Minister Anwar Ibrahim |
| Python OpenAI client | ✅ Pass | Same correct answer |
| `/v1/config` (OVMS-style health/config) | ✅ Pass | `{"gemma": {"model_version_status": [{"version": "1", "state": "AVAILABLE"}]}}` |

### 16.7 Why `/v3`?

OVMS's LLM/VLM endpoint is exposed at `/v3/chat/completions`. Using the same path keeps the server compatible with tools and clients configured for OVMS.

### 16.8 Architecture

```text
Client (curl / OpenAI SDK)
        |
        POST /v3/chat/completions
        |
   openvino-artifacts/scripts/server.py  (FastAPI)
        |
        apply_chat_template() → VLMPipeline.generate()
        |
   openvino_genai 2026.4.0  ← built from source
        |
   Intel Iris Xe iGPU
```
