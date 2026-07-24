# OpenVINO Model Server (OVMS) - Windows VS 2026 Build & Patch Guide

This document explains how to build OpenVINO Model Server (OVMS) on Windows using **Visual Studio 2026 BuildTools (MSVC 14.51)** linked natively against custom locally built **OpenVINO (2026.4.0)** and **OpenVINO GenAI (2026.4.0.0)** libraries.

> **Why This Custom Build is Required:**  
> Standard upstream prebuilt OVMS binaries support standard LLMs, BUT the **Gemma 4 12B model** (`HarmenWessels/gemma-4-12B-it-qat-int4-ov`) uses the new **`gemma4_unified` multi-modal architecture** (which integrates unified text, vision, and text-embedding sub-pipelines). Building OVMS natively against OpenVINO GenAI 2026.4.0.0 unlocks native C++ auto-detection and execution for `gemma4` tool/reasoning parsers (`gemma4_unified`) on GPU and CPU.

---

## 1. Patch Architecture & Strategy

To ensure seamless upgrades when upstream `model_server` releases updates, all modifications are structured into atomic Git commits and exported as a portable Git patch:

- **Git Branch:** `feature/win2026-vs-build`
- **Patch File:** `win2026_genai_build.patch`

### Commit Structure:
1. `d3d0a2a9` **`build(windows): update build scripts and warning flags for VS 2026 BuildTools (MSVC 14.51)`**
   - Updates Visual Studio paths to `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools`
   - Configures MSVC compiler version `BAZEL_VC_FULL_VERSION=14.51.36231`
   - Updates CMake generator strings to `"Visual Studio 18 2026"` and toolset `-T v145`
   - Suppresses MSVC 14.51 C5285 warnings originating from XLA headers in `.bazelrc` (`/wd5285`)

2. `c37c9e23` **`build(bazel): update genai_windows.BUILD globs and transitive headers`**
   - Configures `third_party/genai/genai_windows.BUILD` to consume headers from `genai_hdrs/openvino/**/*.*` with `strip_include_prefix = "genai_hdrs"`
   - Adds `deps = ["@windows_openvino//:openvino_new_headers"]` so core OpenVINO headers (e.g. `<openvino/runtime/tensor.hpp>`) are transitively declared

3. `59c5d73e` **`build(deps): wire //third_party:genai dependencies into src Bazel targets`**
   - Adds `"//third_party:genai"` dependency to targets missing explicit GenAI headers:
     - `src/tokenize/BUILD` (`tokenize_parser`)
     - `src/embeddings/BUILD` (`embeddings_servable`)
     - `src/audio/text_to_speech/BUILD` (`t2s_servable`, `t2s_calculator`)
     - `src/BUILD` (`libovms_version_impl`)

---

## 2. Staging Local OpenVINO & GenAI Dependencies

OVMS expects OpenVINO runtime dependencies in `C:\opt\openvino\runtime`. Before building, local build output binaries and headers must be staged:

### PowerShell Staging Script (`stage_local_deps.ps1`):

```powershell
# Paths
$ovRuntime = "C:\opt\openvino\runtime"
$genaiRepo = "C:\Users\mohds\Documents\GitHub\openvino.genai"
$ovRepo    = "C:\Users\mohds\Documents\GitHub\openvino"

# 1. Stage GenAI C++ headers (source + CMake-generated headers like version.hpp)
$genaiHdrsDst = "$ovRuntime\genai_hdrs\openvino\genai"
New-Item -ItemType Directory -Force -Path $genaiHdrsDst | Out-Null
Copy-Item "$genaiRepo\src\cpp\include\openvino\genai\*" $genaiHdrsDst -Recurse -Force
if (Test-Path "$genaiRepo\build\src\cpp\openvino\genai\version.hpp") {
    Copy-Item "$genaiRepo\build\src\cpp\openvino\genai\version.hpp" $genaiHdrsDst -Force
}

# 2. Stage GenAI Binaries (.dll and .lib)
Copy-Item "$genaiRepo\build\openvino_genai\bin\Release\*.dll" "$ovRuntime\bin\intel64\Release" -Force
Copy-Item "$genaiRepo\build\openvino_genai\lib\Release\*.lib" "$ovRuntime\lib\intel64\Release" -Force

# 3. Stage OpenVINO Core Binaries (.dll and .lib)
Copy-Item "$ovRepo\bin\intel64\Release\*.dll" "$ovRuntime\bin\intel64\Release" -Force
Copy-Item "$ovRepo\bin\intel64\Release\*.lib" "$ovRuntime\lib\intel64\Release" -Force
```

---

## 3. Building and Packaging OVMS

> ⚠️ **CRITICAL PREREQUISITE - ADMINISTRATOR PRIVILEGES:**  
> You **MUST** run all build commands inside an **Administrator Command Prompt (`cmd.exe`)** (or an elevated shell with Windows Developer Mode symlinks enabled). Bazel creates symbolic links and directory junctions during MSVC compilation; running without Administrator privileges will cause Bazel to throw `Access Denied` or `Symlink creation failed` compilation errors.

### Step 1: Clean Build Output Root (Optional if clearing Bazel cache)
```cmd
windows_clean_build.bat opt 1
```

### Step 2: Build OVMS Binary (in Administrator cmd.exe)
```cmd
windows_build.bat opt --with_python
```
*Output binary:* `bazel-bin\src\ovms.exe`

### Step 3: Create Self-Contained Release Package
```cmd
windows_create_package.bat opt --with_python
```
*Output archive:* `dist\windows\ovms.zip` (Extracted at `dist\windows\ovms\`)

*(For general baremetal deployment, extraction, and Windows Service installation instructions, see [docs/deploying_server_baremetal.md](docs/deploying_server_baremetal.md).)*

---

## 4. Serving & Running VLM Models on GPU

### Model Graph Configuration (`graph.pbtxt`):
Ensure the Mediapipe graph options set `device: "GPU"`:

```pbtxt
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
          models_path: "C:/Users/mohds/Documents/GitHub/openvino-artifacts/models/gemma-4-12B-it-qat-int4-ov",
          plugin_config: '{}',
          dynamic_split_fuse: false,
          max_num_seqs: 256,
          max_num_batched_tokens: 8192,
          cache_size: 0,
          device: "GPU",
          pipeline_type: VLM
      }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "LOOPBACK:0"
        }
      }
    }
  }
}
```

### Start Server:
```cmd
call dist\windows\ovms\setupvars.bat
dist\windows\ovms\ovms.exe --config_path C:\Users\mohds\Documents\GitHub\openvino-artifacts\ovms_config.json --port 9000 --rest_port 8000
```

### OpenAI API Chat Completion Request:
```bash
curl -X POST http://localhost:8000/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "messages": [
      {
        "role": "user",
        "content": "Who is the Prime Minister of Malaysia?"
      }
    ],
    "max_tokens": 64
  }'
```

### OpenAI Responses API Request (`/v3/responses`):
```bash
curl -X POST http://localhost:8000/v3/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "input": [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "Who is the Prime Minister of Malaysia?"
          }
        ]
      }
    ],
    "max_output_tokens": 64
  }'
```

### Stop Server:
To stop the server process after completing inference requests:

- **Command Prompt (CMD):**
  ```cmd
  taskkill /F /IM ovms.exe
  ```
- **PowerShell:**
  ```powershell
  Get-Process ovms -ErrorAction SilentlyContinue | Stop-Process -Force
  ```
- **Interactive Console:**
  Press `Ctrl + C` in the active terminal window to initiate a graceful shutdown.

---

## 5. How to Reapply Changes on Upstream Updates

When `model_server` receives updates from upstream `origin/main`:

### Method A: Git Apply (Fastest)
```bash
# 1. Fetch and checkout latest upstream
git checkout main
git pull origin main

# 2. Apply patch file
git apply win2026_genai_build.patch
```

### Method B: Git Rebase (Recommended for Git History)
```bash
# 1. Fetch latest upstream
git fetch origin

# 2. Rebase feature branch onto updated main
git checkout feature/win2026-vs-build
git rebase origin/main
```
*(If merge conflicts occur, resolve the conflict and run `git rebase --continue`).*

---

## 6. Key Troubleshooting Insights

1. **Missing `version.hpp` (`C1083` error):**
   `version.hpp` is a CMake-generated header generated into `openvino.genai\build\src\cpp\openvino\genai\version.hpp`. Ensure `stage_local_deps.ps1` copies from `build\` into `C:\opt\openvino\runtime\genai_hdrs\openvino\genai`.

2. **Bazel Output Root (`C:\opt`):**
   `windows_build.bat` passes `--output_user_root=C:\opt`. Do NOT run plain `bazel clean` without `--output_user_root=C:\opt`, otherwise stale cached external symlinks will persist. Use `windows_clean_build.bat opt 1`.

3. **MSVC 14.51 Compiler Warning (`C5285`):**
   MSVC 14.51 enforces warning C5285 on XLA header files. This is suppressed globally in `.bazelrc` using `--copt=/wd5285` and `--host_copt=/wd5285`.

---

## 7. Future MSVC Toolset Upgrade Guide & Architectural FAQs

When a new Visual Studio IDE version or MSVC compiler toolset update is released in the future, follow this checklist to anticipate and resolve toolchain issues:

### Q1: How do I find and update `BAZEL_VC_FULL_VERSION` for a new MSVC release?
- **Why it matters:** Bazel does not search `%PATH%` for `cl.exe`. It constructs the exact path using `VC\Tools\MSVC\<BAZEL_VC_FULL_VERSION>\`.
- **How to find it:** Run PowerShell:
  ```powershell
  Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio\<VS_VERSION>\BuildTools\VC\Tools\MSVC"
  ```
- **Action:** Update `BAZEL_VC_FULL_VERSION` in `windows_build.bat`, `windows_setupvars.bat`, `windows_install_build_dependencies.bat`, and `windows_test.bat`.

### Q2: Why must CMake Generator `-G` and Toolset `-T` be updated together?
- **Why it matters:** CMake's default generator (e.g. `"Visual Studio 17 2022"`) does not recognize toolsets newer than its release (e.g. `-T v145`). Omitting `-G` causes CMake to fail with `Generator does not support toolset`.
- **Action:** Pair the generator with the matching toolset:
  ```cmd
  cmake -G "Visual Studio 18 2026" -T v145 ..
  ```

### Q3: Why add `--copt=/wd<WARNING_ID>` flags in `.bazelrc`?
- **Why it matters:** New MSVC toolsets introduce new warning codes. Because OVMS compiles with `/WX` (warnings-as-errors), newly introduced MSVC warnings in third-party headers (such as XLA/TensorFlow SIMD headers) will abort the build.
- **Action:** If a new MSVC compiler update introduces a fatal warning (e.g. `fatal error C5285`), suppress it in `.bazelrc` via `/wd<WARNING_ID>`:
  ```ini
  build:windows --copt=/wd5285
  build:windows --host_copt=/wd5285
  ```

### Q4: Is `src/version.hpp` generated automatically during build?
- **Why it matters:** `src/version.hpp` contains `#define PROJECT_VERSION "2026.3.0.921fa646"` and `#define BAZEL_BUILD_FLAGS "--config=win_mp_on_py_on"`.
- **How it works:** When `windows_build.bat` runs, it executes `windows_set_ovms_version.py`. This script queries `git rev-parse --short HEAD` (commit hash `921fa646`) and active Bazel build flags, then automatically generates and overwrites `src/version.hpp`.

### Q5: How does `third_party/genai/genai_windows.BUILD` work?
- **Bazel syntax breakdown:**
  ```python
  cc_library(
      name = "genai_headers",
      hdrs = glob(["genai_hdrs/openvino/**/*.*"]),
      strip_include_prefix = "genai_hdrs",
      deps = ["@windows_openvino//:openvino_new_headers"],
      visibility = ["//visibility:public"],
  )
  ```
  - `glob(["genai_hdrs/openvino/**/*.*"])`: Recursively globs all GenAI C++ headers staged in `C:\opt\openvino\runtime\genai_hdrs\openvino\`.
  - `strip_include_prefix = "genai_hdrs"`: Strips the `genai_hdrs/` prefix from include paths, allowing source files to write `#include <openvino/genai/llm_pipeline.hpp>`.
  - `deps = ["@windows_openvino//:openvino_new_headers"]`: Instructs Bazel to transitively supply core OpenVINO headers (e.g. `<openvino/runtime/tensor.hpp>`) whenever GenAI headers are included.

### Q6: What is `@windows_openvino//:openvino_new_headers` in Bazel label syntax?
- **Bazel label breakdown:** `@repo_name//package_path:target_name`
  - `@windows_openvino`: External repository defined in `WORKSPACE` pointing to `C:\opt\openvino\runtime` using `third_party/openvino/openvino_windows.BUILD`.
  - `//`: Root package path of the repository.
  - `:openvino_new_headers`: C++ header target defined in `third_party/openvino/openvino_windows.BUILD` that globs core OpenVINO headers (`include/openvino/**/*.*`).

### Q7: Why do missing `deps` in Bazel BUILD files pass on Upstream Linux CI, but fail on Windows with `--with_python`?
- **1. Transitive Header Leakage on Linux:** On Linux, OpenVINO core and GenAI headers are installed together in `/usr/local/include/`. When a Bazel target depends on `//third_party:openvino`, Bazel passes `-I/usr/local/include` to GCC. Because both core and GenAI headers live in `/usr/local/include`, GCC accidentally finds `<openvino/genai/tokenizer.hpp>` even if `"//third_party:genai"` was missing from `deps`.
- **2. Strict Header Isolation on Windows:** In our clean build architecture, local GenAI headers are isolated in `genai_hdrs`. If a target omits `"//third_party:genai"`, Bazel refuses to pass `-Igenai_hdrs` to MSVC, immediately triggering `fatal error C1083`.
- **3. Skipped Feature Targets in Upstream Windows CI:** Upstream's minimal Windows CI builds without Python (`--with_python`) and skips compiling MediaPipe + Python feature targets (like `tokenize_parser`, `embeddings_servable`, and `t2s_servable`). Building with `--with_python` (`--config=win_mp_on_py_on`) compiles these targets on Windows for the first time, exposing the missing dependencies.

### Q8: Why replace conditional `select(...)` with unconditional `"//third_party:genai"` in `src/BUILD` (`libovms_version_impl`)?
- **The problem:** `src/version.cpp` unconditionally includes `#include <openvino/genai/version.hpp>` to print the GenAI backend version. Upstream wrapped `"//third_party:genai"` in `select({"//:not_disable_mediapipe": ["//third_party:genai"]})`. On Windows with `--config=win_mp_on_py_on`, `select` evaluated `//conditions:default` (`[]`), stripping GenAI headers and causing `fatal error C1083`.
- **The fix:** Because `version.cpp` unconditionally requires `version.hpp`, GenAI is an unconditional dependency. Replacing `select(...)` with `"//third_party:genai"` directly in `deps` guarantees Bazel supplies GenAI headers on every Windows configuration.

### Q9: What exact target dependency additions were made in Commit `59c5d73e` and why?
- **Audit Table:**

| BUILD File | Target Name | Source File | `#include` in Source | Missing Dep Added |
| :--- | :--- | :--- | :--- | :--- |
| `src/tokenize/BUILD` | `tokenize_parser` | `tokenize_parser.cpp` | `#include <openvino/genai/tokenizer.hpp>` | `"//third_party:genai"` |
| `src/embeddings/BUILD` | `embeddings_servable` | `embeddings_servable.cpp` | `#include <openvino/genai/llm_pipeline.hpp>` | `"//third_party:genai"` |
| `src/audio/text_to_speech/BUILD` | `t2s_servable` & `t2s_calculator` | `t2s_servable.cpp` | `#include <openvino/openvino.hpp>` | `"//third_party:openvino"` |




