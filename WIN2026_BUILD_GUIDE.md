# OpenVINO Model Server (OVMS) - Windows VS 2026 Build & Patch Guide

This document explains how to build OpenVINO Model Server (OVMS) on Windows using **Visual Studio 2026 BuildTools (MSVC 14.51)** linked natively against custom locally built **OpenVINO (2026.4.0)** and **OpenVINO GenAI (2026.4.0.0)** libraries (supporting `gemma4_unified` models on GPU/CPU).

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

### Step 1: Clean Build Output Root (Optional if clearing Bazel cache)
```cmd
windows_clean_build.bat opt 1
```

### Step 2: Build OVMS Binary
```cmd
windows_build.bat opt --with_python
```
*Output binary:* `bazel-bin\src\ovms.exe`

### Step 3: Create Self-Contained Release Package
```cmd
windows_create_package.bat opt --with_python
```
*Output archive:* `dist\windows\ovms.zip` (Extracted at `dist\windows\ovms\`)

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
