# OpenVINO Model Server: Serve Gemma-4-12B-IT on Intel Iris Xe iGPU — Spec

**Short name:** `ovms-gemma4-irisxe`  
**Date:** 2026-07-24  
**Status:** ✅ **VERIFIED & WORKING** — Native OVMS 2026.3.0 compiled on Windows (MSVC 14.51 / VS 2026 BuildTools) linked against local OpenVINO GenAI 2026.4.0.0. Full GPU acceleration for `gemma-4-12B-it-qat-int4-ov` (`gemma4_unified` multi-modal architecture) via OpenAI Chat Completions (`/v3/chat/completions`) & OpenAI Responses API (`/v3/responses`).  
**Author:** Pair-programmed by User & Antigravity  
**Host:** Windows 11, Intel Core i7-1260P, Intel Iris Xe iGPU, 32 GB RAM  
**Model:** `HarmenWessels/gemma-4-12B-it-qat-int4-ov` (~12B INT4 QAT, ~6–7 GB on disk)  
**Inference Hub:** `C:\Users\mohds\Documents\GitHub\openvino-artifacts`  
**Build Source Repo:** `C:\Users\mohds\Documents\GitHub\model_server`  

---

## 1. Executive Summary & Architecture

This specification defines the production architecture for building, deploying, and serving **Gemma 4 12B** on GPU using **OpenVINO Model Server (OVMS)**.

### The 2-Hub Design Pattern:
1. **`model_server` (Build Engineering Workspace):** Contains C++ source modifications, Bazel rules, VS 2026 build scripts, and exports the deployment package `dist\windows\ovms.zip`.
2. **`openvino-artifacts` (Inference Deployment Hub):** Contains downloaded Hugging Face model weights (`models/`), GPU compilation cache (`cache/`), server configuration (`ovms_config.json`), extracted binary runtime (`ovms/`), and inference client test scripts.

---

## 2. End-to-End 4-Step Deployment Blueprint

```
+-----------------------------------------------------------------------------------+
| STEP 1: BUILD (in model_server)                                                  |
| Compile ovms.exe natively with MSVC 14.51 (VS 2026) against GenAI 2026.4.0.0.    |
| Command: windows_build.bat opt --with_python                                     |
| Output:  dist\windows\ovms.zip                                                   |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| STEP 2: DEPLOY (to openvino-artifacts)                                           |
| Extract release archive into the Inference Deployment Hub.                       |
| Destination: C:\Users\mohds\Documents\GitHub\openvino-artifacts\ovms\            |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| STEP 3: LAUNCH (Server Execution on GPU)                                         |
| Initialize environment and launch server pointing to MediaPipe graph.            |
| Command: call ovms\setupvars.bat && ovms\ovms.exe --config_path ovms_config.json |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| STEP 4: INFER (OpenAI API Verification)                                          |
| Execute live queries via OpenAI Python SDK (v2.48.0) or Responses API (/v3/responses)|
+-----------------------------------------------------------------------------------+
```

---

## 3. Step 1: Building OVMS from Source

For detailed Bazel settings, dependency staging, and VS 2026 MSVC toolset configurations, refer to:  
📖 **[WIN2026_BUILD_GUIDE.md](../../WIN2026_BUILD_GUIDE.md)**

```cmd
cd /d C:\Users\mohds\Documents\GitHub\model_server
windows_build.bat opt --with_python
windows_create_package.bat opt --with_python
```
*Generates package:* `C:\Users\mohds\Documents\GitHub\model_server\dist\windows\ovms.zip`

---

## 4. Step 2: Deploying to `openvino-artifacts`

Extract the generated `ovms.zip` directly into `openvino-artifacts\ovms\`:

```cmd
cd /d C:\Users\mohds\Documents\GitHub\openvino-artifacts
tar -xf C:\Users\mohds\Documents\GitHub\model_server\dist\windows\ovms.zip -C C:\Users\mohds\Documents\GitHub\openvino-artifacts
```

### Directory Structure of `openvino-artifacts`:
```text
openvino-artifacts/
├── ovms_config.json                   <-- Root Server Config
├── ovms/                              <-- Extracted Binary Runtime
│   ├── ovms.exe                       <-- Native VS 2026 Binary
│   ├── setupvars.bat                  <-- Runtime Environment Script
│   ├── openvino_genai.dll             <-- Local GenAI 2026.4.0.0 Shared Library
│   └── python/                        <-- Embedded Python 3.12 Runtime
├── models/
│   └── gemma-4-12B-it-qat-int4-ov/   <-- Hugging Face Model Weights & graph.pbtxt
└── cache/
    └── ov_gpu_cache/                  <-- OpenVINO Level Zero GPU Compiled Kernels
```

---

## 5. Step 3: Server Configuration & Launch

### 5.1 Canonical `ovms_config.json` (`openvino-artifacts/ovms_config.json`)
```json
{
    "model_config_list": [],
    "mediapipe_config_list": [
        {
            "name": "gemma",
            "graph_path": "C:/Users/mohds/Documents/GitHub/openvino-artifacts/models/gemma-4-12B-it-qat-int4-ov/graph.pbtxt"
        }
    ]
}
```

### 5.2 GPU Target Device Configuration (`graph.pbtxt`)
GPU target execution is specified inside `models/gemma-4-12B-it-qat-int4-ov/graph.pbtxt`:

```pbtxt
node_options: {
    [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
        models_path: "C:/Users/mohds/Documents/GitHub/openvino-artifacts/models/gemma-4-12B-it-qat-int4-ov",
        device: "GPU"
    }
}
```

### 5.3 Launch Server (GPU Mode)
In an elevated Command Prompt (`cmd.exe`):

```cmd
cd /d C:\Users\mohds\Documents\GitHub\openvino-artifacts
call ovms\setupvars.bat
ovms\ovms.exe --config_path ovms_config.json --port 9000 --rest_port 8000
```

---

## 6. Step 4: Verification & Live Inference

### 6.1 Using the Official OpenAI Python SDK (v2.48.0)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v3",
    api_key="unused"
)

response = client.chat.completions.create(
    model="gemma",
    messages=[{"role": "user", "content": "who is your daddy?"}],
    max_tokens=64
)

print(response.choices[0].message.content)
```

**Output:**  
> *"I was developed by Google DeepMind."*

### 6.2 Using the OpenAI Responses API (`POST /v3/responses`)
```cmd
curl -X POST http://localhost:8000/v3/responses \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"gemma\", \"input\": \"Who is the Prime Minister of Malaysia?\"}"
```

**Output:**  
`"object": "response"`, `"status": "completed"` with answer `"Anwar Ibrahim"`.
