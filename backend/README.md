# ComplexUI Backend

FastAPI service and node-code toolchain for ComplexUI.

- `app/schemas.py` — Pydantic v2 models for node schemas, workflow graphs, and execution telemetry events.
- `app/transpiler.py` — node-code DSL transpiler (PEP 723 extraction, typed
  returns, input/widget/output extraction, function wrapping). Assigned
  `typing.Literal[...]` annotations become fixed-choice widgets: the schema
  carries an `options` list, and defaults, workflow overrides and wired values
  are validated against it. Inputs annotated `Optional[TYPE]` or `TYPE | None`
  with a `= None` default become optional sockets that may stay unconnected and
  are excluded from image batch expansion.
- `app/executor.py` — topological DAG executor with a bounded
  content-addressable LRU cache (64 outputs / 32 MiB), trust gating,
  cancellation checks and WebSocket-ready event streaming. Nodes receive
  `report_progress(fraction, message)`, `raise_if_cancelled()` and
  `outputs_root` globals for cooperative long-running work and generated-file
  paths. `File` widgets are resolved strictly inside the workspace before node
  code runs; `FolderPath` widgets may read a user-picked absolute folder; and
  `ModelSelector` widgets resolve inside the repo model root. File, folder and
  model values are fingerprinted into the cache key.
- `app/execution_host.py` — disposable child-process execution, cache/approval
  transfer, wall-clock and memory guards, process-tree termination and recovery.
- `app/execution_worker.py` — child-process entry point that runs the executor
  and streams events/cache snapshots back to the server process.
- `app/datasets.py` — repo-root table/image/video uploads with sanitized,
  collision-safe names, atomic streaming writes (100 MiB tables, 500 MiB
  media), listing/deletion, bounded Polars table previews, raw media paths, and
  generated-media listing/deletion/resolution under repo-root `outputs/`.
  Generated files may live in nested validated subfolders (for example
  `2026-10-23/shots/file.png`); the store exposes relative ids and rejects
  traversal, absolute paths and symlink escapes. `app/artifacts.py` keeps a
  bounded allowlist of media paths produced by completed runs so external Save
  Media destinations can be previewed, copied, opened and revealed safely.
- `app/models.py` — metadata-only scanner for the repo-root `models/` library.
  It creates `yolo/`, `torchvision/`, `classification/`, `diffusion/`, `unet/`,
  `vae/`, and `clip/`, discovers custom type folders, validates
  paths/extensions/symlinks, and returns size + bounded mtime fingerprints for
  files and diffusers-style folders without importing model contents.
  `COMPLEXUI_MODELS_DIR` overrides the root.
- `app/transpiler.py` recognizes `local_model: ModelSelector =
  ModelSelector(type="yolo")` as a model widget and records `model_type`,
  `model_select` (`file`/`folder`/`any`), `model_required`, and
  `model_include` on the schema.
- `app/exporter.py` — standalone PEP 723 Python and nbformat 4.5 notebook
  generation using the same transpiler/DAG plan (scoped exports and deduped
  dependencies included).
- `app/output_capture.py` — best-effort capture of `print`, `display`, tables,
  images, HTML and Plotly figure side effects for each node.
- `app/env_manager.py` — PEP 723 dependency checks and a cancellable
  `uv pip install` runner. Torch-family packages are routed to the CPU index or
  the cu130 CUDA index selected by `COMPLEXUI_ACCELERATOR`; switching builds
  triggers a targeted reinstall. The dev group uses `tool.uv.sources` to pin
  cu130 on Windows and CPU wheels on Linux, so `uv sync` preserves the intended
  build.
- `app/accelerator.py` — accelerator configuration, NVML-based `auto` detection,
  and the torch CPU/CUDA wheel index selection.
- `app/optimizations.py` — dependency-light runtime optimization configuration:
  attention backend choices, memory profiles, allocator/`PYTORCH_ALLOC_CONF`
  helpers, compile defaults, optional-package detection and the
  `/api/health` optimization diagnostics. It never imports torch, so the server
  can report settings before CUDA initialization.
- `app/attention_dispatch.py` — worker-side capability checks, SDPA sub-backend
  flags and the guarded optional-kernel dispatch wrapper for SageAttention,
  FlashAttention 2/4, xFormers and Comfy Kitchen INT8/SOL. A process-wide
  launch default is combined with a `ContextVar` per-MODEL override, so an
  `Attention Backend` node only affects that model's diffusion forward. It
  falls back to exact SDPA for unsupported shapes, masks, dtypes or versions.
- `app/quantization.py` — metadata-only detection and capability gates for
  pre-quantized SafeTensors/GGUF checkpoints. It reads Comfy `.comfy_quant`
  tensors, sidecar JSON, legacy scaled-FP8 dtypes and bounded GGUF headers
  without loading weights, and exposes the `/api/health` format registry.
- `app/single_file_quantization.py` — single-file compatibility path. It
  dequantizes Comfy LDM tensors, feeds the resulting state dict through
  diffusers' key-based single-file conversion, then re-patches the SD/SDXL UNet
  with native quantized weights using a marker-based module mapping. CLIP
  components in the same checkpoint remain dequantized.
- `app/quantized_loading.py` — worker-side Comfy-format component loader. It
  reconstructs `comfy_kitchen` quantized tensors from `.comfy_quant` metadata,
  replaces matching `nn.Linear`/`nn.Embedding` modules, and exposes a native
  kernel path with an explicit dequantized fallback for
  `full_precision_matrix_mult` or disabled formats. Single-file checkpoints are
  intentionally not accepted here.
- `app/memory_estimate.py` — static model memory estimates from checkpoint size
  and quantization metadata. Run-history peak metrics remain the measured
  source of truth.
- `app/workflow_estimate.py` — discovers typed `ModelSelector` widgets in a
  workflow graph and returns distinct model weights plus resident/offload
  estimates without executing node code.
- `app/nvml.py` — best-effort NVIDIA GPU discovery through `nvidia-ml-py`.
- `app/system_metrics.py` — `psutil` CPU/RAM metrics plus NVML GPU utilization,
  VRAM and temperature. CPU temperature is intentionally not collected.
- `app/security.py` — SHA-256 template registry and session approval tracking;
  flattened nested nodes may supply an `approval_id` to retain node-scoped trust.
- `app/network_security.py` — loopback-first `Host`/WebSocket `Origin`
  validation and a request-size guard with path-prefix overrides (used to allow
  larger uploads at `/api/uploads`).
- `app/template_catalog.py` — display metadata (title, category, description,
  palette visibility) for the official node catalog. Example dataset loaders
  (iris/wine/breast-cancer/diabetes/digits), the synthetic classification,
  messy, image and video generators, the synthetic tensor dataset, and the
  legacy per-mode resize nodes are flagged `hidden`: they ship for bundled
  workflows and older canvases but stay out of the node library.
- `app/workflow_catalog.py` — example workflow models, socket/DAG validation
  and `WorkflowGraph` materialisation.
- `app/workspace.py` — file-backed named projects, 30 retained version
  snapshots, workspace trash, and close/visibility recovery sessions.
- `app/main.py` — FastAPI server: REST endpoints plus `/ws/execution` streaming.
- `app/artifacts.py` — bounded allowlist of media paths produced by completed
  runs, used for external Save Media preview/copy/open/reveal actions.
- `app/os_actions.py` — platform open/reveal helpers and the native folder
  dialog (Windows PowerShell, macOS `osascript`, Linux `zenity`/`kdialog`).
- `app/templates/` — 135 official ML node templates across 11 categories:
  scalar values (int/float/bool/string), data loaders (sklearn sets,
  synthetic/messy generators, CSV/Parquet/Excel/NDJSON), cleaning, EDA
  (histogram, scatter, box, bar and correlation heatmap), preprocessing,
  imbalanced-data samplers, models (scikit-learn + XGBoost/LightGBM) with
  feature-importance charts, evaluation (confusion matrix and regression
  diagnostics), the PyTorch tensor path (training-history chart), image
  generation (SD1.5/SDXL checkpoints plus diffusers component loaders for
  Stable Diffusion, FLUX 1/2/3, Qwen Image 2.1, Z-Image, Krea 2, Ideogram 4,
  Fibo 2, Anima and MOSS, CLIP/dual-CLIP, VAE, conditioning, scheduler and
  sampler nodes), and computer vision (torchvision image
  classification/detection/segmentation, video classification, YOLO
  image/video detection and segmentation, image/video loaders, media preview
  and media saving). Resolution and Empty Latent Image share
  resolution/megapixel modes, and Resize Image/Resize Latent consolidate
  dimensions, side, megapixels and multiplier modes. Optional dependencies
  (`torch`, `torchvision`, `diffusers>=0.42`, `transformers`, `accelerate`,
  `sentencepiece`, `ultralytics`, `opencv-python`, `imageio-ffmpeg`,
  `safetensors`, `onnxruntime`, `xgboost`, `lightgbm`, `fastexcel`,
  `imbalanced-learn`) are declared in PEP 723 headers and installed on demand.
  Generated videos are encoded as browser-playable H.264 MP4/MOV/MKV through
  the bundled `imageio-ffmpeg` encoder. Save Media writes images and videos with
  a configurable folder and filename template. The default is
  `outputs/<YYYY-MM-DD>/<model>_<counter>[_<batch>].<ext>`; the folder may be a
  relative path under `outputs/` or an absolute folder chosen with the native
  Browse dialog. Filename tokens include `%date%`, `%time%`, `%counter%`,
  `%model%`, `%width%`, `%height%` and `%batch%`; `%seed%`, `%input_name%` and
  `%input_folder%` wait for metadata support. Counters are 3-digit and batches
  get an automatic index when the template omits one. PNG/JPEG/WebP/BMP/TIFF/GIF
  format and quality controls are available. The `outputs_root` node global and
  the exporter use the same generated-output root. External destinations are
  allowed; their Open/Copy/Preview/Reveal actions are limited to paths recorded
  by a completed run (live event registry or retained run history).
  Scalar outputs can also target any widget parameter (for example
  `float_value.result -> train_test_split.test_size`); the executor overrides
  that keyword argument after applying stored widget values.
  Image-generation tuning nodes: `Attention Backend` selects a capability-checked
  PyTorch SDPA sub-backend; `Model Precision` converts diffusion weights in place
  and tracks storage/compute/accumulation; `Compile Model` opts in to
  `torch.compile` with eager fallback; and `Model Memory` sets residency,
  model-level offload, attention slicing and channels-last. `Load Diffusion
  Model` and `Load VAE` apply launcher defaults for attention, memory profile,
  compilation and channels-last, and expose `storage_dtype`, `compute_dtype`,
  `load_time_ms` and residency metadata for previews. `Attention Backend` is a
  per-MODEL override: the launcher backend stays the global default outside the
  diffusion forward, and optional kernels are skipped by `auto` compilation mode
  because they can graph-break or fall back to SDPA.
- Pre-quantized checkpoints are detected without loading weights. The format
  registry covers INT8 W8A8, INT8 ConvRot W4A4, GGUF, NVFP4, MXFP8, scaled FP8
  and NF4, with per-format runtime, package, component and compute-capability
  gates. Detection accepts Comfy `.comfy_quant` JSON tensors, native GGUF
  headers, sidecar `.quant.json` files and legacy scaled-FP8 dtypes. The Models
  Library shows factual format/parameter/memory tags; no per-GPU recommendation
  is made. `POST /api/estimate/workflow` returns distinct model weights,
  resident and offload estimates for a `WorkflowGraph`. Diffusers component
  folders with `.comfy_quant` metadata load through `comfy-kitchen` native
  INT8/ConvRot native kernels and FP8/MXFP8/NVFP4 quantized storage with
  dequantized linear fallback, plus INT8/FP8 embeddings; unsupported or
  explicitly emulated layers dequantize to the compute dtype. Single-file
  SD1.5/SDXL quantized checkpoints are converted to diffusers and the UNet is
  re-patched with native INT8/ConvRot weights; CLIP components in the same
  checkpoint remain dequantized. GGUF headers are detected and tagged, but no
  GGUF weight runtime is wired into the loaders yet. Quantized VAEs are out of
  scope.
- `app/workflows/` — 39 example workflow definitions (JSON) served by the API
  and validated at load time, including offline vision image/video pipelines
  and local image-generation workflows.

Ultralytics (YOLO) templates are kept in a separate node family from the
torchvision templates so non-YOLO vision workflows do not install it.
The `ultralytics` package is **AGPL-3.0**; review its license before
distributing generated applications or network services that use YOLO nodes.

## Image batches

`IMAGE_BATCH` is an ordered list of PIL images. When such a list reaches a
scalar `IMAGE` input, the executor runs that node once per image, repeats scalar
`IMAGE` inputs for every item, and requires equal lengths for multiple batch
inputs. Optional inputs (`Optional[IMAGE] = None` or `IMAGE | None = None`) are
exempt: they receive the whole value and may stay unconnected. `MEDIA` and
`IMAGE_BATCH` inputs receive the whole list. A node may read
the current item through the `batch_index`/`batch_size` globals; progress is
mapped across the batch, cancellation is checked between items, errors report
the failing item index, and the limit is 256 items. Empty batches produce empty
output lists without executing the node. `Load Images From Folder` reads a
bounded, name- or mtime-sorted folder of images into a batch; `Load Image`
returns an RGB image plus an alpha/luminance mask output. `FolderPath`
widgets resolve absolute paths or repo-relative paths and are read-only. The
five-input `Switch` passes one renamed input through, and `Resize Image`,
`Resize Latent` and `Preview Mask` accept batches through normal socket
compatibility.

## Attention, precision, compilation and memory

The runtime exposes one capability-oriented SDPA selector rather than a fixed
backend. `Attention Backend` and `Load Diffusion Model` validate the device
capability, torch/CUDA version and requested backend before enabling
`torch.backends.cuda` SDPA sub-backends. `auto` falls back to the safest
available path; an explicit unsupported request fails with the failed
capability and an install/arch hint. The selected backend and reason are printed
by the node and stored on the `MODEL` wrapper for previews and diagnostics.
SageAttention 2/2++, FlashAttention 2/3 and xFormers are selectable when the
matching package is installed; their runtime adapters fall back to exact SDPA
when a request shape, mask, dtype or package constraint is unsupported.
FlexAttention and Comfy Kitchen kernels are detected/reported only. Optional
kernels are not bundled dependencies; install the matching wheel manually.
`--attention-benchmark` or the node's benchmark toggle times available backends
on synthetic tensors, checks correctness against a float reference and selects
the fastest; the per-backend results are retained on the `MODEL` wrapper.

Windows SageAttention 2.2 installation (RTX 50 / cu130 / torch >= 2.10):

```powershell
uv pip install triton-windows
uv pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
```

`post6` is the latest NVIDIA wheel; `post7` is the AMD RDNA4 build. The wheel
uses Python ABI3 (>=3.10), so it works with the backend's Python 3.13 venv.
Select it with `--attention-backend sageattention` or the `Attention Backend`
node. The non-square cross-attention path and unsupported masks deliberately
fall back to SDPA; self-attention uses the quantized kernel. These packages are
environment-local and are not added to `pyproject.toml`; a clean `uv sync
--exact` may remove them, so reinstall the two wheels after rebuilding the
venv. The validated combination on this machine is
`sageattention 2.2.0+cu130torch2.10.0andhigher.post6` plus
`triton-windows 3.9.0.post30`.

`Model Precision` keeps checkpoint storage dtype, compute dtype and accumulation
mode explicit. FP32, FP16 and BF16 conversion is supported for the diffusion
transformer; VAE loading has an independent `compute_dtype` widget. FP16
accumulation is experimental, requires `--experimental-fp16-accumulation`, and
samplers reject non-finite predictions instead of silently propagating NaNs.
Pre-quantized formats (FP8, MXFP8, NVFP4, INT8/W8A8, ConvRot, NF4 and GGUF) are
reported by diagnostics but are refused by the current loader with an actionable
message; they are never treated as interchangeable.

`Compile Model` and `--torch-compile` apply `torch.compile` in place to the
transformer forward call. Compilation is opt-in, uses a stable bounded Inductor
cache directory, reports cold compile time separately, and falls back to eager
on an `auto`/server-mode failure. Apply attention and precision patches before
compilation so the captured graph sees the final backend and dtype. The
`reduce-overhead` mode and compilation combined with offloading are rejected in
this build because CUDA graphs and moving weights are not safe to combine yet.
Channels-last is enabled for SD1.5/SDXL conv UNets and compatible VAEs; DiT
models stay in their native format. The Inductor cache directory is stable but
not automatically pruned; delete `<workspace>/cache/torchinductor` to reclaim
disk space.

`Model Memory` resolves `auto`, `low`, `balanced` and `resident` profiles. `low`
offloads the diffusion transformer between sampler calls and offloads the VAE;
`balanced` keeps the transformer resident and offloads the VAE. Samplers move an
offloaded transformer to the target device before the denoise loop and back to
CPU afterwards. `offload="sequential"` uses diffusers leaf-level hooks;
`offload="block"` and `offload="group"` use diffusers group offloading with
configurable `group_size`, CUDA stream prefetching, non-blocking transfers and
pinned host buffers. Component loaders request `low_cpu_mem_usage` from
diffusers/transformers where supported to reduce peak host memory during
initialization. `VAE Decode` and `VAE Encode` expose slicing/tiling
independently, clamp invalid overlap values, and retry once with tiling after an
out-of-memory failure. Attention slicing is offered as a compatibility/low-memory
fallback, not as a speed feature.

Per-node telemetry includes best-effort CPU RSS, peak allocated/reserved VRAM and
current free/total VRAM. Run history retains the metrics and displays peak VRAM
when present. On OOM the executor drops intermediate cache entries, collects
garbage, calls `torch.cuda.empty_cache()` only after dropping references, and
retries once only when cleanup actually reclaimed memory. Otherwise it raises
`NodeOOM` with the allocation message, memory breakdown and ranked suggestions.
`empty_cache()` does not free live tensors or another process's allocations, so
unchanged operations are never retried indefinitely. A retry re-runs the whole
node, so side-effecting nodes should tolerate a repeated attempt. A caught OOM
fails only the workflow; the disposable worker process is replaced on the next
run if it crashes or loses its CUDA context.

## Memory profiles and allocator configuration

The launcher applies allocator configuration before the backend/worker imports
torch or initializes CUDA:

| Option | Recommended use |
| --- | --- |
| `expandable_segments:True` | Linux default for long-running servers with variable latent sizes; reduces fragmentation |
| `max_split_size_mb:<N>` | Legacy fragmentation mitigation; superseded by expandable segments |
| `garbage_collection_threshold:<f>` | Shared GPUs or multiple workloads |
| `backend:cudaMallocAsync` | Explicit alternative allocator profile; native-only options are ignored and therefore rejected by the launcher |

`PYTORCH_ALLOC_CONF` is the supported variable (`PYTORCH_CUDA_ALLOC_CONF` remains
a deprecated alias). An existing value always wins over launcher defaults.
`--allocator cudaMallocAsync` cannot be combined with `--expandable-segments` or
`--pytorch-alloc-conf`. If reserved memory is much larger than allocated memory,
fragmentation is likely; if allocated is close to capacity, reduce the workload
or use a smaller/quantized checkpoint instead of expecting an allocator flag to
help.

## Local models

The model library lives at the repo root, not inside the active workspace:

```text
models/
├── yolo/             # .pt / .onnx YOLO checkpoints
├── torchvision/      # .pt / .pth / .safetensors detection/segmentation state dicts
├── classification/   # .pt / .pth / .safetensors / .onnx classifiers
├── diffusion/        # checkpoints, full diffusers repos and components
├── unet/             # legacy diffusion UNet component files
├── vae/              # standalone VAE files or folders
└── clip/             # standalone CLIP/T5/Qwen/Mistral text-encoder folders
```

Custom folders are discovered automatically. `COMPLEXUI_MODELS_DIR` can point
elsewhere (for tests, shared caches, or deployments). Model binaries are ignored
by git; only the folder placeholders and documentation are tracked.

Code-first selector syntax:

```python
local_model: ModelSelector = ModelSelector(type="yolo")
diffusion_model: ModelSelector = ModelSelector(
    type="diffusion",
    select="any",       # file | folder | any
    required=True,
)
vae_name: ModelSelector = ModelSelector(
    type="vae",
    select="any",
    required=True,
    include=["diffusion"],  # also search full diffusers repos
)
```

The marker is static DSL metadata and is never imported or executed. It becomes
a STRING-compatible model widget whose value is
`models/<type>/<path>`. Diffusers-style folders are listed when they contain a
`model_index.json`, `modular_model_index.json`, `config.json`, a supported
weight file, or a standard component folder. `include=[...]` adds extra
model-type folders to the picker and resolver, so a VAE selector can also list
full diffusers repos. At execution the executor resolves the selection to an
absolute file or folder path. Empty means use the predefined fallback unless
`required=True`. Missing, mismatched (file versus folder), or unsupported
selections fail during `validate_graph` before a run starts. File
`size:mtime_ns` and bounded recursive folder fingerprints are part of the node
cache key, and exported scripts receive the resolved absolute path.

Diffusion loading is local-only and component-oriented:

- `Load Diffusion Model` reads `model_index.json`/`modular_model_index.json`,
  loads the UNet/transformer and scheduler through `diffusers.AutoModel`, and
  falls back to the repository's `pipeline*.py` classes for custom components
  such as MOSS Image. It stores architecture, latent-space, capability and
  recommended-default metadata.
- `Load CLIP` loads a full repo or component folder and exposes an `encode`
  callable for SD1.5/SDXL, FLUX, Qwen, Z-Image, Krea 2, Ideogram 4, Fibo 2,
  Anima and MOSS. `Load CLIP (Dual)` combines CLIP-L + CLIP-G or CLIP + T5.
  Fibo 2 and MOSS Image accept structured/long prompts; raise
  `max_sequence_length` (up to 8192) when the default truncates them.
- `Load VAE` supports `AutoencoderKL`, `AutoencoderKLFlux2`,
  `AutoencoderKLQwenImage`, `AutoencoderKLQwenImage21` and
  `AutoencoderKLWan`, and exposes encode/decode callables with family-aware
  scaling, shift, patch packing and latent normalization.
- Architecture dropdowns list only implemented image families. Full repos and
  components must be selected with `type="auto"` or the matching family;
  mismatches fail with an actionable error. MOSS custom components require
  `allow_custom_code=True`.
- `KSampler`, `KSampler Advanced`, `Sampler Custom` and
  `Sampler Custom Advanced` route through a per-architecture adapter stored on
  the `MODEL` wrapper. Text-to-image, img2img, multi-reference editing and
  masked inpainting are implemented for SD1.5, SDXL, FLUX.1, FLUX.2, FLUX.3,
  Ideogram 4, Fibo 2, Anima, Qwen Image 2.1, Z-Image, Krea 2 and MOSS Image.
  The adapter handles family-specific latent packing, text-id/RoPE preparation,
  timestep scaling, dual-branch or skip-layer guidance, flow sigma schedules
  and final latent unpacking. Flow schedulers share the same parity rules:
  Anima and Fibo 2 use linear flow sigmas (Fibo 2 applies its resolution shift),
  Ideogram 4 uses its resolution-aware logit-normal schedule, and MOSS Image
  uses its shifted schedule snapped to the training grid; per-step skip-layer
  guidance windows follow the sampler step fraction.
- Recommended guidance defaults are exposed in `MODEL.recommended.guidance` and
  `MODEL.capabilities.guidance_default`: SD1.5/SDXL use CFG 7, FLUX.1 uses
  model guidance 3.5, FLUX.2 uses 4.0, FLUX.3 uses 1.0, Ideogram 4 uses 7.0,
  Fibo 2 uses 5.0, Anima uses 4.0, Qwen Image 2.1 uses 4.0, Z-Image uses 5.0,
  Krea 2 uses 4.5 and MOSS Image uses 3.0.
  `MODEL.capabilities.model_guidance` reports whether the loaded transformer
  accepts a guidance embedding; `dual_branch_guidance` and
  `skip_layer_guidance` mark Ideogram 4 and MOSS Image respectively.
- `Encode Reference Images` adds numbered optional `image1`…`image10` inputs,
  encodes them through the selected VAE and stores their latents on the
  conditioning. FLUX.1 Kontext, FLUX.2, FLUX.3, Fibo 2 and Qwen Image 2.1 consume
  those latents; the node provides up to ten references and other families reject
  them with an actionable error. References are never batch-expanded. Fibo 2
  reference latents condition the transformer; its text encoder receives text
  only, so edit prompts should describe the referenced images.
- `Model Guidance` attaches a guidance-embedding scale to a `MODEL` wrapper for
  guidance-distilled architectures such as FLUX.1 and FLUX.2. It can use the
  architecture's recommended guidance default or an explicit override; models
  without a guidance embedding reject it and use sampler CFG instead. Explicit
  model guidance can be combined with CFG on FLUX adapters; the legacy fallback
  (no `Model Guidance` node) continues to treat sampler CFG as the guidance
  embedding for models that require one.
- `CFG Guider` exposes `cfg`, `guidance_mode` (`inherit`/`override`), `guidance`
  and a fractional `start_percent`/`end_percent` CFG window. The window is
  consumed by `Sampler Custom Advanced`; outside the window CFG runs at 1.
- `MOSS Guidance` configures MOSS Image's negative-branch skip layers and
  optional skip-layer guidance (`slg_scale`, `slg_layers`, `slg_start`,
  `slg_stop`). The defaults reproduce MOSS's CFG with block 22 skipped in the
  negative branch.
- `Inpaint Conditioner` encodes `pixels` and either an `IMAGE` mask or the first
  (largest) candidate from a detection/segmentation `MASKS` value through the
  VAE, then returns positive/negative conditioning carrying the inpaint state.
  Four-channel UNets start from image latents and apply a latent noise mask,
  nine-channel inpaint UNets receive mask plus masked-latent channels, and FLUX
  Fill transformers receive packed mask/masked-image tokens. Ideogram 4, Fibo 2,
  Anima and MOSS Image start from the masked latent (Fibo 2 and Ideogram 4 also
  use it as context) because their reference pipelines have no dedicated mask
  channel. `Load Image` exposes an alpha/luminance mask output for direct
  wiring, and `Make Mask` generates a feathered rectangular or elliptical mask.
- `Conditioning Zero Out` clears `cond`/`pooled` tensors while preserving
  architecture metadata, reference images and inpaint state.

Conditional widgets use the same static marker style:

```python
width: int = ShowIf(default=512, size_mode="Resolution")
allow_upscale: bool = ShowIf(
    default=True,
    mode=["By Side", "By Megapixels"],
)
```

`ShowIf` keeps its literal `default` and stores the visibility conditions on the
widget schema. All conditions must match; a list means any value in that list.
Values may be strings, numbers, booleans or `None`. The frontend hides
non-matching widgets, but keeps them visible if the widget or a controlling
widget is connected. Hidden widgets still retain their saved values, and the
marker is removed before node execution/export.

Supported loading by template family:

| Family | Formats | Notes |
| --- | --- | --- |
| YOLO image/video | `.pt`, `.onnx` | Loaded directly by Ultralytics |
| torchvision classifier | `.pt`, `.pth`, `.safetensors`, `.onnx` | State dict via `weights_only=True` / SafeTensors, or ONNX Runtime; architecture widget still supplies preprocessing/metadata |
| torchvision detector/segmenter | `.pt`, `.pth`, `.safetensors` | State dict only; torchvision ONNX detection/segmentation is rejected (YOLO is the ONNX path) |

Model files are executable input. `.pt` may contain pickle payloads; loading a
YOLO `.pt` is not a sandbox even though workflow execution has resource
isolation. Only add trusted files, and prefer SafeTensors or ONNX where
possible. The API/UI only scans metadata and never imports models while listing.

Note: `train_test_split` now exposes `target_column` and `stratify` widgets.
Canvases saved against the old template keep their old node code, so after
upgrading those split nodes no longer match the official hash and need a
one-time approval (or replacement). Newly dragged split nodes use the new
template.

Note: `save_media` now has folder/filename template widgets, token expansion,
native folder Browse, 3-digit counters and batch indices. Canvases saved against
the old template keep their old node code and need a one-time approval (or
replacement); newly dragged nodes use the new template and bundled workflows
materialize the new code.

Note: `load_diffusion_model`, `load_vae`, `vae_decode` and `vae_encode` now
carry runtime optimization metadata and VAE slicing/auto-tiled retry widgets.
Existing canvases keep their old node code and need a one-time approval (or
replacement) to pick up the new behavior; new nodes and bundled workflows use
the updated templates.

Note: `resolution` and `empty_latent_image` now expose resolution/megapixel
modes and preset fields. Existing canvases keep their old node code and need a
one-time approval (or replacement); new nodes and bundled workflows use the
updated templates. `resize_image` and `resize_latent` replace the visible
per-mode resize nodes; the old templates remain hidden aliases so existing
canvases and workflows continue to run.

Save Media can write to user-selected absolute folders. The native folder
dialog and external artifact endpoints are disabled when the server accepts
non-loopback hosts, and external Open/Copy/Preview/Reveal actions only accept
paths recorded by a completed run. Keep the API loopback-only unless the
machine is trusted: node code already runs with the backend user's permissions.

## API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/parse` | POST | Parse node code into a `NodeSchema` (never executes code). |
| `/api/templates` | GET | Official templates with code, hash, category, description, `hidden` flag and socket schema. Hidden templates are still served (workflows depend on them); the UI keeps them out of the palette. |
| `/api/workflows` | GET | Bundled example workflows (nodes, edges, widget values, positions), ordered by category. |
| `/api/health` | GET | Liveness probe (`status`, `version`, template/workflow counts, `models_root`, accelerator/torch/CUDA/NVIDIA diagnostics, active run limits, attention/precision/memory/compile optimization settings and the quantization format registry) used by the launcher and Settings. |
| `/api/system/metrics` | GET | Live CPU utilization/cores, RAM usage, and NVIDIA GPU utilization/VRAM/temperature for the system monitor. Missing sensors return `null`. |
| `/api/models/types` | GET | List default and custom model type folders with file/folder counts. |
| `/api/models?type=<type>` | GET | List metadata for supported model files and diffusers-style folders (optionally scoped to one type), including `is_directory`, `parameter_count`, pre-quantization metadata and static memory estimates; never loads model contents. |
| `/api/estimate/workflow` | POST | Body `{"workflow": WorkflowGraph}`. Returns distinct model weights, resident/offload estimates and per-reference errors for typed `ModelSelector` widgets without executing node code. |
| `/api/status` | GET | Execution status (`idle`/`running` + `run_id`); clients call it after a WebSocket reconnect to resync. |
| `/api/approve` | POST | Approve custom/modified node code for this session. |
| `/api/export` | POST | Body `{"graph": ..., "format": "python" \| "notebook", "targets": [...] \| null}`. Returns `{filename, media_type, content}` for a standalone Python script or notebook; validates wiring/syntax but does not execute or require trust approval. |
| `/api/projects` | GET | List named projects (slug, display name, save time, node/edge/version counts). |
| `/api/projects` | POST | Body `{"name", "workflow", "overwrite": false}`. Create a project; `409` if the slug exists unless `overwrite` is true. |
| `/api/projects/{slug}` | GET | Load a project document. |
| `/api/projects/{slug}` | PUT | Save over a project; snapshots the previous document first (30 retained). |
| `/api/projects/{slug}` | DELETE | Move a project and its snapshots to `workspace/.trash/`. |
| `/api/projects/{slug}/versions` | GET | List retained version snapshots, newest first. |
| `/api/projects/{slug}/versions/{version}` | GET | Load one snapshot. |
| `/api/session` | GET | Return the latest recovery session or `null`. |
| `/api/session` | PUT | Persist the canvas on page hide/close (`workflow`, `project_slug`, `project_name`). |
| `/api/uploads` | GET | List uploaded assets (`id`, `name`, `inputs/...` path, format, `kind`, size, modified time), newest first. |
| `/api/uploads` | POST | Multipart `file` upload. Accepts CSV, TSV, Parquet, XLSX/XLSM, NDJSON/JSONL (100 MiB), and PNG/JPEG/WebP/GIF/BMP/TIFF or MP4/WebM/MOV/AVI/MKV/M4V media (500 MiB); sanitizes names and suffixes collisions. |
| `/api/uploads/{file_id}` | GET | Return metadata for one uploaded asset. |
| `/api/uploads/{file_id}/raw` | GET | Serve an uploaded image/video inline with range support and `nosniff`. |
| `/api/outputs` | GET | List generated images/videos from repo-root `outputs/`, including date subfolders; newest first (`limit` 1-500, default 200). `id` is the relative path inside `outputs/` (for example `2026-10-23/file.png`). |
| `/api/outputs/{file_id:path}/raw` | GET | Serve a generated image/video from `outputs/` inline with range support; `file_id` may include a date folder. |
| `/api/outputs/{file_id:path}` | DELETE | Delete one generated image/video. |
| `/api/outputs/{file_id:path}/open` | POST | Open one generated image/video with the platform default application (loopback-only; `403` when non-loopback hosts are allowed). |
| `/api/outputs/{file_id:path}/reveal` | POST | Reveal one generated image/video in the platform file manager (loopback-only). |
| `/api/artifacts/raw?path=...` | GET | Serve a media file recorded by a completed Save Media run (external destination; loopback-only). |
| `/api/artifacts/open` | POST | Open a run-produced external media file with the default application. |
| `/api/artifacts/reveal` | POST | Reveal a run-produced external media file in the platform file manager. |
| `/api/fs/pick-folder` | POST | Open the native folder dialog on the server desktop and return the selected absolute path (loopback-only; `409` if one is already open). |
| `/api/uploads/{file_id}/preview` | GET | Return up to `limit` table rows (default 20, max 100) plus `columns`, `rows`, and `truncated`; `422` for media. |
| `/api/uploads/{file_id}` | DELETE | Delete one uploaded asset from `inputs/`. |
| `/api/execute` | POST | Body `{"graph": ..., "targets": [...] \| null, "node_labels": {...} \| null}`. Pre-flights (422 invalid / 403 untrusted / 409 busy), optionally scopes execution to the ancestors of `targets`, then runs the graph; returns `202` with a `run_id`. Labels are retained for run-history display. Flattened subgraph nodes may carry `approval_id` so session approvals remain scoped to the original node id. |
| `/api/cancel` | POST | Ask the running execution to stop between nodes or during a dependency install (`202`; `409` when idle). Completed nodes stay cached. |
| `/api/runs` | GET | List the newest retained terminal run summaries (`limit` 1-200, default 50). |
| `/api/runs/{run_id}` | GET | Load one terminal run with order, labels, per-node statuses/timings/outputs and failure details. |
| `/api/runs/{run_id}` | DELETE | Delete one retained run record and its per-run log. |
| `/ws/execution` | WS | Streams `EXECUTION_STARTED`, per-node `NODE_STARTED`/`NODE_PROGRESS`/`NODE_COMPLETED`/`NODE_ERROR` (including `rich_outputs`), and `EXECUTION_COMPLETED`/`EXECUTION_FAILED`/`EXECUTION_CANCELLED` events as JSON. |

Requests enforce node-code, graph-size, and target-count caps; advertised
request bodies larger than 16 MiB are rejected with `413`, except
`/api/uploads`, which accepts up to 502 MiB so a 500 MiB media asset plus
multipart overhead fits. Upload contents are also capped per asset kind while
streaming. Non-upload bodies without `Content-Length` are buffered and capped
before routing; upload routes stay streamed so their per-asset caps apply. The
server binds
loopback by default and accepts only loopback `Host` headers and WebSocket
`Origin`s; set `COMPLEXUI_ALLOWED_HOSTS` (comma-separated) for custom hostnames
or use the launcher's `--allow-network` flag, which disables the guard and warns.
CORS defaults to the local Vite origins (`http://localhost:5173` and
`http://127.0.0.1:5173`); set `COMPLEXUI_CORS_ORIGINS` to a comma-separated list
for another frontend origin, or `0` to disable CORS middleware. The launcher
derives both the Vite proxy target and the CORS list from its port flags.
The API has no authentication and executes node code, so non-loopback exposure
is only appropriate on trusted networks.

Cancellation takes effect between nodes and while a dependency install runs.
Node code that is already executing is not interrupted by the cancel endpoint;
the per-node timeout terminates the isolated execution process instead.

## Execution isolation

Each run executes in a disposable child process. The parent keeps the HTTP and
WebSocket loop responsive, transfers the content-addressable cache and session
approvals into the child, and updates the parent cache only from a completed,
failed or cancelled terminal snapshot. `NODE_*` telemetry is forwarded live.
If the child crashes, the parent emits a `WorkerCrashed` terminal event and the
next run starts a fresh process. Cache entries whose values cannot be pickled
are intentionally not transferred (the next run recomputes them).

Defaults and environment overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMPLEXUI_NODE_TIMEOUT_SECONDS` | `600` | Maximum wall time from `NODE_STARTED` to completion; `0` disables |
| `COMPLEXUI_RUN_TIMEOUT_SECONDS` | `3600` | Maximum wall time for one run including installs; `0` disables |
| `COMPLEXUI_MAX_MEMORY_MB` | `12288` | Best-effort worker RSS limit; `0` disables |
| `COMPLEXUI_INPUT_DIR` | repo-root `inputs/` | Uploaded-input directory |
| `COMPLEXUI_OUTPUT_DIR` | repo-root `outputs/` | Generated-output directory |
| `COMPLEXUI_ACCELERATOR` | `auto` | `cpu`, `cuda`, or `auto` (NVML GPU detection); the launcher flag `--accelerator` overrides it |
| `COMPLEXUI_ATTENTION` | `auto` | `auto`, SDPA sub-backends, or optional `xformers`, `flash-attn`, `sageattention` (package required) |
| `COMPLEXUI_ATTENTION_BENCHMARK` | unset | Benchmark available attention backends once and select the fastest |
| `COMPLEXUI_MEMORY_PROFILE` | `auto` | `auto`, `low`, `balanced` or `resident`; `low` offloads the transformer |
| `COMPLEXUI_CUDA_ALLOCATOR` | `native` | `native` or `cudaMallocAsync`; applied before CUDA initialization |
| `COMPLEXUI_TORCH_COMPILE` | `off` | `off`, `auto`, `default`, `reduce-overhead` or `max-autotune` |
| `COMPLEXUI_TORCH_COMPILE_RECOMPILE_LIMIT` | `8` | Bounded Dynamo recompile limit (1-64) |
| `COMPLEXUI_COMPILE_CACHE_DIR` | `<workspace>/cache/torchinductor` | Stable Inductor/Triton compile cache root |
| `COMPLEXUI_EXPERIMENTAL_FP16_ACCUM` | unset | Unlocks the experimental FP16-accumulation option |
| `PYTORCH_ALLOC_CONF` | unset / Linux `expandable_segments:True` | Native allocator configuration; existing values win over launcher defaults |
| `COMPLEXUI_TORCH_CUDA_INDEX` | `https://download.pytorch.org/whl/cu130` | CUDA wheel index for CUDA torch installs |
| `COMPLEXUI_INSTALL_TIMEOUT_SECONDS` | `1200` | PEP 723 dependency install timeout |
| `COMPLEXUI_CORS_ORIGINS` | local Vite origins | Comma-separated allowed CORS origins; `0` disables CORS |

On Unix the worker also applies an address-space limit; the parent monitors
resident memory on all platforms. Timeout, memory-limit and crash failures are
persisted in run history with `NodeTimeout`, `RunTimeout`, `MemoryLimit` or
`WorkerCrashed` error types. Out-of-memory failures caught inside a node are
reported as `NodeOOM` with a memory breakdown and cleanup suggestions; the
executor retries only after cleanup reclaimed memory. Per-node CPU RSS and CUDA
peak metrics are attached to completed/error telemetry and retained in run
history. This isolation contains crashes and runaway
processes — it is **not** a filesystem/network security sandbox. Node code still
runs with the backend user's permissions, so only execute trusted code and keep
the API loopback-only unless the network is trusted. `File` widgets specifically
reject paths that resolve outside the workspace (including absolute paths and
symlink escapes).

## Workspace

The workspace root defaults to `backend/workspace`; set `COMPLEXUI_WORKSPACE`
to relocate it. Project documents use the same `.cuigraph.json` format as
frontend import/export:

```text
workspace/
├── workflows/<slug>.cuigraph.json   # current project documents
├── .versions/<slug>/<timestamp>.cuigraph.json
├── .trash/<slug>-<timestamp>/      # deleted project + its snapshots
├── runs/<run_id>.json              # 50 newest terminal run records
├── session.json                    # last close/visibility recovery
└── logs/                           # daily + per-run plain-text logs
```

Uploaded tables, images and videos live in the repo-root `inputs/` directory;
generated images, videos and artifacts normally live in repo-root `outputs/`.
Save Media defaults to a `%date%` folder there but can write to any absolute
folder the user picks. External files do not appear in the Outputs library;
they are available through the Save Media node's Open/Copy/Preview/Show-folder
actions for as long as the run that produced them is retained. Set
`COMPLEXUI_INPUT_DIR` or `COMPLEXUI_OUTPUT_DIR` to relocate the managed roots.

Writes are atomic, slugs are restricted to `[a-z0-9-]`, and project/version
reads revalidate the document before returning it. Overwrites keep the 30 newest
snapshots. Project documents may include canvas frame nodes (`isFrame`, `width`,
`height`) and child links (`parentId`); frames are UI-only, cannot carry edges,
must be valid parents, and are excluded from execution and exported artifacts.
They may also include reusable `subgraphs`: each definition has a unique id, a
shared input/output port contract, its own nodes/edges (including edges to the
reserved `__subgraph_input__` / `__subgraph_output__` managers), and manager
positions. Instance nodes store `isSubgraph` plus `subgraphId`. Validation
rejects missing/cyclic definitions, invalid manager edges, conflicting frame/
instance nodes, and unknown instance ports. The frontend flattens nested
instances before execution/export and scopes ids as `sub:<path>/<id>`.

Terminal runs are saved before their terminal WebSocket event is emitted, so the
history API is immediately consistent with the UI. The 50 newest records are
retained; each stores the scoped node order, labels, targets, per-node status and
timings, previews, errors, and at most 5 MiB of rich outputs per run (excess
payloads are omitted with a truncation flag). History write failures are logged
and never change the execution result. Run-record retention is independent of
the 14-day plain-text log retention. File logs follow `COMPLEXUI_WORKSPACE`
unless `COMPLEXUI_LOG_DIR` overrides them. File widgets store paths like
`inputs/iris.csv`; execution resolves them against the repo-root inputs
directory. Legacy workspace-relative paths remain readable. They may also read
generated files through `outputs/<folder>/<name>` (nested subfolders allowed),
while `OutputFile` names write under repo-root `outputs/`. File widgets still
reject absolute paths and symlink escapes outside the workspace, inputs, and
outputs roots. Exports resolve `OutputFile` widgets against the same generated
output directory and retain input/output paths, so keep those directories
available (or adjust the paths). Save Media's externally chosen destination is
not a `File` widget and does not relax those file-widget rules. `FolderPath`
widgets (used by `Load Images From Folder`) are a separate read-only input:
they accept an absolute user-picked directory or a repo-relative directory and
cannot write files. The loader rejects symlinked files and caps the batch size.

## Rich outputs

Each node runs with a capture context that collects:

- `print()`/stdout as text (flushed before the next `display()` call).
- `display(value)` for Polars/Pandas-style tables, matplotlib/seaborn figures,
  PIL images, Plotly figures, HTML, and `_repr_mimebundle_` / `_repr_png_` /
  `_repr_jpeg_` / `_repr_svg_` / `_repr_html_` objects.
- matplotlib figures created through `pyplot`, including `plt.show()` and
  figures left open when the node finishes.
- Plotly `figure.show()` (including modules imported after node start); calling
  both `display(fig)` and `fig.show()` emits one figure.

Capture is headless (`MPLBACKEND=Agg`, Plotly JSON renderer) and best effort:
a broken repr or oversized figure is dropped/truncated rather than failing the
node. Remote `stderr`/warnings stay in server logs. Outputs are included in
`NODE_COMPLETED`/`NODE_ERROR` telemetry as `rich_outputs`, cached with the node,
and never added to `EXECUTION_COMPLETED` (the per-node frames already carry
them). Matplotlib, seaborn and Plotly are not backend runtime dependencies;
nodes declare them in PEP 723 headers and the capturer detects loaded modules.

Per-node limits: 10 outputs, 12,000 text characters, 60,000 HTML characters,
a 20×30 table preview, 1 MB per image, 1.5 MB per Plotly figure, and 2.5 MB of
rich output total. A truncation notice is emitted when a limit drops content.
Rich outputs are runtime-only and are not exported in `.cuigraph.json`; terminal
run-history records retain a bounded snapshot for later inspection.

Start the server (CORS pre-configured for `http://localhost:5173`; override with
`COMPLEXUI_CORS_ORIGINS`):

```sh
uv run uvicorn app.main:app --reload
```

Production-style single-server run (serves the built frontend from `../frontend/dist`):

```sh
uv run uvicorn app.main:app --port 8000
# PowerShell: $env:COMPLEXUI_SERVE_DIST = "..\frontend\dist"
```

Operator launcher (preferred): `uv run python launch.py [dev|backend|frontend|prod|check]`,
the repo-root `start.cmd` (Windows) or `./start.sh` (macOS/Linux). Serving modes
sync but do not install dev/ML dependencies (`uv run --no-dev`); `check` syncs
the dev group. Frontend dependencies are installed with `npm ci` on a fresh
clone and refreshed with `npm install` when the lockfile hash changes.
Console logging honors `COMPLEXUI_LOG_LEVEL`
(default `INFO`), `COMPLEXUI_LOG_FORMAT` (`text` or newline-delimited `json`;
uvicorn access lines remain plain text), `NO_COLOR` / `FORCE_COLOR`, and
`--no-color` / `--log-level` / `--log-format`. Successful polling requests for
`/api/system/metrics`, `/api/status` and `/api/health` are filtered from the
access logger.

Plain-text logs are also written daily to `backend/workspace/logs/` (14 days
retained), with one file per execution under `backend/workspace/logs/runs/`.
Per-run logs are pruned to the newest 50 as new runs finish and are deleted with
their run record. Set `COMPLEXUI_LOG_DIR` to relocate them, or `0` to disable
files. The launcher accepts `--log-dir PATH` and `--no-log-file`.

Run the test suite from this directory:

```sh
uv run ruff check .    # lint (app/templates excluded: they use the node DSL)
uv run basedpyright    # static type check (app; templates excluded)
uv run pytest --cov=app --cov-report=term-missing
```

Coverage uses the configured 80% backend threshold when `--cov` is enabled;
the full verification gate and CI run it by default.

The default suite skips nodes whose optional runtimes are absent. To exercise
the full catalog (torch, torchvision, Ultralytics, OpenCV, XGBoost, LightGBM),
run the overlay gate:

```sh
uv run --with torch --with torchvision --with ultralytics \
  --with safetensors --with onnxruntime --with xgboost --with lightgbm pytest
```

Vision workflows use offline random weights/YAML models by default, so their
tests do not download checkpoints.
