# TODO

Items 2 and 3 are shipped. The remaining items are expanded below with concrete
scope so they can be implemented without re-discovering the current state.

1. **QoL, improvements, and bug fixes (before adding new node families).**
   Existing-node polish comes first; avoid duplicating features that already
   shipped.
   - **Audit existing nodes** for empty/batch inputs, schema/widget drift,
     stale defaults, cache-key correctness, validation messages and preview
     accuracy. Add regression tests for every fix.
   - **Known issue:** `control_after_generated` is declared but unused in
     `ksampler.py`, `ksampler_advanced.py` and `sampler_custom.py`. Implement
     `fixed` / `increment` / `decrement` / `randomize` with batch-aware seed
     offsets and a post-run seed update in the graph store.
   - **Known issue:** Save Media `%model%` is a manually typed label. Wire it
     to real model provenance from the active diffusion model (see item 6).
   - **Known issue:** `Load Images From Folder` errors when a folder contains
     more images than `max_images`. Add recursive scan, extension filtering and
     an explicit truncate-vs-error option.
   - **Known issue:** Save Media copies video files and ignores `format` and
     `quality`. Add optional MP4/WebM transcode controls, or clearly document
     copy-only video behavior and disable misleading format controls.
   - **Image/vision QoL:** EXIF/ICC handling, optional alpha retention,
     animated WebP/GIF options, preview zoom/pan and batch navigation, mask
     overlay controls and better invalid-image errors.
   - **Error UX:** node failures should name the failing widget/input, suggest a
     fix and never leak tracebacks into the UI. Keep actionable messages in run
     history.
   - **Maintain a “Reported bugs” subsection** under this item: node,
     reproduction, expected behavior, regression test path.

2. **Done — new image-generation and vision nodes.**
   Shipped: five-input `Switch`, batched `Resize Image` / `Resize Latent` modes
   (dimensions, side, megapixels, multiplier), `Preview Mask`,
   `Load Images From Folder`, `Empty Latent Image` resolution/megapixel modes,
   batch-aware previews and batch index tokens. The model-based upscale
   sub-item is still missing from the catalog and is carried into item 4.

3. **Done — batch support.**
   Shipped: `IMAGE_BATCH` sockets, per-image execution for scalar `IMAGE`
   inputs, whole-batch `MEDIA`/`IMAGE_BATCH` handling, batch chips and aggregate
   previews, batch-aware progress, batch indices in Save Media, and
   batch-specific handle styling.

4. **More model support (image generation first).**
   - **Generalize the model contracts** beyond SD1.5/SDXL: add architecture
     metadata and loader/sampler paths for SD3/SD3.5, Flux.1, PixArt and Sana,
     including flow-matching schedulers and per-architecture recommended
     defaults.
   - **Extend loaders:** `Load Checkpoint`, `Load Diffusion Model`, `Load CLIP`
     and `Load VAE` must handle component folders and single-file weights for
     the new architectures (transformer/MMDiT denoisers, CLIP-L/G + T5,
     16-channel VAEs). Validate architecture and dtype combinations early.
   - **Conditioning:** extend `CLIP Text Encode` for multi-encoder models,
     pooled conditioning and model-specific prompt limits.
   - **Adapters/control:** add ControlNet, ControlLoRA, T2I-Adapter and
     IP-Adapter loader/apply nodes with per-adapter strengths.
   - **Upscale/depth/segmentation models:** add `Load Upscale Model` and
     `Upscale Image With Model` (Real-ESRGAN/SwinIR/spandrel, tiled and
     batch-aware), plus depth and SAM/SAM2 model nodes. This closes the missing
     model-upscale sub-item from item 2.
   - **Model library:** add default `models/` types for `clip`, `controlnet`,
     `ipadapter`, `upscale`, `sam`, `depth`, `lora` and `embedding`; keep
     type-aware previews, fingerprints and cache invalidation.
   - **Safety/perf:** local files only, no runtime downloads, and model
     fingerprints must participate in caching. Add offline/mocked contract
     tests and bundled example workflows where practical.

5. **Note and Markdown nodes (frontend-only annotations).**
   - Add `Note` and `Markdown` node kinds with no handles. They are annotations,
     not executable code: exclude them from execution, trust checks and
     Python/notebook export, like frames.
   - Persist text, format (`plain`/`markdown`), color, size and position as
     optional workflow fields. Additive save-format fields only.
   - Markdown editing and preview: headings, lists, code, links; sanitize
     rendered HTML with DOMPurify; cap text length.
   - Add palette category, inline editing, resize, undo/copy/paste/duplicate,
     context-menu actions, keyboard and screen-reader behavior.
   - Cover persistence, import/export, autosave, undo and rendering tests.

6. **Save Media naming and metadata (finish the remaining tokens).**
   - **Already shipped:** folder Browse with absolute paths, filename template,
     `%date%`, `%time%`, `%counter%`, `%model%`, `%width%`, `%height%`,
     `%batch%`, Clear/Reset, live “Will save as” preview, and Save/Preview/Open/
     Copy/Show-folder actions.
   - Define a provenance metadata envelope for `LATENT`, `IMAGE`/`IMAGE_BATCH`
     and video wrappers: seed, model, input name/folder, prompt, steps, cfg,
     sampler, scheduler, dimensions and batch index/size.
   - Populate metadata in loaders, samplers, VAE encode/decode and
     batch-selection nodes; preserve it through resize/upscale/crop
     transforms.
   - Enable `%seed%`, `%input_name%` and `%input_folder%` in backend expansion
     and frontend previews. Disabled buttons become active only when the
     upstream node can supply the value; otherwise show a clear warning.
   - Keep collision-safe counters, automatic batch indices and external-folder
     behavior. Add token-expansion, batch, metadata-propagation and
     unavailable-token tests.

7. **LoRA support.**
   - Add a `lora` model-library type and `Load LoRA` node: file picker,
     architecture detection, trigger words and file fingerprint.
   - Add `Apply LoRA`: inputs `MODEL` + `CLIP` + `LORA`, parameters
     `strength_model`, `strength_clip`, optional `fuse`; outputs patched
     `MODEL` + `CLIP`. Support chaining multiple LoRAs in order.
   - Load adapters into the UNet and text encoders with the correct diffusers
     prefixes; validate the LoRA base architecture against the model; report
     incompatible adapters clearly.
   - Does not break when used with standalone UNet/CLIP loads or checkpoints.
     Model/CLIP values remain non-cacheable; LoRA files contribute
     fingerprints.
   - Add synthetic SafeTensors tests (no downloads), UI strength controls and
     documentation of fuse/unfuse behavior.

8. **Embedding support.**
   - Add an `embedding` model-library type and `Load Embedding` node for
     `.pt`, `.bin` and `.safetensors` textual-inversion files; expose the token
     and vector count in the preview.
   - Add an optional `EMBEDDING` input to `CLIP Text Encode` (and/or a
     dedicated apply node) that injects embeddings with diffusers textual
     inversion. Support multiple embeddings and the SDXL second encoder.
   - Validate embedding architecture/token limits and keep the trigger token
     visible so users can include it in prompts.
   - Add tiny synthetic embedding fixtures for tests and document usage.

9. Attention, precision and quantization optimizations

**P0 shipped:** capability-checked PyTorch SDPA backends, storage/compute/
accumulation tracking, guarded FP16 accumulation, opt-in `torch.compile` with
eager fallback, channels-last, launcher flags and diagnostics. **P1 attention
shipped:** optional SageAttention 2/2++, FlashAttention 2/4, xFormers and Comfy
Kitchen INT8/SOL adapters with package/device gating and safe SDPA fallback,
plus an attention benchmark mode. The launcher backend is the global default and
`Attention Backend` applies a per-MODEL override only during the diffusion
forward. SageAttention 2.2 post6 + triton-windows was validated end-to-end on an
RTX 5080 (sm120) SDXL workflow. **P1 quantization metadata shipped:** Comfy
`.comfy_quant`, native GGUF headers, sidecar JSON, legacy scaled FP8 and the
format registry feed Models Library tags, `/api/health` and static memory
estimates. **Pre-quantized component loading shipped:** diffusers/transformers
component folders load Comfy `int8_tensorwise`, `convrot_w4a4`, FP8, MXFP8 and
NVFP4 weights through `comfy_kitchen`; INT8/ConvRot use native kernels while the
other formats keep quantized storage with a dequantized linear fallback. Single-
file SD1.5/SDXL quantized checkpoints are converted to diffusers and the UNet is
re-patched with native INT8/ConvRot weights; CLIP components stay dequantized.
First-class GGUF weight loading and native single-file CLIP patching remain
deferred; quantized VAEs are explicitly out of scope. CUDA Graphs and GGUF
weight loading remain P2.

9.1 Attention backend registry

- Replace the fixed attention selector with a capability-based registry.
- Investigate PyTorch SDPA, FlashAttention 2/3, xFormers, SageAttention 2/2++, SageAttention 3, FlexAttention, Comfy Kitchen INT8/SOL attention, and split/sub-quadratic fallbacks.
- Separate universal attention APIs from specialized kernels and model-specific attention processors.
- Detect GPU architecture, PyTorch/CUDA versions, optional packages, dtype, masks, attention shape and model compatibility.
- Provide `auto`, explicit backend selection and an optional benchmark mode.
- Make `auto` fall back safely; explicit backend requests must fail with an actionable error if unavailable.
- Log the selected backend and reason. Include effective settings in diagnostics and cache keys where they change model behavior.
- Do not add optional attention packages as mandatory runtime dependencies.
- Test unsupported devices, missing packages, masks, cross-attention, fallback paths and output correctness.

9.2 Precision and quantization

- Track weight storage dtype, compute dtype and accumulation behavior separately.
- Support architecture- and hardware-appropriate FP32, FP16 and BF16 inference paths.
- Investigate FP8, MXFP8, INT8/W8A8, ConvRot, NF4 and supported 4-bit formats through compatible runtimes.
- Keep quantization format, quantization configuration, kernel backend and quantized checkpoint metadata explicit.
- Do not treat INT8, NF4, INT4, MXFP8 and NVFP4 as interchangeable formats.
- Add FP16 accumulation only as an explicit experimental option with numerical validation and overflow tests.
- Validate model compatibility, hardware requirements and dtype combinations before inference.
- Do not quantize arbitrary weights automatically unless the selected method and model architecture are supported.
- Add mocked contract tests and small local fixtures; avoid model downloads in the test suite.
- Document expected speed, VRAM, quality and compatibility trade-offs.

**Remaining quantized work (as of 2026-10-29):**

- [ ] GGUF weight loading. Metadata detection, variant tagging and capability
  gating are done. Still needed: read GGUF tensor types, dequantize with the
  `gguf` runtime, restore `comfy.gguf.orig_shape.*` tensors, choose a component
  config per architecture, and test against synthetic GGUF fixtures. No real
  GGUF checkpoint is available in the repo.
- [ ] Native FP8, MXFP8 and NVFP4 linear kernels. These formats currently keep
  packed storage but fall back to dequantization for each linear forward.
  Either wire Comfy Kitchen `scaled_mm_*`/FP8 kernels with activation
  quantization, or explicitly keep and document the slower fallback.
- [ ] Native single-file CLIP/CLIP-Dual patching. Single-file SD1.5/SDXL UNets
  are re-patched with native INT8/ConvRot weights, but CLIP components from the
  same checkpoint remain dequantized. Diffusers component-folder CLIP already
  supports native INT8/ConvRot.
- [ ] `input_scale` consumption for FP8/NVFP4. `pre_quant_scale` is applied;
  calibrated `input_scale` values are currently ignored with a warning because
  the kernels use dynamic activation scaling.
- [ ] Single-file compatibility for architectures other than SD1.5/SDXL
  (FLUX, SD3/3.5, Qwen Image, etc.). Those paths currently require diffusers
  component folders.
- [ ] Real quantized checkpoint validation. All current tests are synthetic
  contract tests; no actual INT8/ConvRot/FP8/NVFP4/GGUF checkpoint has been
  loaded end-to-end.
- [ ] Workflow memory-estimate UI. The server-side `POST /api/estimate/workflow`
  exists, but no canvas placement or client surface has been chosen.
- [ ] CUDA Graphs and the full benchmark suite. `torch.compile` modes including
  `max-autotune-no-cudagraphs` are available; explicit graph capture and the
  §7 benchmark protocol remain deferred.
- [ ] Optional: quantized VAE loading is explicitly **not required** and should
  stay out of scope.

**Implementation map for other agents:**

| Area | Primary code | Contract tests |
| --- | --- | --- |
| Metadata, format registry, gates | `backend/app/quantization.py` | `backend/tests/test_quantization.py` |
| Component-folder quantized loading | `backend/app/quantized_loading.py` | `backend/tests/test_quantized_loading.py` |
| Single-file conversion and native UNet patch | `backend/app/single_file_quantization.py` | `backend/tests/test_single_file_quantization.py` |
| Loader nodes | `backend/app/templates/load_checkpoint.py`, `load_diffusion_model.py`, `load_clip.py`, `load_clip_dual.py` | `test_diffusion_nodes.py`, `test_component_loaders.py` |
| Model/workflow estimates | `backend/app/memory_estimate.py`, `workflow_estimate.py`, `/api/estimate/workflow` in `main.py` | `test_memory_estimate.py` |
| Attention registry and scoped overrides | `backend/app/attention_dispatch.py`, `optimizations.py`, `templates/attention_backend.py` | `test_attention_runtime.py`, `test_optimizations.py` |
| Compile/launch flags | `backend/launch.py`, `backend/app/optimizations.py` | `test_launch.py`, `test_compile_runtime.py` |

**Reference sources:**

- Comfy quantization format:
  https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md
- Comfy quant format registry and kernels:
  https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/quant_ops.py
- Comfy mixed-precision module loading:
  https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ops.py
- Comfy Kitchen runtime (INT8/ConvRot/FP8/MXFP8/NVFP4 layouts, SOL/INT8 attention):
  https://github.com/Comfy-Org/comfy-kitchen
- City96 GGUF loader and conversion tools:
  https://github.com/city96/ComfyUI-GGUF and
  https://github.com/city96/ComfyUI-GGUF/blob/main/tools/convert.py
- GGUF Python reader/dequantizer: https://pypi.org/project/gguf/
  and GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Diffusers single-file loading:
  https://huggingface.co/docs/diffusers/main/en/api/loaders/single_file
  and key conversion helpers:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/single_file_utils.py
- torchao quantization overview:
  https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html
- SageAttention: https://github.com/thu-ml/SageAttention
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- xFormers: https://github.com/facebookresearch/xformers
- PyTorch SDPA:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch `torch.compile` and CUDA graphs:
  https://pytorch.org/docs/stable/generated/torch.compile.html and
  https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
- PyTorch CUDA allocator configuration:
  https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf

**Validated local environment (2026-10-29):** torch `2.14.1+cu130`,
diffusers `0.42.1`, transformers `5.20.2`, comfy-kitchen `0.2.40`,
torchao `0.18.0`, gguf `0.19.0`, SageAttention
`2.2.0+cu130torch2.10.0andhigher.post6`, triton-windows `3.9.0.post30`.
`flash-attn-4==4.0.0b37` resolves and imports metadata, but `flash_attn.cute`
is unavailable on Windows because its CuTeDSL path imports `fcntl`; treat FA4
as Linux-only until that changes. Optional packages are installed in the
backend venv only and are not in `pyproject.toml`; `uv sync --exact` may remove
them.

9.3 Compilation and execution acceleration

- Investigate optional `torch.compile`, CUDA Graphs and channels-last memory format where supported.
- Make compilation opt-in, handle compilation failures safely, and distinguish first-run compilation time from steady-state inference.
- Avoid unbounded compilation caches or recompilation when input shapes vary.
- Benchmark each optimization on representative image-generation, image-editing and video workflows.
- Keep eager execution available as the compatibility fallback.

10. Memory management, offloading and OOM handling

**P0 shipped:** model-level offload, independent VAE slicing/tiling with
auto-tiled retry, CUDA allocator launch configuration, node-boundary OOM cleanup
and actionable errors, per-node VRAM/CPU metrics and run-history display.
**P1 shipped:** sequential/leaf, block and group offload through diffusers hooks
with streamed prefetch and pinned host buffers, plus `low_cpu_mem_usage` loader
hints. Full meta-device/sharded loading and a custom faulting allocator remain
P2.

- Provide model-level, sequential and group/block offloading where supported by the architecture and runtime.
- Expose VAE slicing and tiling independently; preserve existing VAE tiling behavior.
- Keep attention slicing as a compatibility/low-memory fallback rather than assuming it is faster.
- Investigate low-memory and sharded checkpoint loading to reduce peak memory during model initialization.
- Add documented CUDA allocator configuration, including supported `PYTORCH_ALLOC_CONF` options and `expandable_segments`, applied before CUDA initialization.
- Treat allocator backend selection, such as `cudaMallocAsync` where supported, as a separate, validated configuration.
- Track peak allocated/reserved VRAM, CPU RAM, load time, inference time and offloading overhead when available.
- Catch CUDA OOM and relevant allocation failures at node boundaries; clean up temporary resources and return concise, actionable errors.
- Explain that clearing the CUDA cache does not free live tensors; do not retry unchanged operations indefinitely.
- Keep the server and UI alive where possible; safely restart a worker only when required.
- Preserve existing process memory limits, node timeouts and wall-clock guards.
- Add mocked-OOM, cleanup, fallback and worker-recovery tests.
- Document hardware/backend compatibility and memory/speed trade-offs in the relevant READMEs.

11. **Automatic template updates for uncustomized nodes.**
    - Persist `templateCodeHash` (the template version a node was created from)
      as an optional field on node data and saved workflow nodes. Additive
      change; no workflow format bump.
    - On template refresh, update a node only if its current code hash matches
      the stored baseline. Replace code/schema with the new template, merge
      widget defaults, preserve position, size, color, user labels and genuine
      edits, and reconcile/drop incompatible edges with a visible warning.
    - Update the baseline after a successful upgrade. Legacy nodes without a
      baseline are treated as customized and never overwritten automatically.
    - Exclude frames, subgraphs, notes, scratch nodes and user templates.
      Handle nodes inside subgraph definitions and open tabs.
    - Show “N nodes updated / M customized nodes kept”; add persistence,
      autosave, subgraph and rollback/undo tests.

**Per-item done criteria:** update the owning README, add contract/regression
tests, keep security and path rules intact, and run the full `start.cmd check`
gate plus relevant Playwright/ML-overlay checks before marking an item done.
