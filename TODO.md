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

9. **Attention optimizations (SDPA, xFormers, Sage and more).**
   - Add an `attention` selector (`auto`, `sdpa`, `xformers`, `sage`) and
     optional `attention_slicing` to model/CLIP/VAE loaders.
   - `auto` uses PyTorch SDPA on supported setups and falls back safely.
     Feature-detect optional backends; never add them as hard runtime
     dependencies. Missing backends fail with an actionable message.
   - Keep backend choice visible in model previews and cache keys where it
     affects output. Do not silently mix mutually exclusive backends.
   - Add mocked/feature-detection tests and document compatibility and memory
     tradeoffs.

10. **Internal optimizations (memory, offloading, OOM handling).**
    - Add `offload` modes (`none`, `model`, `sequential`) to the pipeline-based
      loaders and expose VAE slicing plus attention slicing alongside the
      existing VAE tiling controls.
    - Set/document `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (and any
      configurable alternatives) for the worker process.
    - Catch CUDA OOM and allocation failures, clear the CUDA cache, run
      garbage collection and raise a concise node error with remedies (lower
      resolution/batch, enable tiling/offload, close other GPU apps). The
      server, UI and console must stay alive.
    - Ensure cleanup in `finally` paths; keep the existing wall-clock,
      node-timeout and process memory guards intact.
    - Add mocked-OOM and cleanup tests; document settings in the relevant
      READMEs.

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
