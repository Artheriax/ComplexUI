# Local model library

ComplexUI scans model metadata from this repo-root folder. Downloaded model
files are ignored by git.

```text
models/
├── yolo/             # Ultralytics checkpoints (.pt, .onnx)
├── torchvision/      # torchvision state dicts / SafeTensors
├── classification/   # torchvision classifiers (.pt/.pth/.safetensors/.onnx)
├── diffusion/        # checkpoints, full diffusers repos and components
├── unet/             # legacy standalone diffusion UNet components
├── vae/              # standalone VAE files or folders
└── clip/             # standalone CLIP/T5/Qwen/Mistral text encoders
```

A full diffusers repository can be selected directly by the diffusion loaders.
Component folders can also live at the top level of `diffusion/`, `vae/`,
`unet/` or `clip/`; use the loader's architecture `type` when automatic
detection is ambiguous. `Load CLIP (Dual)` combines two component selections
(CLIP-L + CLIP-G or CLIP + T5). Custom type folders are also discovered. Select
a model from the node's `ModelSelector` widget, or set `COMPLEXUI_MODELS_DIR` to
use a different root.

Only add trusted files. Model loading happens in node code and is not a
sandbox. Prefer SafeTensors or ONNX where possible; `.pt` files may contain
pickle payloads that execute code when loaded. MOSS/custom component loading
requires `allow_custom_code=True` on the loader node.
