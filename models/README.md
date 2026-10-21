# Local model library

ComplexUI scans model metadata from this repo-root folder. Downloaded model
files are ignored by git.

```text
models/
├── yolo/             # Ultralytics checkpoints (.pt, .onnx)
├── torchvision/      # torchvision state dicts / SafeTensors
└── classification/   # torchvision classifiers (.pt/.pth/.safetensors/.onnx)
```

Custom type folders are also discovered. Select a model from the node's
`ModelSelector` widget, or set `COMPLEXUI_MODELS_DIR` to use a different root.

Only add trusted files. Model loading happens in node code and is not a
sandbox. Prefer SafeTensors or ONNX where possible; `.pt` files may contain
pickle payloads that execute code when loaded.
