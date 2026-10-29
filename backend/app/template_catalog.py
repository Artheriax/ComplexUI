"""Display metadata for the official node template catalog.

Titles, categories, one-line descriptions and palette visibility (``hidden``
for example dataset loaders / synthetic dev nodes) live here so the API can
serve them and the frontend never needs to hardcode per-template copy. The
registry still discovers ``templates/*.py``; coverage is enforced by
``tests/test_template_catalog.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CATEGORIES", "TemplateMeta", "TEMPLATE_META", "template_meta"]

CATEGORIES: tuple[str, ...] = (
    "Values",
    "Data Loaders",
    "Data Cleaning",
    "EDA",
    "Preprocessing",
    "Imbalanced Data",
    "Models",
    "Evaluation",
    "PyTorch",
    "Computer Vision",
    "Image Generation",
)

FALLBACK_CATEGORY = "Other"


@dataclass(frozen=True, slots=True)
class TemplateMeta:
    title: str
    category: str
    description: str
    hidden: bool = False


TEMPLATE_META: dict[str, TemplateMeta] = {
    "int_value": TemplateMeta(
        "Int Value",
        "Values",
        "Loose integer with adjustable step",
    ),
    "float_value": TemplateMeta(
        "Float Value",
        "Values",
        "Loose float with adjustable step",
    ),
    "bool_value": TemplateMeta(
        "Bool Value",
        "Values",
        "Loose boolean value",
    ),
    "string_value": TemplateMeta(
        "String Value",
        "Values",
        "Loose string value",
    ),
    "switch": TemplateMeta(
        "Switch (5 Inputs)",
        "Values",
        "Pass one of five named inputs through",
    ),
    "load_iris": TemplateMeta(
        "Load Iris Dataset",
        "Data Loaders",
        "Polars DataFrame with 150 iris rows",
        hidden=True,
    ),
    "load_wine": TemplateMeta(
        "Load Wine Dataset",
        "Data Loaders",
        "178 wines, 13 features, 3 classes",
        hidden=True,
    ),
    "load_breast_cancer": TemplateMeta(
        "Load Breast Cancer Dataset",
        "Data Loaders",
        "569 samples, 30 features, binary target",
        hidden=True,
    ),
    "load_diabetes": TemplateMeta(
        "Load Diabetes Dataset",
        "Data Loaders",
        "442 patients, 10 features, continuous target",
        hidden=True,
    ),
    "load_digits": TemplateMeta(
        "Load Digits Dataset",
        "Data Loaders",
        "8x8 digit images flattened to 64 pixel features",
        hidden=True,
    ),
    "make_classification_dataset": TemplateMeta(
        "Synthetic Classification",
        "Data Loaders",
        "make_classification with configurable shape",
        hidden=True,
    ),
    "make_messy_dataset": TemplateMeta(
        "Messy Demo Dataset",
        "Data Loaders",
        "Nulls, duplicates and whitespace for cleaning demos",
        hidden=True,
    ),
    "load_csv": TemplateMeta(
        "Load CSV",
        "Data Loaders",
        "Read a CSV file by path",
    ),
    "load_parquet": TemplateMeta(
        "Load Parquet",
        "Data Loaders",
        "Read a Parquet file by path",
    ),
    "load_excel": TemplateMeta(
        "Load Excel",
        "Data Loaders",
        "Read an .xlsx sheet by path (fastexcel)",
    ),
    "load_json": TemplateMeta(
        "Load NDJSON",
        "Data Loaders",
        "Read newline-delimited JSON by path",
    ),
    "drop_missing": TemplateMeta(
        "Drop Missing Rows",
        "Data Cleaning",
        "Remove rows with nulls (any/all subset)",
    ),
    "fill_missing": TemplateMeta(
        "Fill Missing Values",
        "Data Cleaning",
        "Mean/median/constant imputation per column",
    ),
    "drop_duplicates": TemplateMeta(
        "Drop Duplicate Rows",
        "Data Cleaning",
        "Unique rows, optionally on a subset",
    ),
    "drop_columns": TemplateMeta(
        "Drop Columns",
        "Data Cleaning",
        "Remove comma-separated columns",
    ),
    "filter_rows": TemplateMeta(
        "Filter Rows",
        "Data Cleaning",
        "Keep rows matching a column predicate",
    ),
    "cast_numeric": TemplateMeta(
        "Cast To Numeric",
        "Data Cleaning",
        "Coerce selected columns to int or float",
    ),
    "clip_outliers": TemplateMeta(
        "Clip Outliers",
        "Data Cleaning",
        "Winsorise numeric columns to quantile bounds",
    ),
    "trim_strings": TemplateMeta(
        "Trim Strings",
        "Data Cleaning",
        "Strip, lowercase and collapse whitespace",
    ),
    "describe_dataframe": TemplateMeta(
        "Describe DataFrame",
        "EDA",
        "Shape, dtypes, nulls and numeric summary",
    ),
    "missing_values_report": TemplateMeta(
        "Missing Values Report",
        "EDA",
        "Per-column null counts and fractions",
    ),
    "correlation_matrix": TemplateMeta(
        "Correlation Matrix",
        "EDA",
        "Pearson/Spearman numeric correlations",
    ),
    "value_counts": TemplateMeta(
        "Value Counts",
        "EDA",
        "Top values and shares for a column",
    ),
    "class_balance": TemplateMeta(
        "Class Balance",
        "EDA",
        "Class counts, shares and imbalance ratio",
    ),
    "histogram": TemplateMeta(
        "Histogram",
        "EDA",
        "Distribution of a numeric column with optional grouping",
    ),
    "scatter_plot": TemplateMeta(
        "Scatter Plot",
        "EDA",
        "Numeric relationship with colour and size encoding",
    ),
    "box_plot": TemplateMeta(
        "Box Plot",
        "EDA",
        "Quartiles and outliers, optionally split by a group",
    ),
    "bar_chart": TemplateMeta(
        "Bar Chart",
        "EDA",
        "Category counts or aggregated numeric values",
    ),
    "correlation_heatmap": TemplateMeta(
        "Correlation Heatmap",
        "EDA",
        "Heatmap of Pearson or Spearman correlations",
    ),
    "train_test_split": TemplateMeta(
        "Train / Test Split",
        "Preprocessing",
        "Split into train/test Datasets",
    ),
    "dataframe_to_dataset": TemplateMeta(
        "DataFrame To Dataset",
        "Preprocessing",
        "Convert DataFrame columns into an (X, y) Dataset",
    ),
    "drop_target": TemplateMeta(
        "Drop Target Column",
        "Preprocessing",
        "Features-only DataFrame without the target",
    ),
    "one_hot_encode": TemplateMeta(
        "One-Hot Encode",
        "Preprocessing",
        "Dummy columns for categorical/text features",
    ),
    "standard_scaler": TemplateMeta(
        "Standard Scaler",
        "Preprocessing",
        "Z-score scale numeric features (target excluded)",
    ),
    "minmax_scaler": TemplateMeta(
        "Min-Max Scaler",
        "Preprocessing",
        "Scale numeric features into the 0-1 range",
    ),
    "pca_reduce": TemplateMeta(
        "PCA Reduce",
        "Preprocessing",
        "Project a Dataset onto principal components",
    ),
    "select_k_best": TemplateMeta(
        "Select K Best",
        "Preprocessing",
        "Keep top-k features by f_classif/f_regression",
    ),
    "random_oversample": TemplateMeta(
        "Random Oversample",
        "Imbalanced Data",
        "Duplicate minority-class rows until balanced",
    ),
    "random_undersample": TemplateMeta(
        "Random Undersample",
        "Imbalanced Data",
        "Drop majority-class rows until balanced",
    ),
    "smote_resample": TemplateMeta(
        "SMOTE Resample",
        "Imbalanced Data",
        "Synthesise minority samples with SMOTE",
    ),
    "knn_train": TemplateMeta(
        "KNN Classifier",
        "Models",
        "KNeighborsClassifier (n_neighbors)",
    ),
    "linear_regression": TemplateMeta(
        "Linear Regression",
        "Models",
        "Ordinary least squares regression",
    ),
    "logistic_regression": TemplateMeta(
        "Logistic Regression",
        "Models",
        "Linear classifier with L2 regularisation",
    ),
    "ridge_regression": TemplateMeta(
        "Ridge Regression",
        "Models",
        "L2-regularised linear regression",
    ),
    "lasso_regression": TemplateMeta(
        "Lasso Regression",
        "Models",
        "L1-regularised linear regression",
    ),
    "decision_tree_classifier": TemplateMeta(
        "Decision Tree Classifier",
        "Models",
        "CART classifier with depth control",
    ),
    "decision_tree_regressor": TemplateMeta(
        "Decision Tree Regressor",
        "Models",
        "CART regressor with depth control",
    ),
    "random_forest_classifier": TemplateMeta(
        "Random Forest Classifier",
        "Models",
        "Bagged ensemble of decision trees",
    ),
    "random_forest_regressor": TemplateMeta(
        "Random Forest Regressor",
        "Models",
        "Bagged tree ensemble for regression",
    ),
    "gradient_boosting_classifier": TemplateMeta(
        "Gradient Boosting Classifier",
        "Models",
        "Boosted shallow trees for classification",
    ),
    "gradient_boosting_regressor": TemplateMeta(
        "Gradient Boosting Regressor",
        "Models",
        "Boosted shallow trees for regression",
    ),
    "svm_classifier": TemplateMeta(
        "SVM Classifier",
        "Models",
        "Support vector classifier (kernel configurable)",
    ),
    "gaussian_nb_classifier": TemplateMeta(
        "Gaussian Naive Bayes",
        "Models",
        "Probabilistic baseline classifier",
    ),
    "kmeans_clustering": TemplateMeta(
        "K-Means Clustering",
        "Models",
        "Lloyd k-means with k-means++ init",
    ),
    "xgb_classifier": TemplateMeta(
        "XGBoost Classifier",
        "Models",
        "Extreme gradient boosting (CPU hist)",
    ),
    "xgb_regressor": TemplateMeta(
        "XGBoost Regressor",
        "Models",
        "Extreme gradient boosting for regression",
    ),
    "lgbm_classifier": TemplateMeta(
        "LightGBM Classifier",
        "Models",
        "Leaf-wise gradient boosting classifier",
    ),
    "lgbm_regressor": TemplateMeta(
        "LightGBM Regressor",
        "Models",
        "Leaf-wise gradient boosting regressor",
    ),
    "feature_importance": TemplateMeta(
        "Feature Importance",
        "Models",
        "Rank model importances or linear coefficients",
    ),
    "evaluate_metrics": TemplateMeta(
        "Evaluate Classifier",
        "Evaluation",
        "Accuracy + confusion matrix on a test set",
    ),
    "evaluate_regression": TemplateMeta(
        "Evaluate Regression",
        "Evaluation",
        "MAE, RMSE and R2 on a test set",
    ),
    "regression_diagnostics": TemplateMeta(
        "Regression Diagnostics",
        "Evaluation",
        "Actual-vs-predicted and residual plots with R2",
    ),
    "confusion_matrix": TemplateMeta(
        "Confusion Matrix",
        "Evaluation",
        "Annotated class confusion heatmap for a model",
    ),
    "cross_validate_classifier": TemplateMeta(
        "Cross-Validate Classifier",
        "Evaluation",
        "K-fold accuracy mean and spread",
    ),
    "tensor_dataset": TemplateMeta(
        "Synthetic Tensor Dataset",
        "PyTorch",
        "Synthetic 2-class tensors (torch)",
        hidden=True,
    ),
    "mlp_train": TemplateMeta(
        "MLP Trainer (Torch)",
        "PyTorch",
        "Train a small MLP, returns loss curve",
    ),
    "training_history": TemplateMeta(
        "Training History",
        "PyTorch",
        "Metric curve with optional moving-average smoothing",
    ),
    "evaluate_torch": TemplateMeta(
        "Evaluate Torch Model",
        "PyTorch",
        "Accuracy + loss on tensor data",
    ),
    "dataframe_to_tensors": TemplateMeta(
        "DataFrame To Tensors",
        "PyTorch",
        "Classification frame to float32/long tensors",
    ),
    "torchvision_classifier": TemplateMeta(
        "Torchvision Classifier",
        "Computer Vision",
        "Select a torchvision classification backbone",
    ),
    "torchvision_detector": TemplateMeta(
        "Torchvision Detector",
        "Computer Vision",
        "Select a torchvision object detection backbone",
    ),
    "torchvision_segmenter": TemplateMeta(
        "Torchvision Segmenter",
        "Computer Vision",
        "Select a torchvision semantic segmentation backbone",
    ),
    "classify_image": TemplateMeta(
        "Classify Image",
        "Computer Vision",
        "Top-k ImageNet predictions from a torchvision model",
    ),
    "detect_objects": TemplateMeta(
        "Detect Objects",
        "Computer Vision",
        "Bounding boxes plus an annotated image",
    ),
    "segment_image": TemplateMeta(
        "Segment Image",
        "Computer Vision",
        "Semantic masks plus a colored overlay",
    ),
    "yolo_detector": TemplateMeta(
        "YOLO Detector",
        "Computer Vision",
        "Select an Ultralytics YOLO detection checkpoint",
    ),
    "yolo_segmenter": TemplateMeta(
        "YOLO Segmenter",
        "Computer Vision",
        "Select an Ultralytics YOLO segmentation checkpoint",
    ),
    "yolo_detect": TemplateMeta(
        "YOLO Detect Objects",
        "Computer Vision",
        "Bounding boxes plus an annotated image (Ultralytics)",
    ),
    "yolo_segment": TemplateMeta(
        "YOLO Segment Image",
        "Computer Vision",
        "Instance masks plus a colored overlay (Ultralytics)",
    ),
    "yolo_detect_video": TemplateMeta(
        "YOLO Detect Video",
        "Computer Vision",
        "Annotate a video with YOLO boxes and a run summary",
    ),
    "yolo_segment_video": TemplateMeta(
        "YOLO Segment Video",
        "Computer Vision",
        "Annotate a video with YOLO masks and a run summary",
    ),
    "classify_video": TemplateMeta(
        "Classify Video",
        "Computer Vision",
        "Average torchvision predictions across sampled frames",
    ),
    "load_video": TemplateMeta(
        "Load Video",
        "Computer Vision",
        "Read video metadata and a poster frame",
    ),
    "extract_video_frame": TemplateMeta(
        "Extract Video Frame",
        "Computer Vision",
        "Read one frame from a video into an image",
    ),
    "load_image": TemplateMeta(
        "Load Image",
        "Computer Vision",
        "Read a workspace image into RGB plus an alpha/luminance mask",
    ),
    "load_images_from_folder": TemplateMeta(
        "Load Images From Folder",
        "Computer Vision",
        "Load a folder of images as an ordered image batch",
    ),
    "preview_media": TemplateMeta(
        "Preview Media",
        "Computer Vision",
        "Preview an image or video without writing a file",
    ),
    "preview_mask": TemplateMeta(
        "Preview Mask",
        "Computer Vision",
        "Preview MASKS without modifying them",
    ),
    "save_media": TemplateMeta(
        "Save Media",
        "Computer Vision",
        "Write media with a configurable folder and filename template",
    ),
    "make_demo_image": TemplateMeta(
        "Synthetic Demo Image",
        "Computer Vision",
        "Generated shapes for offline vision workflow tests",
        hidden=True,
    ),
    "make_demo_video": TemplateMeta(
        "Synthetic Demo Video",
        "Computer Vision",
        "Generated moving shapes for offline video workflow tests",
        hidden=True,
    ),
    "load_checkpoint": TemplateMeta(
        "Load Checkpoint",
        "Image Generation",
        "Load a local SD1.5 or SDXL SafeTensors checkpoint and adapter metadata",
    ),
    "clip_text_encode": TemplateMeta(
        "CLIP Text Encode",
        "Image Generation",
        "Encode a prompt with CLIP; SDXL adds pooled embeddings",
    ),
    "conditioning_zero_out": TemplateMeta(
        "Conditioning Zero Out",
        "Image Generation",
        "Zero conditioning tensors for unconditional guidance",
    ),
    "encode_reference_images": TemplateMeta(
        "Encode Reference Images",
        "Image Generation",
        "Encode up to ten numbered reference images into conditioning for editing",
    ),
    "model_guidance": TemplateMeta(
        "Model Guidance",
        "Image Generation",
        "Attach a guidance embedding scale to a diffusion model",
    ),
    "attention_backend": TemplateMeta(
        "Attention Backend",
        "Image Generation",
        "Select a capability-checked PyTorch SDPA attention backend",
    ),
    "model_precision": TemplateMeta(
        "Model Precision",
        "Image Generation",
        "Set diffusion compute dtype and accumulation mode",
    ),
    "compile_model": TemplateMeta(
        "Compile Model",
        "Image Generation",
        "Opt in to torch.compile with eager fallback",
    ),
    "model_memory": TemplateMeta(
        "Model Memory",
        "Image Generation",
        "Set residency, offload, attention slicing and channels-last",
    ),
    "moss_guidance": TemplateMeta(
        "MOSS Guidance",
        "Image Generation",
        "Configure MOSS Image negative skip layers and skip-layer guidance",
    ),
    "inpaint_conditioner": TemplateMeta(
        "Inpaint Conditioner",
        "Image Generation",
        "Encode pixels plus a mask or segmentation masks into inpainting conditioning",
    ),
    "make_mask": TemplateMeta(
        "Make Mask",
        "Image Generation",
        "Generate a feathered rectangular or elliptical mask",
    ),
    "empty_latent_image": TemplateMeta(
        "Empty Latent Image",
        "Image Generation",
        "Create a latent from explicit dimensions, aspect ratio or megapixels",
    ),
    "ksampler": TemplateMeta(
        "KSampler",
        "Image Generation",
        "Sample a latent with CFG, ComfyUI-style samplers and schedules",
    ),
    "vae_decode": TemplateMeta(
        "VAE Decode",
        "Image Generation",
        "Decode a latent batch into images",
    ),
    "image_from_batch": TemplateMeta(
        "Image From Batch",
        "Image Generation",
        "Select one image from an IMAGE_BATCH",
    ),
    "resolution": TemplateMeta(
        "Resolution",
        "Image Generation",
        "Compute width and height from aspect ratio, megapixels and divisibility",
    ),
    "basic_scheduler": TemplateMeta(
        "Basic Scheduler",
        "Image Generation",
        "Build a SIGMAS schedule from a diffusion model",
    ),
    "sampler_select": TemplateMeta(
        "Sampler Select",
        "Image Generation",
        "Choose a sampler contract for advanced sampling",
    ),
    "noise_generator": TemplateMeta(
        "Noise Generator",
        "Image Generation",
        "Euler and RES4LYF noise types for advanced sampling",
    ),
    "cfg_guider": TemplateMeta(
        "CFG Guider",
        "Image Generation",
        "Bundle model, prompt conditioning and CFG into a guider",
    ),
    "sampler_custom_advanced": TemplateMeta(
        "Sampler Custom Advanced",
        "Image Generation",
        "Run custom SAMPLER, SIGMAS, NOISE and GUIDER contracts",
    ),
    "preview_sigmas": TemplateMeta(
        "Preview Sigmas",
        "Image Generation",
        "Inspect any custom SIGMAS schedule",
    ),
    "load_diffusion_model": TemplateMeta(
        "Load Diffusion Model",
        "Image Generation",
        "Load a diffusers repo or component with architecture metadata",
    ),
    "load_clip": TemplateMeta(
        "Load CLIP",
        "Image Generation",
        "Load a CLIP/T5/Qwen/Mistral text-encoder bundle",
    ),
    "load_clip_dual": TemplateMeta(
        "Load CLIP (Dual)",
        "Image Generation",
        "Combine CLIP-L + CLIP-G or CLIP + T5 component loaders",
    ),
    "load_vae": TemplateMeta(
        "Load VAE",
        "Image Generation",
        "Load a diffusers VAE with family-aware latent encoding",
    ),
    "vae_encode": TemplateMeta(
        "VAE Encode",
        "Image Generation",
        "Encode a pixel image into a LATENT for img2img",
    ),
    "image_scale": TemplateMeta(
        "Image Scale (Legacy)",
        "Computer Vision",
        "Legacy fixed-dimension image resize",
        hidden=True,
    ),
    "ksampler_advanced": TemplateMeta(
        "KSampler Advanced",
        "Image Generation",
        "Sample a configurable sigma step range with optional noise",
    ),
    "sampler_custom": TemplateMeta(
        "Sampler Custom",
        "Image Generation",
        "Run a sampler and SIGMAS contract with CFG",
    ),
    "resize_by_side": TemplateMeta(
        "Resize By Side (Legacy)",
        "Computer Vision",
        "Legacy side-length image resize",
        hidden=True,
    ),
    "upscale_image_by": TemplateMeta(
        "Upscale Image By (Legacy)",
        "Computer Vision",
        "Legacy image resize by multiplier",
        hidden=True,
    ),
    "resize_image_by_megapixels": TemplateMeta(
        "Resize Image By Megapixels (Legacy)",
        "Computer Vision",
        "Legacy image resize to a megapixel count",
        hidden=True,
    ),
    "upscale_latent": TemplateMeta(
        "Upscale Latent (Legacy)",
        "Image Generation",
        "Legacy fixed-dimension latent resize",
        hidden=True,
    ),
    "upscale_latent_by": TemplateMeta(
        "Upscale Latent By (Legacy)",
        "Image Generation",
        "Legacy latent resize by multiplier",
        hidden=True,
    ),
    "resize_latent_by_megapixels": TemplateMeta(
        "Resize Latent By Megapixels (Legacy)",
        "Image Generation",
        "Legacy latent resize to a megapixel count",
        hidden=True,
    ),
    "resize_image": TemplateMeta(
        "Resize Image",
        "Computer Vision",
        "Resize images by dimensions, side, megapixels or multiplier",
    ),
    "resize_latent": TemplateMeta(
        "Resize Latent",
        "Image Generation",
        "Resize latents by dimensions, megapixels or multiplier",
    ),
}


def template_meta(name: str) -> TemplateMeta:
    """Return metadata for ``name`` with a safe fallback for unknown files."""
    meta = TEMPLATE_META.get(name)
    if meta is not None:
        return meta
    return TemplateMeta(
        title=name.replace("_", " ").title(),
        category=FALLBACK_CATEGORY,
        description="",
    )
