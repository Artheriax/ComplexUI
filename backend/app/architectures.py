"""Implemented image-generation architecture metadata.

The node DSL is exported as standalone code, so the templates carry their own
small architecture maps. This module is the backend contract used by tests and
sampler adapters; template option lists and class mappings are checked against
it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ARCHITECTURES",
    "ARCHITECTURE_LITERALS",
    "ArchitectureSpec",
    "canonical_architecture",
    "choices_for_template",
]


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """Static metadata for one supported image architecture family."""

    id: str
    label: str
    aliases: tuple[str, ...]
    pipeline_classes: tuple[str, ...]
    transformer_classes: tuple[str, ...]
    text_encoder_classes: tuple[str, ...] = ()
    vae_classes: tuple[str, ...] = ()
    default_width: int = 1024
    default_height: int = 1024
    steps: int = 25
    cfg: float = 7.0
    sampler: str = "euler"
    scheduler: str = "simple"
    negative_prompt: str = "optional"
    guidance: str = "cfg"
    guidance_default: float | None = None
    latent_space: str = "sd"
    latent_channels: int = 4
    latent_packing: int = 1
    vae_scale_factor: int = 8
    supports_text_to_image: bool = True
    supports_img2img: bool = True
    supports_inpaint: bool = True
    supports_edit: bool = False
    max_references: int = 0
    custom_code: bool = False

    def recommended(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "cfg": self.cfg,
            "guidance": (
                self.cfg if self.guidance_default is None else self.guidance_default
            ),
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "width": self.default_width,
            "height": self.default_height,
        }

    def capabilities(self) -> dict[str, object]:
        return {
            "text_to_image": self.supports_text_to_image,
            "img2img": self.supports_img2img,
            "inpaint": self.supports_inpaint,
            "edit": self.supports_edit,
            "max_references": self.max_references,
            "negative_prompt": self.negative_prompt,
            "guidance": self.guidance,
            "guidance_default": (
                self.cfg if self.guidance_default is None else self.guidance_default
            ),
        }


_SPECS: tuple[ArchitectureSpec, ...] = (
    ArchitectureSpec(
        id="sd15",
        label="Stable Diffusion 1.5",
        aliases=("stable_diffusion", "sd15", "sd1.5", "sd_15"),
        pipeline_classes=(
            "StableDiffusionPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
        ),
        transformer_classes=("UNet2DConditionModel",),
        text_encoder_classes=("CLIPTextModel", "CLIPTokenizer"),
        vae_classes=("AutoencoderKL",),
        default_width=512,
        default_height=512,
        steps=25,
        cfg=7.0,
        sampler="dpmpp_2m",
        scheduler="karras",
        negative_prompt="required",
        guidance="cfg",
        latent_space="sd",
        latent_channels=4,
        vae_scale_factor=8,
        max_references=0,
    ),
    ArchitectureSpec(
        id="sdxl",
        label="Stable Diffusion XL",
        aliases=("sdxl", "sd_xl"),
        pipeline_classes=(
            "StableDiffusionXLPipeline",
            "StableDiffusionXLImg2ImgPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLModularPipeline",
        ),
        transformer_classes=("UNet2DConditionModel",),
        text_encoder_classes=(
            "CLIPTextModel",
            "CLIPTextModelWithProjection",
            "CLIPTokenizer",
        ),
        vae_classes=("AutoencoderKL",),
        default_width=1024,
        default_height=1024,
        steps=25,
        cfg=7.0,
        sampler="dpmpp_2m",
        scheduler="karras",
        negative_prompt="required",
        guidance="cfg",
        latent_space="sd",
        latent_channels=4,
        vae_scale_factor=8,
        max_references=0,
    ),
    ArchitectureSpec(
        id="flux1",
        label="FLUX.1",
        aliases=("flux", "flux1", "flux_1"),
        pipeline_classes=(
            "FluxPipeline",
            "FluxImg2ImgPipeline",
            "FluxInpaintPipeline",
            "FluxFillPipeline",
            "FluxKontextPipeline",
            "FluxKontextInpaintPipeline",
            "FluxControlPipeline",
            "FluxModularPipeline",
            "FluxKontextModularPipeline",
        ),
        transformer_classes=("FluxTransformer2DModel",),
        text_encoder_classes=("CLIPTextModel", "T5EncoderModel"),
        vae_classes=("AutoencoderKL",),
        default_width=1024,
        default_height=1024,
        steps=28,
        cfg=3.5,
        sampler="euler",
        scheduler="simple",
        negative_prompt="optional",
        guidance="true_cfg",
        guidance_default=3.5,
        latent_space="flux",
        latent_channels=16,
        latent_packing=2,
        vae_scale_factor=8,
        max_references=2,
        supports_edit=True,
    ),
    ArchitectureSpec(
        id="flux2",
        label="FLUX.2",
        aliases=("flux2", "flux_2"),
        pipeline_classes=(
            "Flux2Pipeline",
            "Flux2KleinPipeline",
            "Flux2ModularPipeline",
            "Flux2KleinModularPipeline",
            "Flux2KleinBaseModularPipeline",
        ),
        transformer_classes=("Flux2Transformer2DModel",),
        text_encoder_classes=(
            "Mistral3ForConditionalGeneration",
            "Qwen3ForCausalLM",
        ),
        vae_classes=("AutoencoderKLFlux2",),
        default_width=1024,
        default_height=1024,
        steps=25,
        cfg=1.0,
        sampler="euler",
        scheduler="simple",
        negative_prompt="none",
        guidance="distilled",
        guidance_default=4.0,
        latent_space="flux2",
        latent_channels=32,
        latent_packing=2,
        vae_scale_factor=8,
        max_references=10,
        supports_edit=True,
    ),
    ArchitectureSpec(
        id="flux3",
        label="FLUX.3",
        aliases=("flux3", "flux_3"),
        pipeline_classes=(
            "Flux3ModularPipeline",
            "Flux3DistilledModularPipeline",
        ),
        transformer_classes=("Flux3Transformer2DModel",),
        text_encoder_classes=("Qwen3VLForConditionalGeneration",),
        vae_classes=("AutoencoderKLFlux2",),
        default_width=1024,
        default_height=1024,
        steps=16,
        cfg=1.0,
        sampler="euler",
        scheduler="simple",
        negative_prompt="none",
        guidance="distilled",
        latent_space="flux3",
        latent_channels=32,
        latent_packing=2,
        vae_scale_factor=8,
        max_references=10,
        supports_edit=True,
    ),
    ArchitectureSpec(
        id="qwen_image",
        label="Qwen Image 2.1",
        aliases=("qwen_image", "qwenimage", "qwen_image_21", "qwen_image_2.1"),
        pipeline_classes=(
            "QwenImage21Pipeline",
            "QwenImage21ModularPipeline",
            "QwenImage21TurboModularPipeline",
            "QwenImagePipeline",
            "QwenImageModularPipeline",
            "QwenImageEditPlusPipeline",
            "QwenImageEditPipeline",
        ),
        transformer_classes=("QwenImage21Transformer2DModel", "QwenImageTransformer2DModel"),
        text_encoder_classes=("Qwen3VLForConditionalGeneration",),
        vae_classes=("AutoencoderKLQwenImage21", "AutoencoderKLQwenImage"),
        default_width=1024,
        default_height=1024,
        steps=20,
        cfg=4.0,
        sampler="euler",
        scheduler="simple",
        negative_prompt="optional",
        guidance="true_cfg",
        latent_space="qwen",
        latent_channels=64,
        vae_scale_factor=16,
        max_references=10,
        supports_edit=True,
    ),
    ArchitectureSpec(
        id="z_image",
        label="Z-Image",
        aliases=("z_image", "zimage", "z-image"),
        pipeline_classes=(
            "ZImagePipeline",
            "ZImageModularPipeline",
            "ZImageImg2ImgPipeline",
            "ZImageInpaintPipeline",
            "ZImageControlNetPipeline",
            "ZImageControlNetInpaintPipeline",
        ),
        transformer_classes=("ZImageTransformer2DModel",),
        text_encoder_classes=("Qwen3Model", "Qwen2Tokenizer"),
        vae_classes=("AutoencoderKL",),
        default_width=1024,
        default_height=1024,
        steps=30,
        cfg=5.0,
        sampler="euler",
        scheduler="simple",
        negative_prompt="optional",
        guidance="cfg",
        latent_space="z_image",
        latent_channels=16,
        vae_scale_factor=8,
        max_references=0,
    ),
    ArchitectureSpec(
        id="krea2",
        label="Krea 2",
        aliases=("krea2", "krea_2"),
        pipeline_classes=(
            "Krea2Pipeline",
            "Krea2ModularPipeline",
            "Krea2TurboModularPipeline",
        ),
        transformer_classes=("Krea2Transformer2DModel",),
        text_encoder_classes=("Qwen3VLModel",),
        vae_classes=("AutoencoderKLQwenImage",),
        default_width=1024,
        default_height=1024,
        steps=25,
        cfg=4.5,
        sampler="euler",
        scheduler="simple",
        negative_prompt="optional",
        guidance="cfg",
        latent_space="krea2",
        latent_channels=16,
        latent_packing=2,
        vae_scale_factor=8,
        max_references=0,
        supports_edit=False,
    ),
    ArchitectureSpec(
        id="ideogram4",
        label="Ideogram 4.0",
        aliases=("ideogram4", "ideogram_4"),
        pipeline_classes=("Ideogram4Pipeline", "Ideogram4ModularPipeline"),
        transformer_classes=("Ideogram4Transformer2DModel",),
        text_encoder_classes=("Qwen3VLModel", "Qwen2Tokenizer"),
        vae_classes=("AutoencoderKLFlux2",),
        default_width=2048,
        default_height=2048,
        steps=20,
        cfg=7.0,
        sampler="euler",
        scheduler="simple",
        negative_prompt="none",
        guidance="dual_branch",
        latent_space="ideogram4",
        latent_channels=32,
        latent_packing=2,
        vae_scale_factor=8,
        max_references=0,
    ),
    ArchitectureSpec(
        id="fibo2",
        label="Fibo 2.0",
        aliases=("fibo2", "fibo_2", "bria_fibo2"),
        pipeline_classes=("BriaFibo2Pipeline", "BriaFiboEditPipeline"),
        transformer_classes=("BriaFibo2Transformer2DModel",),
        text_encoder_classes=("Qwen3VLForConditionalGeneration",),
        vae_classes=("AutoencoderKLFlux2",),
        default_width=1024,
        default_height=1024,
        steps=25,
        cfg=5.0,
        sampler="euler",
        scheduler="simple",
        negative_prompt="optional",
        guidance="cfg",
        latent_space="fibo2",
        latent_channels=32,
        latent_packing=2,
        vae_scale_factor=8,
        max_references=5,
        supports_edit=True,
    ),
    ArchitectureSpec(
        id="anima",
        label="Anima",
        aliases=("anima", "anima_base"),
        pipeline_classes=("AnimaModularPipeline",),
        transformer_classes=("CosmosTransformer3DModel",),
        text_encoder_classes=("Qwen3Model", "Qwen2Tokenizer", "T5Tokenizer"),
        vae_classes=("AutoencoderKLQwenImage",),
        default_width=1024,
        default_height=1024,
        steps=25,
        cfg=4.0,
        sampler="euler",
        scheduler="simple",
        negative_prompt="required",
        guidance="cfg",
        latent_space="anima",
        latent_channels=16,
        vae_scale_factor=8,
        max_references=0,
    ),
    ArchitectureSpec(
        id="moss_image",
        label="MOSS Image",
        aliases=("moss_image", "moss"),
        pipeline_classes=("MOSSImagePipeline",),
        transformer_classes=("MOSSImageTransformer",),
        text_encoder_classes=(
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5Model",
        ),
        vae_classes=("AutoencoderKLFlux2",),
        default_width=1024,
        default_height=1024,
        steps=50,
        cfg=3.0,
        sampler="euler",
        scheduler="simple",
        negative_prompt="optional",
        guidance="cfg",
        latent_space="moss",
        latent_channels=32,
        latent_packing=2,
        vae_scale_factor=8,
        custom_code=True,
    ),
)

ARCHITECTURES: dict[str, ArchitectureSpec] = {spec.id: spec for spec in _SPECS}

ARCHITECTURE_LITERALS: tuple[str, ...] = (
    "auto",
    "stable_diffusion",
    "sd15",
    "sdxl",
    "flux",
    "flux1",
    "flux2",
    "flux3",
    "qwen_image",
    "z_image",
    "krea2",
    "ideogram4",
    "fibo2",
    "anima",
    "moss_image",
)

_ALIASES: dict[str, str] = {}
for _spec in _SPECS:
    _ALIASES[_spec.id] = _spec.id
    for _alias in _spec.aliases:
        _ALIASES[_alias] = _spec.id


def canonical_architecture(value: str) -> str | None:
    """Return the canonical id for a template dropdown value."""
    cleaned = value.strip().lower()
    if cleaned == "auto":
        return None
    return _ALIASES.get(cleaned)


def choices_for_template() -> tuple[str, ...]:
    """Return the shared dropdown choices (kept in template Literal order)."""
    return ARCHITECTURE_LITERALS
