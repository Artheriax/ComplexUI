# /// script
# dependencies = ["torch>=2.2", "transformers>=4.46"]
# ///
import torch

clip: CLIP
text: str = ""

if not isinstance(clip, dict):
    raise TypeError("clip must be a CLIP wrapper")
architecture = str(clip.get("arch", ""))
device = torch.device(str(clip.get("device", "cpu")))
dtype = clip.get("dtype", torch.float32)
text_encoder = clip.get("text_encoder")
tokenizer = clip.get("tokenizer")
if text_encoder is None or tokenizer is None:
    raise ValueError("CLIP wrapper is missing its text encoder or tokenizer")


def encode_tokens(encoder, current_tokenizer, prompt: str):
    tokens = current_tokenizer(
        prompt,
        padding="max_length",
        max_length=int(current_tokenizer.model_max_length),
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    with torch.inference_mode():
        output = encoder(input_ids, output_hidden_states=True)
    return output.hidden_states[-2], output


with torch.inference_mode():
    cond, output = encode_tokens(text_encoder, tokenizer, text)
    pooled = None
    if architecture == "sdxl":
        text_encoder_2 = clip.get("text_encoder_2")
        tokenizer_2 = clip.get("tokenizer_2")
        if text_encoder_2 is None or tokenizer_2 is None:
            raise ValueError("SDXL CLIP wrapper is missing its second encoder")
        cond_2, output_2 = encode_tokens(text_encoder_2, tokenizer_2, text)
        cond = torch.cat([cond, cond_2], dim=-1)
        pooled = output_2[0]
    elif architecture != "sd15":
        raise ValueError(f"Unsupported CLIP architecture: {architecture}")
    cond = cond.to(dtype=dtype).detach().cpu()
    if pooled is not None:
        pooled = pooled.to(dtype=dtype).detach().cpu()

conditioning = {
    "kind": "conditioning",
    "arch": architecture,
    "cond": cond,
    "pooled": pooled,
    "text": text,
}

return conditioning: CONDITIONING
